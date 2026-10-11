"""rag_lite.py - a phone-friendly RAG: no PyTorch, no installs, Python standard library only.

What it does
  1. Cuts the FastAPI docs into chunks (using chunker_v2.py).
  2. Finds the chunks that best match your question using KEYWORD search (BM25).
  3. Builds a ready-to-paste prompt: your question + the best chunks.
     You paste that prompt into any chatbot. (That paste step is the "G" in RAG.)

Honest limit: keyword search matches WORDS, not MEANING. "make it optional" will not
match a chunk that only says "default to None" unless some words overlap. The embedding
version (index.py / search.py) fixes that, and needs a machine that can run PyTorch.

Run (from the folder that contains chunker_v2.py and the cloned fastapi/ folder):
    python rag_lite.py
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

from chunker_v2 import chunk_file, find_docs

TOP_K = 3

STOPWORDS = set("""a an and are as at be by can do does for from how i in is it its of on or
that the this to was what when where which who why will with you your me my we show give""".split())


def tokenize(text):
    """lowercase words, minus stopwords, with a tiny plural-stripper (queries -> query)."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")):   # isn't -> isnt, not isn + t
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    """Keyword scoring. A word scores higher if it is rare overall and frequent in this chunk."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(docs)
        df = Counter(w for d in self.tf for w in d)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query_tokens):
        result = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for w in set(query_tokens):
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            result.append(s)
        return result


def build_prompt(question, picked):
    context = "\n\n---\n\n".join(c["text"] for c in picked)
    return (
        "Answer the question using ONLY the text below.\n"
        'If the text does not contain the answer, say "I don\'t know."\n\n'
        f"Text:\n{context}\n\nQuestion: {question}"
    )


def main():
    DOCS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fastapi/docs/en/docs/tutorial")
    REPO = sys.argv[2] if len(sys.argv) > 2 else "fastapi"   # code pointers are resolved from here
    chunks = [c for p in find_docs(DOCS) for c in chunk_file(p, REPO)]
    if not chunks:
        sys.exit(f"No chunks found. Is '{DOCS}' the right folder? (Run this from the folder that contains fastapi/)")
    # the heading is counted twice, so a chunk whose TITLE matches your question ranks higher
    index = BM25([tokenize(c["heading"] + " " + c["heading"] + " " + c["text"]) for c in chunks])
    print(f"{len(chunks)} chunks ready. Type a question (or press Enter on an empty line to quit).")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not question:
            break
        q = tokenize(question)
        scores = index.scores(q)
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        order = [i for i in order if scores[i] > 0]
        if not order:
            print("No chunk shares any word with that question. Try different words.")
            continue
        print(f"(searched for: {', '.join(sorted(set(q)))})")
        for rank, i in enumerate(order, 1):
            c = chunks[i]
            kind = "has code" if "```" in c["text"] else "text only"
            print(f"\n#{rank}  score {scores[i]:.1f}  [{kind}]  {Path(c['source']).name}")
            print(f"    {c['heading']}")
            preview = c["text"][len(c["heading"]):].strip().replace("\n", " ")
            print(f"    {preview[:220]}{'...' if len(preview) > 220 else ''}")
        prompt = build_prompt(question, [chunks[i] for i in order])
        Path("last_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"\nPrompt saved to last_prompt.txt ({len(prompt.split())} words). Paste it into a chatbot.")
        print("Show it here with:  cat last_prompt.txt")


if __name__ == "__main__":
    main()
