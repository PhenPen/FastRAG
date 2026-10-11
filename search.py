"""Step 2 of the embedding RAG: ask a question, get the closest chunks and a ready-to-paste prompt.

    python search.py                       # interactive: type questions, empty line to quit
    python search.py "How do I make a query parameter optional?"     # one question
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, DB_PATH, COLLECTION

TOP_K = 3

model = SentenceTransformer(MODEL_NAME)
collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)


def retrieve(question, k=TOP_K):
    """Return the k closest chunks as dicts: text, heading, source, score (1 = identical meaning)."""
    vector = model.encode([question]).tolist()
    r = collection.query(query_embeddings=vector, n_results=k)
    return [
        {"text": doc, "heading": meta["heading"], "source": meta["source"], "score": 1 - dist}
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


def build_prompt(question, hits):
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    return (
        "Answer the question using ONLY the text below.\n"
        'If the text does not contain the answer, say "I don\'t know."\n\n'
        f"Text:\n{context}\n\nQuestion: {question}"
    )


def show(question):
    hits = retrieve(question)
    for rank, h in enumerate(hits, 1):
        kind = "has code" if "```" in h["text"] else "text only"
        print(f"\n#{rank}  similarity {h['score']:.2f}  [{kind}]  {Path(h['source']).name}")
        print(f"    {h['heading']}")
        preview = h["text"][len(h["heading"]):].strip().replace("\n", " ")
        print(f"    {preview[:220]}{'...' if len(preview) > 220 else ''}")
    prompt = build_prompt(question, hits)
    Path("last_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"\nPrompt saved to last_prompt.txt ({len(prompt.split())} words). Paste it into a chatbot.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        print("Type a question (empty line to quit).")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except EOFError:
                break
            if not q:
                break
            show(q)
