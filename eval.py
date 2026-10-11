"""Step 3 of the embedding RAG: a retrieval test.

For every question in questions.json, check whether the expected text appears (all of it, together)
inside one of the top-K chunks that search returns. This tests the FINDING half only: it says
nothing about how well an AI writes the answer.

    python eval.py          # top 3 chunks per question
    python eval.py 5        # top 5

A MISS means one of two things, and you have to look to tell which:
  - retrieval failed (the right chunk exists but ranked too low), or
  - the chunker split the answer across chunks (so no single chunk has all of it).
"""
import json
import sys

if hasattr(sys.stdout, "reconfigure"):          # Windows consoles can choke on emoji in the docs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from search import retrieve                     # loads the model and the database

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
with open("questions.json", encoding="utf-8") as f:
    questions = json.load(f)

in_top1 = in_topk = scored = 0
for q in questions:
    results = retrieve(q["question"], K)
    best = results[0]
    if not q["expect"]:                         # unanswerable question: nothing can be a "hit"
        print(f"?     {q['question']}")
        print(f"      not in the docs. Best match anyway: {best['score']:.2f}  {best['heading']}")
        continue
    scored += 1
    rank = next((i for i, r in enumerate(results, 1)
                 if all(e in r["text"] for e in q["expect"])), None)
    if rank:
        in_topk += 1
        in_top1 += rank == 1
        print(f"HIT   (rank {rank}, similarity {results[rank - 1]['score']:.2f})  {q['question']}")
    else:
        print(f"MISS  {q['question']}")
        print(f"      top result was: {best['score']:.2f}  {best['heading']}")

print(f"\nAnswer in the top 1 chunk: {in_top1}/{scored}")
print(f"Answer in the top {K} chunks: {in_topk}/{scored}")
