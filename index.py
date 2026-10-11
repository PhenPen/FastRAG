"""Step 1 of the embedding RAG: chunk the docs, embed every chunk, store them in ChromaDB.

Run once (and again whenever you change the chunker or the docs):
    python index.py
or with explicit paths:
    python index.py <docs folder> <repo root>
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunker_v2 import chunk_file, find_docs
from config import MODEL_NAME, DB_PATH, COLLECTION   

DOCS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fastapi/docs/en/docs/tutorial")
REPO = sys.argv[2] if len(sys.argv) > 2 else "fastapi"   # code pointers are resolved from here

# 1. Cut the docs into chunks (see chunker_v2.py)
chunks = [c for md in find_docs(DOCS) for c in chunk_file(md, REPO)]
if not chunks:
    sys.exit(f"No chunks found under '{DOCS}'. Did you run: git clone --depth 1 https://github.com/fastapi/fastapi ?")
texts = [c["text"] for c in chunks]
print(f"{len(chunks)} chunks to embed")

# 2. Load the model, and measure something we could only guess on the phone:
#    how many chunks are longer than the model can read? (It silently ignores the rest.)
model = SentenceTransformer(MODEL_NAME)
try:
    limit = model.max_seq_length
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    too_long = sum(n > limit for n in lengths)
    print(f"Model reads at most {limit} tokens per chunk. {too_long} of {len(texts)} chunks "
          f"({too_long * 100 // len(texts)}%) are longer, so their tails are not searchable.")
except Exception as e:                   # a measuring problem must never stop the indexing
    print("(could not measure token lengths:", type(e).__name__, ")")

# 3. Turn every chunk into a vector (384 numbers each). This is the slow part.
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# 4. Store text + vector + origin info in a persistent database (the folder rag_db)
client = chromadb.PersistentClient(path=DB_PATH)
try:
    client.delete_collection(COLLECTION)   # we start the collection clean, so no stale chunks survive a re run
except Exception:
    pass                                   # first run: nothing to delete
collection = client.create_collection(
    COLLECTION,
    metadata={"hnsw:space": "cosine"},     # compare vectors by angle, like cos_sim
)
BATCH = 500                                # keep each add() call comfortably small
for start in range(0, len(chunks), BATCH):
    part = slice(start, start + BATCH)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))][part],
        documents=texts[part],
        embeddings=embeddings[part].tolist(),
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in chunks[part]],
    )
print("stored", collection.count(), "chunks in", DB_PATH)
