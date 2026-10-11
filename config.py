"""Shared settings. index.py and search.py both import these, so they can never disagree.

The most important one is MODEL_NAME: the documents and the questions must be embedded by the
SAME model. Vectors from different models live on different "maps", and mixing them fails
silently (plausible-looking but meaningless results). If you change it, re-run index.py.
"""
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "rag_db"
COLLECTION = "fastapi_docs"
