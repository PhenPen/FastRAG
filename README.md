# FastAPI docs RAG (a learning project)

Ask questions about the FastAPI tutorial and get answers grounded in the docs themselves.
Built from scratch (no LangChain / LlamaIndex) so every step is visible.

```
INDEXING (once)
  FastAPI docs --> chunker_v2.py --> chunks --> embedding model --> ChromaDB (folder rag_db/)

QUERYING (every question)
  question --> embedding model --> closest chunks --> prompt --> an AI writes the answer
```

**Retrieval** finds the right text. **Generation** writes the answer from it. Most RAG failures
are retrieval failures, which is why `eval.py` tests retrieval on its own.

## Files

| File | What it does |
|---|---|
| `chunker_v2.py` | Cuts the markdown docs into chunks. Keeps code blocks whole, puts the heading trail on every chunk, and pastes in the real code that the docs only point to (`{* ... *}`). |
| `config.py` | Model name, database folder, collection name. Shared on purpose, so indexing and searching can never use different models. |
| `index.py` | Chunks the docs, embeds every chunk, stores them in ChromaDB. Run once. |
| `search.py` | Finds the closest chunks for a question and writes a ready-to-paste prompt. |
| `ask.py` | Same, then gets an answer: pasted by you into a chatbot (default) or from a local Ollama model. API functionality later |
| `eval.py` + `questions.json` | A retrieval test: is the answer inside one of the top chunks? |
| `rag_lite.py` | Phone-friendly version: keyword search, standard library only. Also the baseline to beat. |

## Setup

Needs Python 3.10 or newer and git. Run these from the folder that holds these files.

```bash
git clone --depth 1 https://github.com/fastapi/fastapi     # the docs (a clone, not part of this repo)
python -m venv .venv

# Windows PowerShell:   .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
# (PowerShell refusing to run the script? Try: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

pip install -r requirements.txt
```

`sentence-transformers` brings PyTorch with it, which is a large download.

## Run

```bash
python index.py                      # downloads the ~90 MB model on first run, then embeds all chunks
python search.py "How do I make a query parameter optional?"
python eval.py
python ask.py "How do I make a query parameter optional?"
```

`index.py` also prints how many chunks are longer than the model can read (its limit is 256 tokens;
the rest of a longer chunk is silently ignored when searching).

## Measure it

`rag_lite.py` (keyword search) on the 7 answerable questions in `questions.json`:
**answer in the top 1 chunk: 2/7, in the top 3: 3/7.**

Your embedding version: top 1 ___ / 7, top 3 ___ / 7

Look especially at "How do I stop a URL parameter from being mandatory?". The answer is in the
docs, but not in those words.

## Getting an answer without pasting (optional)

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
python ask.py --ollama "How do I make a query parameter optional?"
```

Set `OLLAMA_MODEL` to use a different model. Answer quality depends on the model and your hardware.

## Known limits

- **Only the tutorial is indexed by default.** To index more:
  `python index.py fastapi/docs/en/docs fastapi` (release notes and similar pages are skipped).
- **Similar does not mean correct.** Search finds text *about* your question, not text that
  *answers* it. `eval.py` includes a question that is not in the docs ("Who created FastAPI?").
  Check that the best match for it scores low, or at least that the AI says "I don't know".
- **The model must match.** If you change `MODEL_NAME` in `config.py`, run `index.py` again.
- **Long code chunks** can exceed the model's 256-token window, so only their beginning is searchable.
- **List continuations:** a paragraph indented under a list item can end up in a different chunk
  from its item (blank-line splitting).
- **Ollama context:** `ask.py` asks for an 8192 token window so the prompt is not cut. That needs more memory.


## Next up

1. Wrap `ask.py` in a small FastAPI endpoint (`POST /ask`).
2. Hybrid search: combine keyword and embedding scores.
3. Merge very small chunks into a neighbour.
4. Replace the manual paste with a hosted LLM API 
