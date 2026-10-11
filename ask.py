"""Step 4 of the embedding RAG: the whole thing in one command.

    python ask.py "How do I make a query parameter optional?"

It finds the closest chunks (search.py), builds the prompt, and then someone writes the answer.
Who writes it is your choice:

    python ask.py "..."            default: saves the prompt to last_prompt.txt, you paste it into
                                   any chatbot yourself. Free, nothing to set up.
    python ask.py --ollama "..."   sends the prompt to a model running on your own PC
                                   (https://ollama.com). Free and private, but needs a PC that can
                                   run the model.

Ollama settings (environment variables, both optional):
    OLLAMA_MODEL   which model to use          (default: llama3.2, run `ollama list` to see yours)
    OLLAMA_URL     where Ollama is listening   (default: http://localhost:11434)
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):          # Windows consoles can choke on emoji in the docs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from search import build_prompt, retrieve       # loads the model and the database

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def call_ollama(prompt):
    """Send the prompt to a local Ollama server and return its answer text."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Ollama's default context window can be shorter than our prompt. A prompt that does not
        # fit gets cut, and the part at risk is the START, where the "use ONLY this text" rule is.
        "options": {"num_ctx": 8192},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)["response"]
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Ollama at {OLLAMA_URL} ({e.reason}).\n"
                 f"Is it running, and have you pulled the model?  ollama pull {OLLAMA_MODEL}")


def main():
    args = sys.argv[1:]
    use_ollama = "--ollama" in args or os.environ.get("LLM_BACKEND") == "ollama"
    question = " ".join(a for a in args if a != "--ollama").strip()
    if not question:
        sys.exit('Usage: python ask.py [--ollama] "your question"')

    hits = retrieve(question)
    print("Chunks used:")
    for rank, h in enumerate(hits, 1):
        print(f"  #{rank}  {h['score']:.2f}  {h['heading']}  ({Path(h['source']).name})")

    prompt = build_prompt(question, hits)
    if use_ollama:
        print(f"\nAsking {OLLAMA_MODEL} ...\n")
        print(call_ollama(prompt))
    else:
        Path("last_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"\nPrompt saved to last_prompt.txt ({len(prompt.split())} words). Paste it into a chatbot.")


if __name__ == "__main__":
    main()
