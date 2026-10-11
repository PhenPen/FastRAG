"""Stage 2 (v2): split FastAPI markdown docs into chunks, with code examples restored.

Changes from chunker.py:
  1. {* path ln[a:b] *} pointers are replaced by the real code from that file.
  2. '#' comment lines inside code blocks are no longer mistaken for headings.
  3. Over long text blocks (big bullet lists) are split between lines. Code is never split.

Usage:  python chunker_v2.py <docs folder> <repo root>
"""
import re
from pathlib import Path

MAX_WORDS = 150  # all-MiniLM-L6-v2 reads ~256 tokens, so keep text chunks under that

ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}")          # "# Title { #title }" -> "# Title"
INCLUDE = re.compile(r"^\{\*\s*(\S+)(.*?)\*\}\s*$")  # "{* ../../docs_src/x.py ln[1:7] hl[7] *}"
LINES = re.compile(r"ln\[(\d+):(\d+)\]")          # ln[1:7] = show only lines 1 to 7
HEADING = re.compile(r"^(#{1,6})\s+(.*)")
ADMONITION = re.compile(r"^/{3,4}\s*(\w*).*$")   # "/// tip", "/// warning", "///", "//// tab | X"

# Pages that are not "how do I use FastAPI" documentation. You can add unwanted documentation here to filter out noise
SKIP_FILES = {
    "release-notes.md",      # ~50k words of version numbers
    "fastapi-people.md", "management.md", "external-links.md",
    "translation-banner.md", "newsletter.md",
}
ADMONITION_LABELS = {"tip", "warning", "note", "info", "check", "danger"}


def find_docs(folder):
    """All markdown files under folder, minus test files (_name.md) and SKIP_FILES."""
    return [p for p in sorted(Path(folder).rglob("*.md"))
            if p.name not in SKIP_FILES and not p.name.startswith("_")]


def load_code(pointer_path, options, repo_root):
    """Read the file a pointer refers to and return it as a fenced code block."""
    rel = pointer_path
    while rel.startswith("../"):                   # pointers are relative to the repo root
        rel = rel[3:]
    root = Path(repo_root).resolve()
    file = (root / rel).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        return None                                # never read outside the repo (Security flaw, if allowed could access files outside specified directory)
    code = file.read_text(encoding="utf-8").splitlines()
    m = LINES.search(options)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        code = code[start - 1:end]                 # line numbers start at 1
    return f"```{file.suffix.lstrip('.')}\n" + "\n".join(code) + "\n```"


def split_into_blocks(text):
    """Turn text into blocks: paragraphs, or whole fenced code blocks."""
    blocks, current, in_code = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            current.append(line)
            if in_code:                            # closing fence: block is done
                blocks.append("\n".join(current))
                current = []
            in_code = not in_code
        elif in_code:
            current.append(line)                   # inside code: never split
        elif line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_long(block, max_words):
    """Split an over-long TEXT block between lines. Code and single lines stay whole."""
    if block.startswith("```") or "\n" not in block or len(block.split()) <= max_words:
        return [block]
    # Group lines so an indented line (a sub bullet or continuation) stays with its parent.
    # A sub bullet is just indentation /n + tab in some cases
    units = []
    for line in block.splitlines():
        if units and line[:1] in (" ", "\t"):
            units[-1] += "\n" + line
        else:
            units.append(line)
    if len(units) == 1:                              # one giant item: fall back to single lines
        units = block.splitlines()
    return pack(units, max_words, sep="\n")


def pack(blocks, max_words, sep="\n\n"):
    """Group blocks into chunks without going over max_words."""
    chunks, current, count = [], [], 0
    for block in blocks:
        for piece in split_long(block, max_words):
            n = len(piece.split())
            if current and count + n > max_words:
                chunks.append(sep.join(current))
                current, count = [], 0
            current.append(piece)
            count += n
    if current:
        chunks.append(sep.join(current))
    return chunks


def split_into_sections(path, repo_root):
    """Job A: cut one file into (heading trail, body) sections."""
    sections, stack, lines, in_code = [], [], [], False

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("```"):          # track code blocks so their '#' lines
            in_code = not in_code                  # are not read as headings
            lines.append(raw)
            continue
        if in_code:
            lines.append(raw)
            continue
        line = ANCHOR.sub("", raw)
        adm = ADMONITION.match(line)
        if adm:                                    # keep "Tip:" style labels, drop the rest
            label = adm.group(1).lower()
            if label in ADMONITION_LABELS:
                lines.append(f"{label.capitalize()}:")
            continue
        inc = INCLUDE.match(line)
        if inc:
            code = load_code(inc.group(1), inc.group(2), repo_root)
            if code:
                lines.append(code)
            continue
        m = HEADING.match(line)
        if m:
            flush()
            lines = []
            level = len(m.group(1))
            stack = [(l, t) for l, t in stack if l < level]
            stack.append((level, m.group(2).strip()))
        else:
            lines.append(line)
    flush()
    return sections


def chunk_file(path, repo_root):
    """Job B: turn each section into one or more chunks."""
    chunks = []
    for heading, body in split_into_sections(path, repo_root):
        for piece in pack(split_into_blocks(body), MAX_WORDS):
            chunks.append({
                "source": str(path),
                "heading": heading,
                "text": f"{heading}\n\n{piece}" if heading else piece,
            })
    return chunks


if __name__ == "__main__":
    import sys
    docs, repo_root = Path(sys.argv[1]), sys.argv[2]
    all_chunks = []
    for md in find_docs(docs):
        all_chunks.extend(chunk_file(md, repo_root))
    sizes = sorted(len(c["text"].split()) for c in all_chunks)
    print(f"{len(all_chunks)} chunks")
    print(f"median {sizes[len(sizes)//2]} words, smallest {sizes[0]}, largest {sizes[-1]}")
    print("with code:", sum("```" in c["text"] for c in all_chunks),
          "| under 20 words:", sum(s < 20 for s in sizes),
          "| over 180 words:", sum(s > 180 for s in sizes))
