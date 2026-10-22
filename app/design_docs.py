"""
The Design & Architecture tab: markdown + mermaid documents, served live.

The documents live in `design/*.md` and are parsed into blocks here. The
principle borrowed from the AI-Agents-Patterns dashboard is that nothing
is a screenshot: every diagram is Mermaid source stored with the prose it
explains, and it is rendered at view time. A diagram that has changed
cannot be mistaken for a correct one.

The parser is deliberately small and total: it recognises exactly the
block types the documents use, and it fails loudly on anything else, so a
document cannot silently lose a diagram.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DESIGN_DIR = Path(__file__).resolve().parent.parent / "design"

# Block types the renderer understands. The parser is strict about these so
# that an unsupported construct is a visible error rather than a document
# that quietly renders wrong.
KNOWN_FENCE_LANGUAGES = {"mermaid", "python", "bash", "json", "text"}


def list_documents() -> List[Dict[str, Any]]:
    """Every design document, in order, with its title and outline."""
    docs = []
    for path in sorted(DESIGN_DIR.glob("*.md")):
        text = path.read_text()
        title = _first_heading(text) or path.stem
        docs.append({
            "id": path.stem,
            "file": path.name,
            "title": title,
            "summary": _first_paragraph(text),
            "outline": _outline(text),
            "blocks": parse_markdown(text),
        })
    return docs


def document(doc_id: str) -> Optional[Dict[str, Any]]:
    for doc in list_documents():
        if doc["id"] == doc_id:
            return doc
    return None


# ------------------------------------------------------------------ parsing

def _first_heading(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def _first_paragraph(text: str) -> str:
    """The first substantial paragraph, for the document list."""
    body = re.sub(r"^#.*$", "", text, flags=re.M)
    for chunk in re.split(r"\n\s*\n", body):
        c = chunk.strip()
        if not c or c.startswith("```") or c.startswith("|") or c.startswith("-"):
            continue
        if c.startswith("*") or c.startswith(">"):
            continue
        return c
    return ""


def _outline(text: str) -> List[Dict[str, str]]:
    """The headings of a document, for the in-page outline."""
    out = []
    for m in re.finditer(r"^(#{2,3})\s+(.+)$", text, re.M):
        level = len(m.group(1))
        out.append({"level": str(level), "title": m.group(2).strip(),
                    "id": _slug(m.group(2))})
    return out


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def parse_markdown(text: str) -> List[Dict[str, Any]]:
    """
    Parse a document into blocks.

    Block kinds: heading, paragraph, list, code (with an optional language),
    mermaid, table, quote, rule.
    """
    blocks: List[Dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # fenced code / mermaid
        fence = re.match(r"^```(\w*)\s*$", line)
        if fence:
            lang = fence.group(1) or "text"
            if lang not in KNOWN_FENCE_LANGUAGES:
                raise ValueError(f"unsupported fence language '{lang}'")
            body: List[str] = []
            i += 1
            while i < n and not re.match(r"^```\s*$", lines[i]):
                body.append(lines[i])
                i += 1
            if i >= n:
                raise ValueError("unterminated code fence")
            i += 1  # closing fence
            src = "\n".join(body)
            if lang == "mermaid":
                blocks.append({"kind": "mermaid", "source": src})
            else:
                blocks.append({"kind": "code", "language": lang, "source": src})
            continue

        # heading
        h = re.match(r"^(#{1,4})\s+(.+)$", line)
        if h:
            level = len(h.group(1))
            title = h.group(2).strip()
            blocks.append({"kind": "heading", "level": level, "title": title,
                           "id": _slug(title)})
            i += 1
            continue

        # horizontal rule
        if re.match(r"^\s*---\s*$", line):
            blocks.append({"kind": "rule"})
            i += 1
            continue

        # table
        if line.strip().startswith("|") and i + 1 < n and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = _split_row(line)
            i += 2
            body_rows = []
            while i < n and lines[i].strip().startswith("|"):
                body_rows.append(_split_row(lines[i]))
                i += 1
            blocks.append({"kind": "table", "header": header, "rows": body_rows})
            continue

        # list
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items: List[Dict[str, Any]] = []
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            while i < n and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                raw = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i])
                # continuation lines are indented and belong to the item
                i += 1
                while i < n and lines[i].strip() and \
                        not re.match(r"^\s*([-*+]|\d+\.|#{1,4}\s|\||```)", lines[i]) and \
                        lines[i].startswith((" ", "\t")):
                    raw += " " + lines[i].strip()
                    i += 1
                items.append({"text": raw})
            blocks.append({"kind": "list", "ordered": ordered, "items": items})
            continue

        # block quote
        if line.strip().startswith(">"):
            quoted = []
            while i < n and lines[i].strip().startswith(">"):
                quoted.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append({"kind": "quote", "text": " ".join(q for q in quoted if q)})
            continue

        # paragraph
        if line.strip():
            para: List[str] = []
            while i < n and lines[i].strip() and \
                    not re.match(r"^\s*(#{1,4}\s|```|\||>|---)", lines[i]) and \
                    not re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                para.append(lines[i].strip())
                i += 1
            if para:
                blocks.append({"kind": "paragraph", "text": " ".join(para)})
            continue

        i += 1
    return blocks


def _split_row(line: str) -> List[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def diagram_stats() -> Dict[str, Any]:
    """How much diagram the documentation carries — for the tab's header."""
    docs = list_documents()
    diagrams = 0
    kinds: Dict[str, int] = {}
    for d in docs:
        for b in d["blocks"]:
            kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
            if b["kind"] == "mermaid":
                diagrams += 1
    return {"documents": len(docs), "diagrams": diagrams, "blocks": kinds}
