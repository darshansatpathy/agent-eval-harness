"""The knowledge base: Markdown articles with front matter, split into paragraphs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

WORD = re.compile(r"[a-z0-9$%][a-z0-9$%.\-]*")
STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "we",
        "our",
        "me",
        "there",
        "their",
        "them",
        "they",
        "than",
        "then",
    ]
)


@dataclass(frozen=True)
class Chunk:
    doc: str
    title: str
    text: str


def tokens(text: str) -> list[str]:
    out = []
    for w in WORD.findall(text.lower()):
        w = w.rstrip(".")
        if w in STOP:
            continue
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]  # deleting -> delet
        elif len(w) > 5 and w.endswith("ed"):
            w = w[:-2]  # failed -> fail
        elif len(w) > 4 and w.endswith(("xes", "ches", "shes")):
            w = w[:-2]  # taxes -> tax
        elif len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us")):
            w = w[:-1]  # notebooks -> notebook
        out.append(w)
    return out


def load(directory: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text()
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front)
        for para in (p.strip() for p in body.strip().split("\n\n")):
            if para:
                chunks.append(Chunk(meta["id"], meta["title"], para))
    return chunks


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z$0-9])", text) if s.strip()]
