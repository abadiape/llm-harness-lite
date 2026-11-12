"""Deterministic char-based chunker with overlap. Swap for token-based later."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    meta: dict

def chunk_text(text: str, source: str = "doc", size: int = 600,
               overlap: int = 80) -> list[Chunk]:
    text = " ".join(text.split())
    out, i, n = [], 0, 0
    while i < len(text):
        piece = text[i:i + size]
        out.append(Chunk(text=piece, meta={"source": source, "n": n}))
        n += 1
        i += max(1, size - overlap)
    return out
