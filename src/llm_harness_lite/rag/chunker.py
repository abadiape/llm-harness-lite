"""Deterministic char-based chunker with overlap. Swap for token-based later."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    meta: dict


def chunk_text(text: str, source: str = "doc", size: int = 600,
               overlap: int = 80) -> list[Chunk]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("chunk size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError("chunk overlap must be a non-negative integer smaller than size")
    text = " ".join(text.split())
    out, i, n = [], 0, 0
    while i < len(text):
        piece = text[i:i + size]
        out.append(Chunk(text=piece, meta={"source": source, "n": n}))
        n += 1
        i += size - overlap
    return out
