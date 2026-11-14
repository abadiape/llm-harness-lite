"""Shared RAG contract for the supported pgvector and Qdrant stores."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Record:
    """Vector-store row; ``id`` is the stable identity used for upserts."""

    id: int | str | None
    text: str
    embedding: list[float]
    meta: dict[str, Any]


def record_key(record: Record) -> str:
    if record.id is not None:
        if isinstance(record.id, bool) or not isinstance(record.id, (int, str)):
            raise ValueError("record id must be an integer, string, or None")
        return str(record.id)
    canonical = json.dumps(
        {"text": record.text, "meta": record.meta},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_embedding(vector: list[float], dim: int | None = None) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("embedding must be a non-empty list of finite numbers")
    if dim is not None and len(vector) != dim:
        raise ValueError(f"embedding dimension mismatch: expected {dim}, got {len(vector)}")
    normalized = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("embedding values must be finite numbers")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("embedding values must be finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite numbers")
        normalized.append(number)
    return normalized


def validate_record(record: Record, dim: int) -> tuple[str, list[float]]:
    if not isinstance(record.text, str):
        raise TypeError("record text must be a string")
    if not isinstance(record.meta, dict):
        raise TypeError("record metadata must be a dictionary")
    try:
        json.dumps(record.meta, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("record metadata must contain JSON-compatible values") from exc
    return record_key(record), validate_embedding(record.embedding, dim)


def validate_dimension(dim: int) -> None:
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError("embedding dimension must be a positive integer")


def validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


class VectorStore(Protocol):
    def init(self) -> None: ...
    def upsert(self, records: list[Record]) -> int: ...
    def search(self, query_vec: list[float], top_k: int = 5,
               filters: dict[str, Any] | None = None) -> list[Record]: ...


def create_store(kind: str, **kw):
    """Create one of the supported vector stores for use with the shared Retriever."""
    if kind == "pgvector":
        from llm_harness_lite.rag.store_pgvector import PgVectorStore
        return PgVectorStore(kw["dsn"], kw.get("table", "rag_chunks"), kw.get("dim", 768))
    if kind == "qdrant":
        from llm_harness_lite.rag.store_qdrant import QdrantStore
        return QdrantStore(url=kw.get("url", "http://localhost:6333"),
                           collection=kw.get("collection", "rag_chunks"),
                           dim=kw.get("dim", 768),
                           api_key=kw.get("api_key"))
    raise ValueError(f"unknown store kind: {kind}")
