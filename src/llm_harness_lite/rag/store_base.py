"""Shared RAG contract for the supported pgvector and Qdrant stores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Record:
    id: int | str | None
    text: str
    embedding: list[float]
    meta: dict[str, Any]

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
