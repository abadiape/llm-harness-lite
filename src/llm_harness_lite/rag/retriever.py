"""High-level RAG API: ingest + search. Works with any VectorStore backend."""
from __future__ import annotations
from typing import Any
from llm_harness_lite.rag.chunker import chunk_text
from llm_harness_lite.rag.store_base import Record

class Retriever:
    def __init__(self, embedder, store, chunk_size=600, chunk_overlap=80):
        self.embedder, self.store = embedder, store
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def ingest(self, text: str, source: str = "doc", batch: int = 64) -> int:
        chunks = chunk_text(text, source, self.chunk_size, self.chunk_overlap)
        vecs: list[list[float]] = []
        for i in range(0, len(chunks), batch):
            vecs += self.embedder.embed([c.text for c in chunks[i:i + batch]])
        assert len(vecs) == len(chunks), "embedder batch mismatch"
        return self.store.upsert([Record(id=None, text=c.text, embedding=v,
            meta={**c.meta, "embed_model": getattr(self.embedder, "model", "?")})
            for c, v in zip(chunks, vecs)])

    def search(self, query: str, top_k: int = 5,
               filters: dict[str, Any] | None = None) -> list[Record]:
        qv = self.embedder.embed([query])[0]
        try:
            return self.store.search(qv, top_k=top_k, filters=filters)
        except TypeError:  # old stores without filters param
            return self.store.search(qv, top_k=top_k)

    def context_block(self, query: str, top_k: int = 5,
                      filters: dict[str, Any] | None = None) -> str:
        hits = self.search(query, top_k, filters)
        if not hits:
            return "[no hits]"
        return "\n\n".join(f"[{i+1}] {h.text}\nSource: {h.meta}" for i, h in enumerate(hits))
