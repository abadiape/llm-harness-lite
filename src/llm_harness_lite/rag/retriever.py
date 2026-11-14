"""High-level RAG API: ingest + search. Works with any VectorStore backend."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_harness_lite.rag.chunker import chunk_text
from llm_harness_lite.rag.store_base import (
    Record,
    validate_dimension,
    validate_embedding,
    validate_top_k,
)


class Retriever:
    def __init__(self, embedder, store, chunk_size: int = 600, chunk_overlap: int = 80):
        self.embedder, self.store = embedder, store
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        store_dim = getattr(store, "dim", None)
        embedder_dim = getattr(embedder, "dim", None)
        if store_dim is not None:
            validate_dimension(store_dim)
        if embedder_dim is not None:
            validate_dimension(embedder_dim)
        if store_dim is not None and embedder_dim is not None and store_dim != embedder_dim:
            raise ValueError("embedder and vector store dimensions do not match")
        self.dim = store_dim if store_dim is not None else embedder_dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
        return [validate_embedding(vector, self.dim) for vector in vectors]

    def ingest(self, text: str, source: str = "doc", batch: int = 64) -> int:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        chunks = chunk_text(text, source, self.chunk_size, self.chunk_overlap)
        vectors = []
        for start in range(0, len(chunks), batch):
            vectors.extend(self._embed([chunk.text for chunk in chunks[start:start + batch]]))
        model = getattr(self.embedder, "model", "?")
        records = []
        for chunk, vector in zip(chunks, vectors):
            identity = json.dumps([source, chunk.meta["n"]], separators=(",", ":"))
            key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            records.append(
                Record(
                    id=key,
                    text=chunk.text,
                    embedding=vector,
                    meta={**chunk.meta, "embed_model": model},
                )
            )
        return self.store.upsert(records)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Record]:
        validate_top_k(top_k)
        query_vectors = self._embed([query])
        return self.store.search(query_vectors[0], top_k=top_k, filters=filters)

    def context_block(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> str:
        hits = self.search(query, top_k, filters)
        if not hits:
            return "[no hits]"
        return "\n\n".join(
            f"[{index + 1}] {hit.text}\nSource: {hit.meta}"
            for index, hit in enumerate(hits)
        )
