"""Qdrant store. Same VectorStore protocol as pgvector."""
from __future__ import annotations

import uuid
from typing import Any

from llm_harness_lite.rag.store_base import (
    Record,
    validate_dimension,
    validate_embedding,
    validate_record,
    validate_top_k,
)


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "rag_chunks",
        dim: int = 768,
        api_key: str | None = None,
    ):
        validate_dimension(dim)
        self.url, self.collection, self.dim = url, collection, dim
        self.api_key = api_key
        self._client = None

    def _get(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def init(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get()
        if not client.collection_exists(self.collection):
            client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, records: list[Record]) -> int:
        prepared = [(record, *validate_record(record, self.dim)) for record in records]
        if not prepared:
            return 0
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"llm-harness-lite:{key}")),
                vector=vector,
                payload={"record_key": key, "text": record.text, "meta": record.meta},
            )
            for record, key, vector in prepared
        ]
        self._get().upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Record]:
        validate_top_k(top_k)
        vector = validate_embedding(query_vec, self.dim)
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qfilter = None
        if filters:
            qfilter = Filter(
                must=[
                    FieldCondition(key=f"meta.{key}", match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        client = self._get()
        if callable(getattr(client, "query_points", None)):
            points = client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
                query_filter=qfilter,
            ).points
        else:
            points = client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                query_filter=qfilter,
            )
        records = []
        for point in points:
            payload = point.payload or {}
            records.append(
                Record(
                    id=payload.get("record_key", str(point.id)),
                    text=payload.get("text", ""),
                    meta=payload.get("meta", {}),
                    embedding=[],
                )
            )
        return records
