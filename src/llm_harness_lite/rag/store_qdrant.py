"""Qdrant store. Same VectorStore protocol as pgvector — swap via create_store('qdrant')."""
from __future__ import annotations
import uuid
from typing import Any
from llm_harness_lite.rag.store_base import Record

class QdrantStore:
    def __init__(self, url: str = "http://localhost:6333",
                 collection: str = "rag_chunks", dim: int = 768,
                 api_key: str | None = None):
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
        c = self._get()
        if not c.collection_exists(self.collection):
            c.create_collection(self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))

    def upsert(self, records: list[Record]) -> int:
        from qdrant_client.models import PointStruct
        c = self._get()
        points = [PointStruct(id=str(uuid.uuid4()),
                    vector=r.embedding,
                    payload={"text": r.text, "meta": r.meta}) for r in records]
        c.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_vec: list[float], top_k: int = 5,
               filters: dict[str, Any] | None = None) -> list[Record]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        c = self._get()
        qfilter = None
        if filters:
            qfilter = Filter(must=[FieldCondition(key=f"meta.{k}",
                        match=MatchValue(value=v)) for k, v in filters.items()])
        # qdrant-client >=1.9 uses query_points; older uses search
        try:
            res = c.query_points(collection_name=self.collection, query=query_vec,
                                 limit=top_k, query_filter=qfilter)
            pts = res.points
        except (AttributeError, TypeError):
            pts = c.search(collection_name=self.collection, query_vector=query_vec,
                           limit=top_k, query_filter=qfilter)
        out = []
        for p in pts:
            payload = p.payload or {}
            out.append(Record(id=str(p.id), text=payload.get("text", ""),
                              meta=payload.get("meta", {}), embedding=[]))
        return out
