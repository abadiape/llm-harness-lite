"""pgvector store. Implements VectorStore from store_base."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from llm_harness_lite.rag.store_base import (
    Record,
    VectorStore,  # noqa: F401
    validate_dimension,
    validate_embedding,
    validate_record,
    validate_top_k,
)


class PgVectorStore:
    """Requires the PostgreSQL vector extension; init creates its table and indexes."""

    def __init__(self, dsn: str, table: str = "rag_chunks", dim: int = 768):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("table must be a simple SQL identifier")
        validate_dimension(dim)
        self.dsn, self.table, self.dim = dsn, table.lower(), dim

    def _index_name(self, suffix: str) -> str:
        digest = hashlib.sha256(self.table.encode("utf-8")).hexdigest()[:8]
        return f"{self.table[:40]}_{digest}_{suffix}"

    def init(self) -> None:
        import psycopg

        table = f'"{self.table}"'
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"id SERIAL PRIMARY KEY, record_key TEXT, text TEXT, meta JSONB, "
                f"embedding vector({self.dim}))"
            )
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS record_key TEXT")
            conn.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS "{self._index_name("record_key")}" '
                f"ON {table} (record_key)"
            )
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{self._index_name("hnsw")}" '
                f"ON {table} USING hnsw (embedding vector_cosine_ops)"
            )

    def upsert(self, records: list[Record]) -> int:
        prepared = [(record, *validate_record(record, self.dim)) for record in records]
        if not prepared:
            return 0
        import psycopg
        from pgvector import Vector
        from pgvector.psycopg import register_vector

        query = (
            f'INSERT INTO "{self.table}" (record_key, text, meta, embedding) '
            "VALUES (%s, %s, %s, %s) ON CONFLICT (record_key) DO UPDATE SET "
            "text = EXCLUDED.text, meta = EXCLUDED.meta, embedding = EXCLUDED.embedding"
        )
        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            for record, key, vector in prepared:
                conn.execute(
                    query,
                    (key, record.text, json.dumps(record.meta, allow_nan=False), Vector(vector)),
                )
        return len(prepared)

    def search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Record]:
        validate_top_k(top_k)
        vector = validate_embedding(query_vec, self.dim)
        import psycopg
        from pgvector import Vector
        from pgvector.psycopg import register_vector

        clauses, params = [], []
        for key, value in (filters or {}).items():
            clauses.append("meta->>%s = %s")
            params.extend((key, str(value)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f'SELECT record_key, text, meta FROM "{self.table}" {where} '
            f"ORDER BY embedding <=> %s LIMIT %s"
        )
        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            rows = conn.execute(query, (*params, Vector(vector), top_k)).fetchall()
        return [
            Record(id=row[0], text=row[1], meta=row[2] or {}, embedding=[])
            for row in rows
        ]
