"""pgvector store. Implements VectorStore from store_base."""
from __future__ import annotations
from typing import Any
from llm_harness_lite.rag.store_base import Record, VectorStore  # noqa: F401  (re-export)

class PgVectorStore:
    """Requires: CREATE EXTENSION vector; table with embedding vector(dim)."""
    def __init__(self, dsn: str, table: str = "rag_chunks", dim: int = 768):
        self.dsn, self.table, self.dim = dsn, table, dim

    def init(self) -> None:
        import psycopg
        with psycopg.connect(self.dsn, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"""CREATE TABLE IF NOT EXISTS {self.table}(
              id SERIAL PRIMARY KEY, text TEXT, meta JSONB,
              embedding vector({self.dim}))""")
            c.execute(f"""CREATE INDEX IF NOT EXISTS {self.table}_h
              ON {self.table} USING hnsw (embedding vector_cosine_ops)""")

    def upsert(self, records: list[Record]) -> int:
        import psycopg
        from pgvector import Vector
        from pgvector.psycopg import register_vector
        with psycopg.connect(self.dsn, autocommit=True) as c:
            register_vector(c)
            import json
            for r in records:
                c.execute(f"INSERT INTO {self.table}(text, meta, embedding) VALUES(%s,%s,%s)",
                          (r.text, json.dumps(r.meta), Vector(r.embedding)))
        return len(records)

    def search(self, query_vec: list[float], top_k: int = 5,
               filters: dict[str, Any] | None = None) -> list[Record]:
        import psycopg, json
        from pgvector import Vector
        from pgvector.psycopg import register_vector
        where, params = "", []
        if filters:  # simple equality on meta->>key, e.g. {"source": "a.md"}
            clauses = []
            for k, v in filters.items():
                clauses.append(f"meta->>%s = %s")
                params += [k, str(v)]
            where = "WHERE " + " AND ".join(clauses)
        with psycopg.connect(self.dsn) as c:
            register_vector(c)
            rows = c.execute(
                f"SELECT id, text, meta FROM {self.table} {where}"
                f" ORDER BY embedding <=> %s LIMIT %s",
                (*params, Vector(query_vec), top_k)).fetchall()
            return [Record(id=r[0], text=r[1], meta=r[2] or {}, embedding=[]) for r in rows]
