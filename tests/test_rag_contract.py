import json
import sys
import types

import pytest

from llm_harness_lite.rag.chunker import chunk_text
from llm_harness_lite.rag.retriever import Retriever
from llm_harness_lite.rag.store_base import Record
from llm_harness_lite.rag.store_pgvector import PgVectorStore
from llm_harness_lite.rag.store_qdrant import QdrantStore


class FakeEmbedder:
    model = "offline"
    dim = 2

    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class FakePgConnection:
    def __init__(self):
        self.records = {}
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, params=()):
        if query.startswith("INSERT"):
            key, text, meta, vector = params
            self.records[key] = (text, json.loads(meta), vector.to_list())
        elif query.startswith("SELECT"):
            self.rows = [
                (key, text, meta)
                for key, (text, meta, _) in list(self.records.items())[:params[-1]]
            ]
        return self

    def fetchall(self):
        return self.rows


class FakePoint:
    def __init__(self, id, vector, payload):
        self.id, self.vector, self.payload = id, vector, payload


class FakeQdrantClient:
    def __init__(self, **kwargs):
        self.points = {}

    def collection_exists(self, collection):
        return collection in getattr(self, "collections", set())

    def create_collection(self, collection, vectors_config):
        self.collections = {collection}

    def upsert(self, collection_name, points):
        self.points.update({point.id: point for point in points})

    def query_points(self, collection_name, query, limit, query_filter):
        return types.SimpleNamespace(points=list(self.points.values())[:limit])


def install_fake_pgvector(monkeypatch, connection):
    class Vector(list):
        def to_list(self):
            return list(self)

    psycopg = types.ModuleType("psycopg")
    psycopg.connect = lambda *args, **kwargs: connection
    pgvector = types.ModuleType("pgvector")
    pgvector.__path__ = []
    pgvector.Vector = Vector
    pgvector_psycopg = types.ModuleType("pgvector.psycopg")
    pgvector_psycopg.register_vector = lambda conn: None
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "pgvector", pgvector)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", pgvector_psycopg)


def install_fake_qdrant(monkeypatch):
    client_module = types.ModuleType("qdrant_client")
    client_module.QdrantClient = FakeQdrantClient
    models = types.ModuleType("qdrant_client.models")
    models.Distance = types.SimpleNamespace(COSINE="cosine")
    models.VectorParams = lambda **kwargs: kwargs
    models.PointStruct = FakePoint
    models.Filter = lambda **kwargs: kwargs
    models.FieldCondition = lambda **kwargs: kwargs
    models.MatchValue = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "qdrant_client", client_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)


def exercise_retriever(store, count):
    store.init()
    retriever = Retriever(FakeEmbedder(), store, chunk_size=4, chunk_overlap=1)
    assert retriever.ingest("alpha beta", source="fixture") == count
    assert retriever.ingest("alpha beta", source="fixture") == count
    assert len(retriever.search("query", top_k=2)) == 2
    direct_record = Record(id=None, text="direct", embedding=[6.0, 1.0], meta={})
    assert store.upsert([direct_record]) == 1
    assert store.upsert([direct_record]) == 1


def test_stores_share_idempotent_retriever_contract(monkeypatch):
    pg_connection = FakePgConnection()
    install_fake_pgvector(monkeypatch, pg_connection)
    exercise_retriever(PgVectorStore("unused", dim=2), 4)
    assert len(pg_connection.records) == 5
    install_fake_qdrant(monkeypatch)
    qdrant = QdrantStore(dim=2)
    exercise_retriever(qdrant, 4)
    assert len(qdrant._client.points) == 5


def test_retriever_rejects_bad_vectors_and_limits():
    class BadEmbedder(FakeEmbedder):
        def embed(self, texts):
            return [[float("inf"), 1.0] for _ in texts]

    class MissingEmbedder(FakeEmbedder):
        def embed(self, texts):
            return []

    retriever = Retriever(BadEmbedder(), QdrantStore(dim=2))
    with pytest.raises(ValueError, match="finite"):
        retriever.search("query")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("query", top_k=0)
    with pytest.raises(ValueError, match="0 vectors"):
        Retriever(MissingEmbedder(), QdrantStore(dim=2)).search("query")
    with pytest.raises(ValueError, match="batch"):
        retriever.ingest("query", batch=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", size=4, overlap=4)
