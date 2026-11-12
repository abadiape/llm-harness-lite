"""Offline contracts for the three add-ons (no servers needed)."""
from llm_harness_lite.rag.store_base import Record, create_store
from llm_harness_lite.rag.store_pgvector import PgVectorStore
from llm_harness_lite.utils.cache import ChatCache, CachedLLM
from llm_harness_lite.llm.types import Message, ChatResponse, Usage
from llm_harness_lite.tools.registry import ToolRegistry
from pgvector import Vector
import tempfile, os

class FakeEmbed:
    model = "fake"
    def embed(self, texts):
        # deterministic toy vectors so cosine is predictable
        return [[float(len(t)), 1.0, 0.0] for t in texts]

class FakeLLM:
    model = "fake"
    def __init__(self): self.calls = 0
    def chat(self, messages, tools=None, temperature=0.2):
        self.calls += 1
        return ChatResponse(Message("assistant", "hello"), "stop", Usage(5, 3, 0.001, "fake", 0.0))

def test_cache_exact_hit():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.db")
        base, cache = FakeLLM(), ChatCache(p)
        llm = CachedLLM(base, cache)
        msgs = [Message("user", "hi")]
        llm.chat(msgs)
        llm.chat(msgs)  # second = exact hit
        assert base.calls == 1 and cache.hits_exact == 1

def test_cache_semantic_hit():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.db")
        base, cache = FakeLLM(), ChatCache(p)
        llm = CachedLLM(base, cache, embedder=FakeEmbed(), semantic_threshold=0.99)
        llm.chat([Message("user", "hello world")])
        # different wording, similar length -> toy embedder gives high cosine
        llm.chat([Message("user", "hello world!")])
        assert cache.hits_sem >= 1

def test_qdrant_interface_shape():
    # import-only check so CI without qdrant still validates the protocol
    import ast
    src = open("src/harness/rag/store_qdrant.py").read()
    tree = ast.parse(src)
    methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"init", "upsert", "search"} <= methods

def test_create_store_unknown():
    try:
        create_store("nope")
        assert False
    except ValueError:
        assert True

def test_pgvector_binds_vector_parameters(monkeypatch):
    class FakeConnection:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params):
            self.calls.append(params)
            return self

        def fetchall(self):
            return []

    connection = FakeConnection()
    monkeypatch.setattr("psycopg.connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr("pgvector.psycopg.register_vector", lambda conn: None)

    store = PgVectorStore("unused")
    store.upsert([Record(id=None, text="chunk", embedding=[0.1, 0.2, 0.3], meta={})])
    store.search([0.1, 0.2, 0.3])

    assert isinstance(connection.calls[0][-1], Vector)
    assert isinstance(connection.calls[1][-2], Vector)
