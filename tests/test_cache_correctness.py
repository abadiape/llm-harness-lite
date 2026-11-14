import json
import sqlite3

from llm_harness_lite.llm.router import LLMRouter
from llm_harness_lite.llm.types import ChatResponse, Message, ToolSpec, Usage
from llm_harness_lite.utils.cache import CachedLLM, ChatCache


class FakeEmbedder:
    model = "embed-model"

    def __init__(self):
        self.prompts = []

    def embed(self, prompts):
        self.prompts.extend(prompts)
        return [[float(len(prompt)), 1.0] for prompt in prompts]


class FakeBackend:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls += 1
        return ChatResponse(
            Message("assistant", f"{self.model}:{self.calls}"),
            usage=Usage(input_tokens=4, output_tokens=2, model=self.model),
        )


def test_exact_cache_key_includes_temperature_and_full_tool_schema(tmp_path):
    backend = FakeBackend("model")
    cache = CachedLLM(backend, ChatCache(tmp_path / "cache.db"), cache_tools=True)
    messages = [Message("user", "question")]
    tool_a = [ToolSpec("lookup", "lookup", {"type": "object", "properties": {"a": {}}})]
    tool_b = [ToolSpec("lookup", "lookup", {"type": "object", "properties": {"b": {}}})]

    cache.chat(messages, tools=tool_a, temperature=0.1)
    cache.chat(messages, tools=tool_b, temperature=0.1)
    cache.chat(messages, tools=tool_b, temperature=0.4)
    cache.chat(messages, tools=tool_b, temperature=0.4)

    assert backend.calls == 3
    assert cache.cache.hits_exact == 1


def test_semantic_cache_is_scoped_by_model_temperature_and_full_context(tmp_path):
    cache = ChatCache(tmp_path / "cache.db")
    embedder = FakeEmbedder()
    messages = [Message("system", "be concise"), Message("user", "hello world")]
    first = FakeBackend("model-a")
    CachedLLM(first, cache, embedder=embedder).chat(messages, temperature=0.1)

    other_model = FakeBackend("model-b")
    CachedLLM(other_model, cache, embedder=embedder).chat(
        [Message("system", "be concise"), Message("user", "hello world!")],
        temperature=0.1,
    )
    other_temperature = FakeBackend("model-a")
    CachedLLM(other_temperature, cache, embedder=embedder).chat(
        [Message("system", "be concise"), Message("user", "hello world!")],
        temperature=0.4,
    )

    assert first.calls == other_model.calls == other_temperature.calls == 1
    assert cache.hits_sem == 0
    assert json.loads(embedder.prompts[0])[0]["role"] == "system"


def test_router_cache_uses_the_selected_backend_identity(tmp_path):
    local, remote = FakeBackend("small"), FakeBackend("large")
    router = LLMRouter(
        {"local": local, "remote": remote},
        lambda request: "local" if "local" in request.messages[-1].content else "remote",
    )
    cache = ChatCache(tmp_path / "cache.db")
    wrapped = CachedLLM(router, cache, embedder=FakeEmbedder())

    wrapped.chat([Message("user", "similar local prompt")])
    wrapped.chat([Message("user", "similar remote prompt")])

    assert local.calls == remote.calls == 1
    assert cache.hits_sem == 0


def test_existing_cache_schema_adds_scope_without_dropping_rows(tmp_path):
    path = tmp_path / "cache.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE llm_cache(
          key TEXT PRIMARY KEY, model TEXT, prompt TEXT, embedding TEXT,
          content TEXT, toolcalls TEXT, tokens_in INT, tokens_out INT, ts REAL)""")
        connection.execute(
            "INSERT INTO llm_cache VALUES(?,?,?,?,?,?,?,?,?)",
            ("old", "model", "prompt", None, "cached", "[]", 1, 2, 1.0),
        )

    ChatCache(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(llm_cache)")}
        row = connection.execute(
            "SELECT content, scope, finish FROM llm_cache WHERE key='old'"
        ).fetchone()

    assert "scope" in columns
    assert "finish" in columns
    assert row == ("cached", "", "stop")


def test_cache_preserves_response_finish_reason(tmp_path):
    cache = ChatCache(tmp_path / "cache.db")
    response = ChatResponse(
        Message("assistant", "partial"),
        finish="length",
        usage=Usage(input_tokens=2, output_tokens=3),
    )
    cache.put("key", "model", "prompt", response)

    assert cache.get_exact("key").finish == "length"
