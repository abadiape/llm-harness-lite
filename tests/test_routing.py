import asyncio

import pytest

from llm_harness_lite.llm.router import LLMRouter
from llm_harness_lite.llm.types import ChatResponse, Message, ToolSpec, Usage


class FakeBackend:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls.append((messages, tools, temperature))
        return ChatResponse(Message("assistant", self.model), usage=Usage(model=self.model))


class AsyncBackend(FakeBackend):
    async def achat(self, messages, tools=None, temperature=0.2):
        self.calls.append((messages, tools, temperature))
        return ChatResponse(Message("assistant", self.model), usage=Usage(model=self.model))


def test_router_dispatches_by_explicit_policy_context():
    cheap, capable = FakeBackend("small"), FakeBackend("large")
    router = LLMRouter(
        {"cheap": cheap, "capable": capable},
        lambda request: "capable" if request.tools else "cheap",
    )
    message = [Message("user", "question")]
    tool = [ToolSpec("lookup", "Find data", {"type": "object"})]

    assert router.chat(message).message.content == "small"
    assert router.chat(message, tools=tool).message.content == "large"
    assert router.model_for_request(message, tools=tool) == '["capable","large"]'
    assert len(cheap.calls) == len(capable.calls) == 1


def test_router_supports_native_async_and_sync_backends():
    async_backend = AsyncBackend("async")
    sync_backend = FakeBackend("sync")
    router = LLMRouter(
        {"async": async_backend, "sync": sync_backend},
        lambda request: "async" if request.temperature == 0 else "sync",
    )

    async def call_routes():
        return (
            await router.achat([Message("user", "one")], temperature=0),
            await router.achat([Message("user", "two")], temperature=0.5),
        )

    first, second = asyncio.run(call_routes())
    assert first.message.content == "async"
    assert second.message.content == "sync"


def test_router_rejects_unknown_policy_route():
    router = LLMRouter({"only": FakeBackend("model")}, lambda request: "missing")

    with pytest.raises(ValueError, match="unknown route"):
        router.chat([Message("user", "question")])
