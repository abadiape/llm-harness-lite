import asyncio
from types import SimpleNamespace

import pytest

openai_compat = pytest.importorskip("llm_harness_lite.llm.openai_compat")
OpenAICompatBackend = openai_compat.OpenAICompatBackend
from llm_harness_lite.llm.types import Message, ToolSpec
from llm_harness_lite.utils.costs import CostManager


class FakeCompletions:
    def __init__(self, response=None, started=None):
        self.response = response
        self.started = started
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        if self.started:
            self.started.set()
            await asyncio.Event().wait()
        return self.response


class FakeAsyncOpenAI:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False
        self.chat = SimpleNamespace(
            completions=FakeCompletions(
                SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="lookup",
                                    arguments='{"query":"docs"}',
                                ),
                            )],
                        ),
                        finish_reason="tool_calls",
                    )],
                    usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
                )
            )
        )

    async def close(self):
        self.closed = True


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False

    def close(self):
        self.closed = True


def test_openai_async_chat_normalizes_tools_and_closes_clients(monkeypatch):
    async def scenario():
        monkeypatch.setattr(openai_compat, "OpenAI", FakeOpenAI)
        monkeypatch.setattr(openai_compat, "AsyncOpenAI", FakeAsyncOpenAI)
        manager = CostManager(overrides={"local-model": (0, 0)})
        backend = OpenAICompatBackend(
            "local-model", cost_manager=manager, timeout=2.5
        )
        spec = ToolSpec("lookup", "Search", {"type": "object", "properties": {}})
        result = await backend.achat([Message("user", "find docs")], [spec])
        assert result.finish == "tool_calls"
        assert result.message.tool_calls[0].arguments == {"query": "docs"}
        assert result.usage.input_tokens == 5
        assert backend._async_client.options["timeout"] == 2.5
        await backend.aclose()
        assert backend._async_client is None
        assert backend.client.closed
    asyncio.run(scenario())

def test_openai_async_chat_propagates_cancellation(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        monkeypatch.setattr(openai_compat, "OpenAI", FakeOpenAI)

        class PendingAsyncOpenAI(FakeAsyncOpenAI):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.chat.completions = FakeCompletions(started=started)

        monkeypatch.setattr(openai_compat, "AsyncOpenAI", PendingAsyncOpenAI)
        manager = CostManager(overrides={"local-model": (0, 0)})
        backend = OpenAICompatBackend("local-model", cost_manager=manager)
        task = asyncio.create_task(backend.achat([Message("user", "wait")]))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await backend.aclose()
    asyncio.run(scenario())
