import asyncio
from types import SimpleNamespace

import pytest

from llm_harness_lite.llm.types import Message, ToolSpec
from llm_harness_lite.utils.costs import CostManager

anthropic_backend = pytest.importorskip("llm_harness_lite.llm.anthropic_backend")
AnthropicBackend = anthropic_backend.AnthropicBackend


class FakeMessages:
    async def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Checking."),
                SimpleNamespace(
                    type="tool_use",
                    id="call-1",
                    name="lookup",
                    input={"query": "docs"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=4, output_tokens=3),
        )


class FakeAsyncAnthropic:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.messages = FakeMessages()
        self.closed = False

    async def close(self):
        self.closed = True


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False

    def close(self):
        self.closed = True


def test_anthropic_async_chat_normalizes_tools_and_closes_clients(monkeypatch):
    async def scenario():
        monkeypatch.setattr(anthropic_backend.anthropic, "Anthropic", FakeAnthropic)
        monkeypatch.setattr(
            anthropic_backend.anthropic, "AsyncAnthropic", FakeAsyncAnthropic
        )
        manager = CostManager(overrides={"claude-test": (0, 0)})
        backend = AnthropicBackend(
            "claude-test", cost_manager=manager, api_key="test-key", timeout=3.0
        )
        spec = ToolSpec("lookup", "Search", {"type": "object", "properties": {}})
        result = await backend.achat(
            [Message("system", "Use tools."), Message("user", "find docs")],
            [spec],
        )

        assert result.finish == "tool_calls"
        assert result.message.content == "Checking."
        assert result.message.tool_calls[0].arguments == {"query": "docs"}
        assert backend._async_client.options["timeout"] == 3.0
        await backend.aclose()
        assert backend.client.closed
        assert backend._async_client is None
    asyncio.run(scenario())
