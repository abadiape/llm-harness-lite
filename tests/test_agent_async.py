import asyncio

import pytest

from llm_harness_lite.agent.loop import Agent, AgentStopReason
from llm_harness_lite.llm.types import Message, ToolCallRequest
from llm_harness_lite.tools.registry import ToolRegistry
from tests.agent_fakes import FakeLLM, response


def test_async_tool_runs_through_async_agent_api():
    registry = ToolRegistry()
    calls = []

    async def lookup(query: str) -> str:
        calls.append(query)
        return "async result"

    registry.register(lookup)
    llm = FakeLLM([
        response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "lookup", {"query": "docs"})],
            ),
            finish="tool_calls",
        ),
        response(Message(role="assistant", content="Done.")),
    ])
    result = asyncio.run(Agent(llm, registry).arun("Find the docs."))

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert calls == ["docs"]
    assert llm.calls[1][-1].content == "async result"

def test_async_backend_runs_through_async_agent_api():
    class AsyncFakeLLM:
        model = "async-fake"

        async def achat(self, messages, tools=None, temperature=0.2):
            return response(Message(role="assistant", content="Async answer."))

    result = asyncio.run(Agent(AsyncFakeLLM(), ToolRegistry()).arun("Respond."))

    assert result.answer == "Async answer."
    assert result.usage.model == "async-fake"

def test_run_rejects_nested_event_loop_use():
    agent = Agent(FakeLLM([]), ToolRegistry())

    async def call_run():
        with pytest.raises(RuntimeError, match="await Agent.arun"):
            agent.run("goal")

    asyncio.run(call_run())
