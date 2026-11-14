import asyncio

import pytest

from llm_harness_lite.agent.loop import Agent, AgentStopReason
from llm_harness_lite.llm.types import (
    ChatResponse,
    Message,
    ToolCallRequest,
    Usage,
)
from llm_harness_lite.tools.registry import ToolExecutionError, ToolRegistry


class FakeLLM:
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def _response(message, finish="stop", cost=0.001):
    return ChatResponse(
        message=message,
        finish=finish,
        usage=Usage(input_tokens=2, output_tokens=1, cost_usd=cost, model="fake"),
    )


def test_agent_runs_tool_then_returns_final_answer():
    registry = ToolRegistry()

    def lookup(query: str) -> str:
        return f"found: {query}"

    registry.register(lookup)
    llm = FakeLLM([
        _response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "lookup", {"query": "docs"})],
            ),
            finish="tool_calls",
        ),
        _response(Message(role="assistant", content="Here is the answer.")),
    ])

    result = Agent(llm, registry).run("Find the docs.")

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.answer == "Here is the answer."
    assert result.steps == 2
    assert result.usage.input_tokens == 4
    assert llm.calls[1][-1].role == "tool"
    assert llm.calls[1][-1].content == "found: docs"
    assert llm.calls[1][-1].tool_call_id == "call-1"


def test_budget_overrun_skips_requested_tools():
    registry = ToolRegistry()
    side_effects = []
    registry.register(lambda: side_effects.append("ran"))
    llm = FakeLLM([
        _response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "<lambda>", {})],
            ),
            finish="tool_calls",
            cost=0.02,
        ),
    ])

    result = Agent(llm, registry, budget_usd=0.01).run("Do the work.")

    assert result.stop_reason is AgentStopReason.BUDGET_EXCEEDED
    assert result.steps == 1
    assert result.usage.cost_usd == pytest.approx(0.02)
    assert side_effects == []
    assert result.history[-1].role == "tool"
    assert "skipped" in result.history[-1].content


def test_max_steps_stops_without_an_extra_model_call():
    registry = ToolRegistry()
    registry.register(lambda: "done", name="work")
    llm = FakeLLM([
        _response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "work", {})],
            ),
            finish="tool_calls",
        ),
    ])

    result = Agent(llm, registry, max_steps=1).run("Do the work.")

    assert result.stop_reason is AgentStopReason.MAX_STEPS
    assert result.steps == 1
    assert len(llm.calls) == 1


def test_tool_calls_finish_without_calls_is_invalid():
    llm = FakeLLM([_response(Message(role="assistant"), finish="tool_calls")])

    result = Agent(llm, ToolRegistry()).run("Do the work.")

    assert result.stop_reason is AgentStopReason.INVALID_RESPONSE
    assert result.steps == 1


@pytest.mark.parametrize(
    ("finish", "stop_reason"),
    [
        ("length", AgentStopReason.INCOMPLETE),
        ("error", AgentStopReason.MODEL_ERROR),
    ],
)
def test_provider_termination_is_reported(finish, stop_reason):
    llm = FakeLLM([_response(Message(role="assistant", content="partial"), finish)])

    result = Agent(llm, ToolRegistry()).run("Respond.")

    assert result.stop_reason is stop_reason
    assert result.answer == "partial"


def test_async_tool_runs_through_async_agent_api():
    registry = ToolRegistry()
    calls = []

    async def lookup(query: str) -> str:
        calls.append(query)
        return "async result"

    registry.register(lookup)
    llm = FakeLLM([
        _response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "lookup", {"query": "docs"})],
            ),
            finish="tool_calls",
        ),
        _response(Message(role="assistant", content="Done.")),
    ])

    result = asyncio.run(Agent(llm, registry).arun("Find the docs."))

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert calls == ["docs"]
    assert llm.calls[1][-1].content == "async result"


def test_async_backend_runs_through_async_agent_api():
    class AsyncFakeLLM:
        model = "async-fake"

        async def achat(self, messages, tools=None, temperature=0.2):
            return _response(Message(role="assistant", content="Async answer."))

    result = asyncio.run(Agent(AsyncFakeLLM(), ToolRegistry()).arun("Respond."))

    assert result.answer == "Async answer."
    assert result.usage.model == "async-fake"


def test_recoverable_tool_error_is_returned_to_model():
    registry = ToolRegistry()

    def lookup(query: str) -> str:
        raise ToolExecutionError("source unavailable")

    registry.register(lookup)
    llm = FakeLLM([
        _response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "lookup", {"query": "docs"})],
            ),
            finish="tool_calls",
        ),
        _response(Message(role="assistant", content="I could not retrieve that.")),
    ])

    result = Agent(llm, registry).run("Find the docs.")

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert "source unavailable" in llm.calls[1][-1].content


def test_unexpected_tool_error_is_raised():
    registry = ToolRegistry()

    def broken() -> str:
        raise ValueError("bug in tool")

    registry.register(broken)
    llm = FakeLLM([
        _response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "broken", {})],
            ),
            finish="tool_calls",
        ),
    ])

    with pytest.raises(ValueError, match="bug in tool"):
        Agent(llm, registry).run("Run the tool.")


def test_run_rejects_nested_event_loop_use():
    agent = Agent(FakeLLM([]), ToolRegistry())

    async def call_run():
        with pytest.raises(RuntimeError, match="await Agent.arun"):
            agent.run("goal")

    asyncio.run(call_run())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"budget_usd": -1}, "budget_usd"),
        ({"budget_usd": float("inf")}, "budget_usd"),
    ],
)
def test_agent_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Agent(FakeLLM([]), ToolRegistry(), **kwargs)
