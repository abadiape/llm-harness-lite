import pytest

from llm_harness_lite.agent.loop import Agent
from llm_harness_lite.llm.types import Message, ToolCallRequest, Usage
from llm_harness_lite.tools.registry import ToolRegistry
from tests.agent_fakes import FakeLLM, response


def test_agent_emits_correlated_metadata_without_prompt_or_tool_content():
    events = []
    registry = ToolRegistry()

    def lookup(secret: str) -> str:
        return f"found {secret}"

    registry.register(lookup)
    llm = FakeLLM([
        response(
            Message(
                role="assistant",
                tool_calls=[ToolCallRequest("call-1", "lookup", {"secret": "private-tool-arg"})],
            ),
            finish="tool_calls",
        ),
        response(Message(role="assistant", content="Done.")),
    ])

    result = Agent(llm, registry, observer=events.append).run("private goal")

    assert result.answer == "Done."
    assert [event.name for event in events] == [
        "agent.started", "llm.completed", "tool.completed", "llm.completed", "agent.completed",
    ]
    assert len({event.run_id for event in events}) == 1
    attributes = [repr(dict(event.attributes)) for event in events]
    assert all("private goal" not in item and "private-tool-arg" not in item for item in attributes)
    assert dict(events[-1].attributes)["stop_reason"] == "completed"


def test_invalid_usage_is_reported_and_cannot_bypass_budget():
    events = []
    invalid = response(Message(role="assistant", content="bad"))
    invalid.usage = Usage(input_tokens=1, output_tokens=1, cost_usd=float("nan"))

    with pytest.raises(ValueError, match="cost_usd"):
        Agent(FakeLLM([invalid]), ToolRegistry(), observer=events.append).run("goal")

    assert events[-1].name == "llm.invalid_response"


def test_agent_rejects_non_callable_observer():
    with pytest.raises(TypeError, match="observer"):
        Agent(FakeLLM([]), ToolRegistry(), observer=object())
