"""Low-overhead structured events for agent runs."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from llm_harness_lite.agent.types import AgentResult, AgentStopReason
from llm_harness_lite.llm.types import Message, Usage


@dataclass(frozen=True)
class AgentEvent:
    name: str
    run_id: str
    timestamp: float
    attributes: Mapping[str, Any]


AgentObserver = Callable[[AgentEvent], None]


def emit_event(
    observer: AgentObserver | None,
    name: str,
    run_id: str,
    **attributes: Any,
) -> None:
    if observer is not None:
        observer(AgentEvent(name, run_id, time.time(), attributes))


def completed_result(
    observer: AgentObserver | None,
    run_id: str,
    answer: str,
    steps: int,
    usage: Usage,
    history: list[Message],
    stop_reason: AgentStopReason,
) -> AgentResult:
    result = AgentResult(answer, steps, usage, history, stop_reason)
    emit_event(
        observer,
        "agent.completed",
        run_id,
        steps=steps,
        stop_reason=stop_reason.value,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
    )
    return result
