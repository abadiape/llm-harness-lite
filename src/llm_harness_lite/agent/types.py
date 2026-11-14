"""Public result and stop-reason types for agent runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from llm_harness_lite.llm.types import Message, Usage


class AgentStopReason(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    INCOMPLETE = "incomplete"
    MODEL_ERROR = "model_error"
    INVALID_RESPONSE = "invalid_response"


@dataclass
class AgentResult:
    answer: str
    steps: int
    usage: Usage
    history: list[Message] = field(default_factory=list)
    stop_reason: AgentStopReason = AgentStopReason.COMPLETED
