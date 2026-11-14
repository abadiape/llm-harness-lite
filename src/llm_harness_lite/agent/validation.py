"""Runtime validation for provider responses before budget accounting."""
from __future__ import annotations

import math

from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, Usage


def validate_agent_config(max_steps: int, budget_usd: float, observer) -> float:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(budget_usd, bool) or not isinstance(budget_usd, (int, float)):
        raise TypeError("budget_usd must be a number")
    try:
        budget = float(budget_usd)
    except OverflowError as exc:
        raise ValueError("budget_usd must be finite and non-negative") from exc
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget_usd must be finite and non-negative")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable")
    return budget


def validate_response(response: ChatResponse) -> None:
    if not isinstance(response, ChatResponse):
        raise TypeError("LLM backend must return a ChatResponse")
    if not isinstance(response.message, Message):
        raise TypeError("LLM response message must be a Message")
    if not isinstance(response.message.content, str):
        raise TypeError("LLM response content must be a string")
    if not isinstance(response.finish, str) or response.finish not in {
        "stop", "tool_calls", "length", "error"
    }:
        raise ValueError(f"unsupported LLM finish reason: {response.finish!r}")
    if not isinstance(response.message.tool_calls, list):
        raise TypeError("LLM response tool_calls must be a list")
    for call in response.message.tool_calls:
        if (not isinstance(call, ToolCallRequest) or not isinstance(call.id, str)
                or not isinstance(call.name, str) or not isinstance(call.arguments, dict)):
            raise TypeError("LLM response contains an invalid tool call")
    if not isinstance(response.usage, Usage):
        raise TypeError("LLM response usage must be a Usage")

    for field in ("input_tokens", "output_tokens"):
        value = getattr(response.usage, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"LLM usage {field} must be a non-negative integer")
    for field in ("cost_usd", "latency_s"):
        value = getattr(response.usage, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"LLM usage {field} must be a number")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"LLM usage {field} must be finite and non-negative") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"LLM usage {field} must be finite and non-negative")
    if not isinstance(response.usage.model, str):
        raise TypeError("LLM usage model must be a string")
