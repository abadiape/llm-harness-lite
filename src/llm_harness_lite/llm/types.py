"""Normalized types. OpenAI `tools` format is the internal lingua-franca."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str | None = None   # for role=="tool"
    name: str | None = None           # tool name for role=="tool"

@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolSpec:
    """JSON-Schema style tool definition (OpenAI-compatible)."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema object

    def to_openai(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    latency_s: float = 0.0

@dataclass
class ChatResponse:
    message: Message
    finish: Literal["stop", "tool_calls", "length", "error"] = "stop"
    usage: Usage = field(default_factory=Usage)
