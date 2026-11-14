"""Backend contract. Every new provider implements this Protocol."""
from __future__ import annotations

from typing import Protocol

from llm_harness_lite.llm.types import ChatResponse, Message, ToolSpec


class LLMBackend(Protocol):
    model: str

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse: ...


class AsyncLLMBackend(Protocol):
    model: str

    async def achat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                    temperature: float = 0.2) -> ChatResponse: ...
    async def aclose(self) -> None: ...
