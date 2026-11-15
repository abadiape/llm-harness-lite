"""Async adapters for synchronous and asynchronous backend calls."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Protocol

from llm_harness_lite.llm.types import ChatResponse, Message, ToolSpec
from llm_harness_lite.tools.registry import ToolExecutionError

from .telemetry import AgentObserver, emit_event


class SyncChatBackend(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
    ) -> ChatResponse: ...


class AsyncChatBackend(Protocol):
    def achat(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float,
    ) -> Any: ...


class ToolCatalog(Protocol):
    def specs(self) -> list[ToolSpec]: ...


class SyncToolExecutor(ToolCatalog, Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> Any: ...


class AsyncToolExecutor(ToolCatalog, Protocol):
    async def acall(self, name: str, arguments: dict[str, Any]) -> Any: ...


AgentBackend = SyncChatBackend | AsyncChatBackend
AgentToolExecutor = SyncToolExecutor | AsyncToolExecutor


async def chat_backend(
    llm: AgentBackend,
    messages: list[Message],
    tools: list[ToolSpec],
    temperature: float,
) -> ChatResponse:
    async_chat = getattr(llm, "achat", None)
    if async_chat is None:
        sync_chat = getattr(llm, "chat", None)
        if sync_chat is None:
            raise TypeError("LLM backend must provide chat() or achat()")
        return await asyncio.to_thread(sync_chat, messages, tools=tools, temperature=temperature)
    if inspect.iscoroutinefunction(async_chat):
        return await async_chat(messages, tools=tools, temperature=temperature)
    result = await asyncio.to_thread(async_chat, messages, tools=tools, temperature=temperature)
    return await result if inspect.isawaitable(result) else result


async def call_tool(
    registry: AgentToolExecutor,
    name: str,
    arguments: dict[str, Any],
    observer: AgentObserver | None,
    run_id: str,
) -> str:
    try:
        async_call = getattr(registry, "acall", None)
        if async_call is None:
            sync_call = getattr(registry, "call", None)
            if sync_call is None:
                raise TypeError("tool executor must provide call() or acall()")
            result = await asyncio.to_thread(sync_call, name, arguments)
        elif inspect.iscoroutinefunction(async_call):
            result = await async_call(name, arguments)
        else:
            result = await asyncio.to_thread(async_call, name, arguments)
            if inspect.isawaitable(result):
                result = await result
    except ToolExecutionError as exc:
        emit_event(
            observer,
            "tool.recoverable_error",
            run_id,
            tool=name,
            error_type=type(exc).__name__,
        )
        result = f"Error executing {name}: {exc}"
    return result if isinstance(result, str) else str(result)
