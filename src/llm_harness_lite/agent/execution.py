"""Async adapters for synchronous and asynchronous backend calls."""
from __future__ import annotations

import asyncio
import inspect

from llm_harness_lite.agent.telemetry import AgentObserver, emit_event
from llm_harness_lite.llm.types import Message, ToolSpec
from llm_harness_lite.tools.registry import ToolExecutionError


async def chat_backend(llm, messages: list[Message], tools: list[ToolSpec], temperature: float):
    async_chat = getattr(llm, "achat", None)
    if async_chat is None:
        return await asyncio.to_thread(llm.chat, messages, tools=tools, temperature=temperature)
    if inspect.iscoroutinefunction(async_chat):
        return await async_chat(messages, tools=tools, temperature=temperature)
    result = await asyncio.to_thread(async_chat, messages, tools=tools, temperature=temperature)
    return await result if inspect.isawaitable(result) else result


async def call_tool(registry, name: str, arguments: dict, observer: AgentObserver | None,
                    run_id: str) -> str:
    try:
        async_call = getattr(registry, "acall", None)
        if async_call is None:
            result = await asyncio.to_thread(registry.call, name, arguments)
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
