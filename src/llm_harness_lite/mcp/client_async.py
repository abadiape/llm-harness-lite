"""Async MCP client with explicit connection and request lifecycles."""
from __future__ import annotations

import asyncio
import math
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from llm_harness_lite.llm.types import ToolSpec
from llm_harness_lite.tools.registry import ToolExecutionError


@asynccontextmanager
async def _timeout_after(timeout: float):
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("timeout requires a running asyncio task")
    expired = False

    def cancel_task():
        nonlocal expired
        expired = True
        task.cancel()

    handle = asyncio.get_running_loop().call_later(timeout, cancel_task)
    try:
        yield
    except asyncio.CancelledError as exc:
        if expired:
            raise TimeoutError from exc
        raise
    else:
        if expired:
            raise TimeoutError
    finally:
        handle.cancel()


class MCPClient:
    """Own one MCP server connection; close it with ``aclose`` or ``async with``."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        transport: str = "stdio",
        timeout: float = 60.0,
    ):
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        if transport not in {"stdio", "sse", "http", "streamable_http"}:
            raise ValueError(f"unknown MCP transport {transport!r}")
        if transport == "stdio" and not command:
            raise ValueError("command is required for stdio transport")
        if transport != "stdio" and not url:
            raise ValueError(f"url is required for {transport} transport")
        self.command, self.args = command, args or []
        self.url, self.transport = url, transport
        self.timeout = float(timeout)
        self._stack: AsyncExitStack | None = None
        self._session = None
        self._connecting = False

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def _enter_transport(self, stack: AsyncExitStack):
        if self.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=self.command, args=self.args)
            async with _timeout_after(self.timeout):
                streams = await stack.enter_async_context(stdio_client(params))
            return streams
        if self.transport == "sse":
            from mcp.client.sse import sse_client

            async with _timeout_after(self.timeout):
                return await stack.enter_async_context(sse_client(self.url))
        from mcp.client.streamable_http import streamablehttp_client

        async with _timeout_after(self.timeout):
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(self.url)
            )
        return read, write

    def _session_factory(self):
        from mcp import ClientSession

        return ClientSession

    async def connect(self) -> MCPClient:
        if self._connecting or self._stack is not None:
            raise RuntimeError("MCP client is already connected or connecting")
        self._connecting = True
        stack = AsyncExitStack()
        connected = False
        try:
            await stack.__aenter__()
            read, write = await self._enter_transport(stack)
            async with _timeout_after(self.timeout):
                session = await stack.enter_async_context(
                    self._session_factory()(read, write)
                )
            async with _timeout_after(self.timeout):
                await session.initialize()
            self._stack, self._session = stack, session
            connected = True
            return self
        finally:
            self._connecting = False
            if not connected:
                async with _timeout_after(self.timeout):
                    await stack.aclose()

    async def aclose(self) -> None:
        if self._connecting:
            raise RuntimeError("cannot close an MCP client while it is connecting")
        if self._stack is None:
            return
        stack = self._stack
        try:
            async with _timeout_after(self.timeout):
                await stack.aclose()
        finally:
            self._stack, self._session = None, None

    async def close(self) -> None:
        """Backward-compatible alias for ``aclose``."""
        await self.aclose()

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _require_session(self):
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        return self._session

    async def tool_specs(self) -> list[ToolSpec]:
        async with _timeout_after(self.timeout):
            from_tools = await self._require_session().list_tools()
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or tool.name,
                parameters=(
                    getattr(tool, "input_schema", None)
                    or getattr(tool, "inputSchema", None)
                    or {"type": "object", "properties": {}}
                ),
            )
            for tool in from_tools.tools
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            async with _timeout_after(self.timeout):
                result = await self._require_session().call_tool(name, arguments)
        except TimeoutError as exc:
            raise ToolExecutionError(
                f"MCP tool {name!r} timed out after {self.timeout:g}s"
            ) from exc
        parts = []
        for block in result.content or []:
            content = getattr(block, "text", None) or getattr(block, "data", None)
            if content:
                parts.append(str(content))
        text = "\n".join(parts) or "[empty MCP result]"
        if getattr(result, "isError", False):
            raise ToolExecutionError(text)
        return text

    async def register_into(self, registry, prefix: str = "") -> list[str]:
        """Register remote tools with their original schemas for Agent.arun."""
        registered = []
        for remote_spec in await self.tool_specs():
            local_name = f"{prefix}{remote_spec.name}"

            def make_call(remote_name):
                async def call_remote(**arguments):
                    return await self.call(remote_name, arguments)
                return call_remote

            local_spec = ToolSpec(
                name=local_name,
                description=remote_spec.description,
                parameters=remote_spec.parameters,
            )
            registry.register_spec(make_call(remote_spec.name), local_spec)
            registered.append(local_name)
        return registered
