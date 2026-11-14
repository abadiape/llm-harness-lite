"""Synchronous compatibility bridge for the async MCP client."""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from llm_harness_lite.llm.types import ToolSpec
from llm_harness_lite.mcp.client_async import MCPClient


def _run_blocking(coro, loop: asyncio.AbstractEventLoop, timeout: float):
    if not loop.is_running() or loop.is_closed():
        coro.close()
        raise RuntimeError("MCP event loop is not running")
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        coro.close()
        raise RuntimeError("MCP sync bridge cannot be called from its own event loop")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout)
    except TimeoutError:
        future.cancel()
        raise


class PersistentMCP:
    """Keep one async MCP session open for synchronous callers."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        transport: str = "stdio",
        timeout: float = 60.0,
    ):
        self._client = MCPClient(
            command, args=args, url=url, transport=transport, timeout=timeout
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> PersistentMCP:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("MCP bridge is already running")
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def serve():
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

        self._loop = loop
        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        if not ready.wait(self._client.timeout):
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(self._client.timeout + 1)
            if self._thread.is_alive():
                raise TimeoutError("MCP event loop failed to stop after startup timeout")
            self._loop, self._thread = None, None
            raise TimeoutError("MCP event loop failed to start")
        connected = False
        try:
            _run_blocking(
                self._client.connect(), loop, self._client.timeout + 1
            )
            connected = True
            return self
        finally:
            if not connected:
                self.stop()

    def stop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            if loop.is_running():
                _run_blocking(
                    self._client.aclose(), loop, self._client.timeout + 1
                )
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            thread.join(self._client.timeout + 1)
            if thread.is_alive():
                raise TimeoutError("MCP event loop did not stop in time")
            self._loop, self._thread = None, None

    def specs(self) -> list[ToolSpec]:
        if self._loop is None:
            raise RuntimeError("MCP bridge is not started")
        return _run_blocking(
            self._client.tool_specs(), self._loop, self._client.timeout + 1
        )

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self._loop is None:
            raise RuntimeError("MCP bridge is not started")
        return _run_blocking(
            self._client.call(name, arguments),
            self._loop,
            self._client.timeout + 1,
        )

    def register_into(self, registry, prefix: str = "") -> list[str]:
        registered = []
        for remote_spec in self.specs():
            local_name = f"{prefix}{remote_spec.name}"

            def make_call(remote_name):
                def call_remote(**arguments):
                    return self.call(remote_name, arguments)
                return call_remote

            spec = ToolSpec(
                name=local_name,
                description=remote_spec.description,
                parameters=remote_spec.parameters,
            )
            registry.register_spec(make_call(remote_spec.name), spec)
            registered.append(local_name)
        return registered


async def load_mcp_tools_stdio(
    registry,
    command: str,
    args: list[str] | None = None,
    timeout: float = 60.0,
) -> list[str]:
    """Register tools that open a short-lived connection for each invocation."""
    async with MCPClient(command, args=args or [], timeout=timeout) as client:
        specs = await client.tool_specs()
    registered = []
    for spec in specs:
        def make_call(remote_name):
            async def call_remote(**arguments):
                async with MCPClient(
                    command, args=args or [], timeout=timeout
                ) as client:
                    return await client.call(remote_name, arguments)
            return call_remote

        registry.register_spec(make_call(spec.name), spec)
        registered.append(spec.name)
    return registered
