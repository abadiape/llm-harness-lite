"""Synchronous compatibility bridge for the async MCP client."""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any

from llm_harness_lite.llm.types import ToolSpec

from .client_async import MCPClient


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
        self._commands: asyncio.Queue | None = None
        self._ready: threading.Event | None = None
        self._startup_error: Exception | None = None

    def start(self) -> PersistentMCP:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("MCP bridge is already running")
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        self._ready = ready
        self._startup_error = None

        def serve():
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._serve())
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
        if not ready.wait(self._client.timeout + 1):
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(self._client.timeout + 1)
            if self._thread.is_alive():
                raise TimeoutError("MCP event loop failed to stop after startup timeout")
            self._loop, self._thread = None, None
            raise TimeoutError("MCP event loop failed to start")
        if self._startup_error is not None:
            self._thread.join(self._client.timeout + 1)
            self._loop, self._thread = None, None
            raise self._startup_error
        return self

    def stop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            if loop.is_running():
                self._request("stop")
        finally:
            thread.join(self._client.timeout + 1)
            if thread.is_alive():
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
                thread.join(self._client.timeout + 1)
            if thread.is_alive():
                raise TimeoutError("MCP event loop did not stop in time")
            self._loop, self._thread = None, None
            self._commands = None
            self._ready = None

    def specs(self) -> list[ToolSpec]:
        if self._loop is None:
            raise RuntimeError("MCP bridge is not started")
        return self._request("specs")

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self._loop is None:
            raise RuntimeError("MCP bridge is not started")
        return self._request("call", name, arguments)

    async def _serve(self) -> None:
        commands: asyncio.Queue = asyncio.Queue()
        self._commands = commands
        try:
            await self._client.connect()
        except Exception as exc:  # noqa: BLE001
            self._startup_error = exc
            if self._ready is not None:
                self._ready.set()
            return
        if self._ready is not None:
            self._ready.set()

        try:
            while True:
                operation, args, response = await commands.get()
                if not response.set_running_or_notify_cancel():
                    continue
                if operation == "stop":
                    try:
                        await self._client.aclose()
                    except Exception as exc:  # noqa: BLE001
                        response.set_exception(exc)
                    else:
                        response.set_result(None)
                    return
                try:
                    if operation == "specs":
                        result = await self._client.tool_specs()
                    elif operation == "call":
                        result = await self._client.call(*args)
                    else:
                        raise RuntimeError(f"unknown MCP bridge operation {operation!r}")
                except Exception as exc:  # noqa: BLE001
                    response.set_exception(exc)
                else:
                    response.set_result(result)
        finally:
            self._commands = None

    def _request(self, operation: str, *args):
        loop, commands = self._loop, self._commands
        if loop is None or commands is None or not loop.is_running():
            raise RuntimeError("MCP event loop is not running")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            raise RuntimeError("MCP sync bridge cannot be called from its own event loop")

        response = Future()
        loop.call_soon_threadsafe(
            commands.put_nowait, (operation, args, response)
        )
        try:
            return response.result(self._client.timeout + 1)
        except TimeoutError:
            response.cancel()
            raise

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
