"""Persistent MCP client.

- Async-first: `MCPClient` holds ONE stdio/SSE session open across calls.
- Sync-friendly: `PersistentMCP` runs that session on a background loop so
  the sync Agent/ToolRegistry can call it without reconnecting per call.
- Registry adapter: `register_mcp_into(registry, ...)` exposes remote tools
  as local ToolSpecs with a sync bridge.

Requires: pip install mcp
"""
from __future__ import annotations
import asyncio
import threading
from contextlib import AsyncExitStack
from typing import Any

# ---------- async persistent client ----------

class MCPClient:
    """One persistent connection to one MCP server."""
    def __init__(self, command: str | None = None, args: list[str] | None = None,
                 url: str | None = None, transport: str = "stdio"):
        self.command, self.args = command, args or []
        self.url, self.transport = url, transport
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def connect(self):
        from mcp import ClientSession
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        if self.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(command=self.command, args=self.args)
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif self.transport in ("sse", "http"):
            try:
                from mcp.client.sse import sse_client
                read, write = await self._stack.enter_async_context(sse_client(self.url))
            except ImportError:
                from mcp.client.streamable_http import streamablehttp_client as _hc
                read, write, _ = await self._stack.enter_async_context(_hc(self.url))
        else:
            raise ValueError(f"unknown transport {self.transport}")
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def close(self):
        if self._stack:
            await self._stack.aclose()
            self._stack, self._session = None, None

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, *exc):
        await self.close()

    async def tool_specs(self):
        from llm_harness_lite.llm.types import ToolSpec
        tools = (await self._session.list_tools()).tools
        return [ToolSpec(name=t.name, description=t.description or t.name,
                         parameters=t.inputSchema or {"type": "object", "properties": {}})
                for t in tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        res = await self._session.call_tool(name, arguments)
        parts = []
        for b in (res.content or []):
            parts.append(getattr(b, "text", None) or getattr(b, "data", None) or str(b))
        return "\n".join(p for p in parts if p) or "[empty MCP result]"


# ---------- sync bridge on a background loop (for sync Agent) ----------

def _run_blocking(coro, loop: asyncio.AbstractEventLoop, timeout: float = 60.0):
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result(timeout)

class PersistentMCP:
    """Keeps an MCPClient alive on a daemon thread so sync code reuses one session.

    Usage:
        mcp = PersistentMCP(command='npx', args=[...]).start()
        mcp.register_into(registry)
        mcp.call('read_file', {'path': '/tmp/x'})
        mcp.stop()
    """
    def __init__(self, command: str | None = None, args: list[str] | None = None,
                 url: str | None = None, transport: str = "stdio"):
        self._client = MCPClient(command, url=url, transport=transport,
                                 args=args)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "PersistentMCP":
        self._loop = asyncio.new_event_loop()
        def _run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        _run_blocking(self._client.connect(), self._loop)
        return self

    def stop(self):
        if self._loop:
            _run_blocking(self._client.close(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)

    def specs(self):
        return _run_blocking(self._client.tool_specs(), self._loop)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        return _run_blocking(self._client.call(name, arguments), self._loop)

    def register_into(self, registry, prefix: str = "") -> list[str]:
        """Expose all remote tools as local registry tools (sync bridge)."""
        names = []
        for spec in self.specs():
            lname = f"{prefix}{spec.name}"
            def _make(remote=spec.name, s=self):
                def _fn(**kw):
                    return s.call(remote, kw)
                _fn.__name__ = lname
                _fn.__doc__ = spec.description
                return _fn
            registry.register(_make(), name=lname, description=spec.description)
            names.append(lname)
        return names


# Backwards-compat helper used in earlier scaffold
async def load_mcp_tools_stdio(registry, command: str, args: list[str] | None = None):
    """One-shot load (connects, registers, closes). For prod use PersistentMCP."""
    async with MCPClient(command=command, args=args or []) as c:
        specs = await c.tool_specs()
        # NOTE: one-shot sessions can't serve later calls; register a lazy
        # reconnecting stub so old examples keep working.
        for spec in specs:
            def _make(s=spec, cmd=command, a=args or []):
                def _fn(**kw):
                    async def _once():
                        async with MCPClient(command=cmd, args=a) as cc:
                            return await cc.call(s.name, kw)
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        return asyncio.run(_once())
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        return ex.submit(asyncio.run, _once()).result(timeout=60)
                _fn.__name__ = s.name
                _fn.__doc__ = s.description
                return _fn
            registry.register(_make(), name=spec.name, description=spec.description)
        return [s.name for s in specs]
