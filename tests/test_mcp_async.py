import asyncio
from types import SimpleNamespace

import pytest

from llm_harness_lite.mcp.client import PersistentMCP
from llm_harness_lite.mcp.client_async import MCPClient
from llm_harness_lite.tools.registry import ToolExecutionError, ToolRegistry


class FakeSession:
    def __init__(self):
        self.initialized = False
        self.closed = False
        self.started = None
        self.cancelled = False
        self.close_started = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        if self.close_started:
            self.close_started.set()
            await asyncio.Event().wait()
        self.closed = True

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name="search",
                description="Search remote docs",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ])

    async def call_tool(self, name, arguments):
        if self.started:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"{name}: {arguments['query']}")],
            isError=False,
        )


def test_mcp_connect_close_and_close_idempotently(monkeypatch):
    async def scenario():
        client = MCPClient(command="fake-server", timeout=1)
        session = FakeSession()

        async def enter_transport(stack):
            return "read", "write"

        monkeypatch.setattr(client, "_enter_transport", enter_transport)
        monkeypatch.setattr(client, "_session_factory", lambda: lambda *_: session)
        await client.connect()

        assert session.initialized
        assert client.connected
        assert client._session is session
        await client.close()
        await client.close()
        assert session.closed
        assert not client.connected
        assert client._session is None
    asyncio.run(scenario())

def test_persistent_mcp_bridge_closes_its_loop(monkeypatch):
    async def scenario():
        bridge = PersistentMCP(command="fake-server", timeout=1)
        calls = []

        async def connect():
            calls.append("connect")

        async def close():
            calls.append("close")

        monkeypatch.setattr(bridge._client, "connect", connect)
        monkeypatch.setattr(bridge._client, "aclose", close)
        bridge.start()
        bridge.stop()

        assert calls == ["connect", "close"]
        assert bridge._loop is None
    asyncio.run(scenario())

def test_mcp_connect_failure_closes_entered_resources(monkeypatch):
    async def scenario():
        client = MCPClient(command="fake-server", timeout=1)
        session = FakeSession()

        async def enter_transport(stack):
            return "read", "write"

        async def fail_initialize():
            raise ValueError("initialization failed")

        session.initialize = fail_initialize
        monkeypatch.setattr(client, "_enter_transport", enter_transport)
        monkeypatch.setattr(client, "_session_factory", lambda: lambda *_: session)

        with pytest.raises(ValueError, match="initialization failed"):
            await client.connect()
        assert session.closed
        assert client._session is None
    asyncio.run(scenario())

def test_mcp_tools_keep_remote_schema_and_call_async():
    async def scenario():
        client = MCPClient(command="fake-server")
        session = FakeSession()
        client._session = session
        registry = ToolRegistry()

        assert await client.register_into(registry, prefix="mcp_") == ["mcp_search"]
        spec = registry.specs()[0]
        assert spec.parameters["required"] == ["query"]
        assert await registry.acall("mcp_search", {"query": "billing"}) == (
            "search: billing"
        )
    asyncio.run(scenario())

def test_mcp_call_timeout_cancels_remote_request():
    async def scenario():
        client = MCPClient(command="fake-server", timeout=0.01)
        session = FakeSession()
        session.started = asyncio.Event()
        client._session = session

        with pytest.raises(ToolExecutionError, match="timed out"):
            await client.call("search", {"query": "wait"})
        assert session.cancelled
    asyncio.run(scenario())

def test_mcp_call_propagates_external_cancellation():
    async def scenario():
        client = MCPClient(command="fake-server", timeout=10)
        session = FakeSession()
        session.started = asyncio.Event()
        client._session = session
        task = asyncio.create_task(client.call("search", {"query": "wait"}))
        await session.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.cancelled
    asyncio.run(scenario())

def test_mcp_close_has_timeout_and_marks_client_closed(monkeypatch):
    async def scenario():
        client = MCPClient(command="fake-server", timeout=0.01)
        session = FakeSession()
        session.close_started = asyncio.Event()

        async def enter_transport(stack):
            return "read", "write"

        monkeypatch.setattr(client, "_enter_transport", enter_transport)
        monkeypatch.setattr(client, "_session_factory", lambda: lambda *_: session)
        await client.connect()

        with pytest.raises(asyncio.TimeoutError):
            await client.close()
        assert session.close_started.is_set()
        assert not client.connected
    asyncio.run(scenario())

def test_mcp_rejects_invalid_lifecycle_configuration():
    with pytest.raises(ValueError, match="command is required"):
        MCPClient()
    with pytest.raises(ValueError, match="timeout"):
        MCPClient(command="fake-server", timeout=0)
