"""Tool registry: python function -> ToolSpec (JSON Schema via inspect)."""
from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_harness_lite.llm.types import ToolSpec

_PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
            list: "array", dict: "object"}

@dataclass
class _Entry:
    fn: Callable
    spec: ToolSpec


class ToolExecutionError(Exception):
    """A recoverable tool failure that an agent may return to the model."""


class ToolRegistry:
    """Register tools; only explicit ToolExecutionError failures are recoverable."""

    def __init__(self):
        self._tools: dict[str, _Entry] = {}

    def register(self, fn: Callable, name: str | None = None,
                 description: str | None = None) -> ToolSpec:
        sig = inspect.signature(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            ann = p.annotation if p.annotation is not inspect._empty else str
            props[pname] = {"type": _PY2JSON.get(ann, "string")}
            if p.default is inspect._empty:
                required.append(pname)
        spec = ToolSpec(name=name or fn.__name__,
                        description=description or (fn.__doc__ or "").strip() or fn.__name__,
                        parameters={"type": "object", "properties": props,
                                    "required": required})
        self._tools[spec.name] = _Entry(fn, spec)
        return spec

    def register_spec(self, fn: Callable, spec: ToolSpec) -> ToolSpec:
        """Register a callable with an existing schema, e.g. a remote tool schema."""
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} is already registered")
        self._tools[spec.name] = _Entry(fn, spec)
        return spec

    def specs(self) -> list[ToolSpec]:
        return [e.spec for e in self._tools.values()]

    @staticmethod
    def _serialize(out: Any) -> str:
        return out if isinstance(out, str) else json.dumps(out, default=str)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {list(self._tools)}"
        fn = self._tools[name].fn
        if inspect.iscoroutinefunction(fn):
            return f"Error executing {name}: asynchronous tools require registry.acall()."
        out = fn(**arguments)
        if inspect.isawaitable(out):
            if inspect.iscoroutine(out):
                out.close()
            raise TypeError("tool returned an awaitable; use registry.acall()")
        return self._serialize(out)

    async def acall(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {list(self._tools)}"
        fn = self._tools[name].fn
        if inspect.iscoroutinefunction(fn):
            out = await fn(**arguments)
        else:
            out = await asyncio.to_thread(fn, **arguments)
            if inspect.isawaitable(out):
                out = await out
        return self._serialize(out)

def tool(registry: ToolRegistry, **kw):
    def deco(fn: Callable):
        registry.register(fn, **kw)
        return fn
    return deco
