"""Tool registry: python function -> ToolSpec (JSON Schema via inspect)."""
from __future__ import annotations
import inspect, json
from dataclasses import dataclass
from typing import Any, Callable
from llm_harness_lite.llm.types import ToolSpec

_PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
            list: "array", dict: "object"}

@dataclass
class _Entry:
    fn: Callable
    spec: ToolSpec

class ToolRegistry:
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

    def specs(self) -> list[ToolSpec]:
        return [e.spec for e in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {list(self._tools)}"
        try:
            out = self._tools[name].fn(**arguments)
            return out if isinstance(out, str) else json.dumps(out, default=str)
        except Exception as e:
            return f"Error executing {name}: {e}"

def tool(registry: ToolRegistry, **kw):
    def deco(fn: Callable):
        registry.register(fn, **kw)
        return fn
    return deco
