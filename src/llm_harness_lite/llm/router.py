"""Model-agnostic LLM routing through an explicit application policy."""
from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from llm_harness_lite.llm.base import LLMBackend
from llm_harness_lite.llm.types import ChatResponse, Message, ToolSpec


@dataclass(frozen=True)
class RoutingRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] | None
    temperature: float
    candidates: tuple[str, ...]


RoutingPolicy = Callable[[RoutingRequest], str]


class LLMRouter:
    """Dispatch each request to the named backend selected by ``policy``.

    Policies should be deterministic for equivalent requests so cache lookups
    and subsequent dispatch select the same backend.
    """

    def __init__(self, routes: Mapping[str, LLMBackend], policy: RoutingPolicy):
        if not routes or any(not isinstance(name, str) or not name.strip() for name in routes):
            raise ValueError("routes must be a non-empty mapping with non-empty names")
        if not callable(policy):
            raise TypeError("policy must be callable")
        for backend in routes.values():
            if not isinstance(getattr(backend, "model", None), str):
                raise TypeError("each routed backend must expose a string model")
        self.routes = dict(routes)
        self.policy = policy

    @property
    def model(self) -> str:
        route_models = sorted(f"{name}={backend.model}" for name, backend in self.routes.items())
        return "router[" + ",".join(route_models) + "]"

    def _select(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        temperature: float,
    ) -> tuple[str, LLMBackend]:
        request = RoutingRequest(
            messages=tuple(messages),
            tools=None if tools is None else tuple(tools),
            temperature=temperature,
            candidates=tuple(self.routes),
        )
        name = self.policy(request)
        if not isinstance(name, str) or name not in self.routes:
            raise ValueError(f"routing policy selected unknown route {name!r}")
        return name, self.routes[name]

    def model_for_request(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
    ) -> str:
        name, backend = self._select(messages, tools, temperature)
        return json.dumps([name, backend.model], separators=(",", ":"))

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse:
        _, backend = self._select(messages, tools, temperature)
        return backend.chat(messages, tools=tools, temperature=temperature)

    async def achat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                    temperature: float = 0.2) -> ChatResponse:
        _, backend = self._select(messages, tools, temperature)
        async_chat = getattr(backend, "achat", None)
        if async_chat is None:
            return await asyncio.to_thread(
                backend.chat, messages, tools=tools, temperature=temperature
            )
        if inspect.iscoroutinefunction(async_chat):
            return await async_chat(messages, tools=tools, temperature=temperature)
        result = await asyncio.to_thread(
            async_chat, messages, tools=tools, temperature=temperature
        )
        return await result if inspect.isawaitable(result) else result
