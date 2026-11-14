"""Sync and async OpenAI-compatible backend."""
from __future__ import annotations

import json
import math
import os
import time
from urllib.parse import urlsplit

from openai import AsyncOpenAI, OpenAI

from llm_harness_lite.llm.types import (
    ChatResponse,
    Message,
    ToolCallRequest,
    ToolSpec,
    Usage,
)
from llm_harness_lite.utils.costs import CostManager, estimate_cost


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cost_manager: CostManager | None = None,
        local: bool | None = None,
        timeout: float = 60.0,
    ):
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        self.model = model
        self.cost_manager = cost_manager
        host = urlsplit(base_url or "").hostname
        self.local = local if local is not None else host in {"localhost", "127.0.0.1", "::1"}
        self.timeout = float(timeout)
        self._client_kwargs = {
            "base_url": base_url,
            "api_key": api_key or os.getenv("OPENAI_API_KEY", "dummy"),
            "timeout": self.timeout,
        }
        self.client = OpenAI(**self._client_kwargs)
        self._async_client: AsyncOpenAI | None = None

    @staticmethod
    def _messages(messages: list[Message]) -> list[dict]:
        converted = []
        for message in messages:
            if message.role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                })
            elif message.role == "assistant" and message.tool_calls:
                converted.append({
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ],
                })
            else:
                converted.append({"role": message.role, "content": message.content})
        return converted

    def _request(self, messages: list[Message], tools: list[ToolSpec] | None,
                 temperature: float) -> dict:
        request = {
            "model": self.model,
            "messages": self._messages(messages),
            "temperature": temperature,
        }
        if tools:
            request["tools"] = [spec.to_openai() for spec in tools]
            request["tool_choice"] = "auto"
        return request

    def _normalize(self, response, started: float) -> ChatResponse:
        choice = response.choices[0]
        message = choice.message
        calls = [
            ToolCallRequest(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (message.tool_calls or [])
        ]
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        finish = "tool_calls" if calls else (
            "length" if choice.finish_reason == "length" else "stop"
        )
        return ChatResponse(
            message=Message(role="assistant", content=message.content or "", tool_calls=calls),
            finish=finish,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=estimate_cost(
                    self.model, input_tokens, output_tokens,
                    manager=self.cost_manager, local=self.local,
                ),
                model=self.model,
                latency_s=time.perf_counter() - started,
            ),
        )

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            **self._request(messages, tools, temperature)
        )
        return self._normalize(response, started)

    async def achat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                    temperature: float = 0.2) -> ChatResponse:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_kwargs)
        started = time.perf_counter()
        response = await self._async_client.chat.completions.create(
            **self._request(messages, tools, temperature)
        )
        return self._normalize(response, started)

    def close(self) -> None:
        self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
        self.client.close()
