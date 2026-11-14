"""Sync and async Anthropic backend."""
from __future__ import annotations

import math
import time

import anthropic

from llm_harness_lite.llm.types import (
    ChatResponse,
    Message,
    ToolCallRequest,
    ToolSpec,
    Usage,
)
from llm_harness_lite.utils.costs import CostManager, estimate_cost


class AnthropicBackend:
    def __init__(
        self,
        model: str,
        cost_manager: CostManager | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        self.model = model
        self.cost_manager = cost_manager
        self.timeout = float(timeout)
        self._client_kwargs = {"api_key": api_key, "timeout": self.timeout}
        self.client = anthropic.Anthropic(**self._client_kwargs)
        self._async_client: anthropic.AsyncAnthropic | None = None

    @staticmethod
    def _messages(messages: list[Message]) -> tuple[str, list[dict]]:
        system = "\n".join(message.content for message in messages if message.role == "system")
        conversation = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                conversation.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }],
                })
            elif message.role == "assistant" and message.tool_calls:
                blocks = [{"type": "text", "text": message.content}] if message.content else []
                blocks.extend({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                } for call in message.tool_calls)
                conversation.append({"role": "assistant", "content": blocks})
            else:
                conversation.append({"role": message.role, "content": message.content})
        return system, conversation

    def _request(self, messages: list[Message], tools: list[ToolSpec] | None,
                 temperature: float) -> dict:
        system, conversation = self._messages(messages)
        request = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": conversation,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [{
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.parameters,
            } for spec in tools]
        return request

    def _normalize(self, response, started: float) -> ChatResponse:
        text = ""
        calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input),
                ))
        finish = "tool_calls" if response.stop_reason == "tool_use" else (
            "length" if response.stop_reason == "max_tokens" else "stop"
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return ChatResponse(
            message=Message(role="assistant", content=text, tool_calls=calls),
            finish=finish,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=estimate_cost(
                    self.model, input_tokens, output_tokens, manager=self.cost_manager
                ),
                model=self.model,
                latency_s=time.perf_counter() - started,
            ),
        )

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse:
        started = time.perf_counter()
        response = self.client.messages.create(
            **self._request(messages, tools, temperature)
        )
        return self._normalize(response, started)

    async def achat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                    temperature: float = 0.2) -> ChatResponse:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(**self._client_kwargs)
        started = time.perf_counter()
        response = await self._async_client.messages.create(
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
