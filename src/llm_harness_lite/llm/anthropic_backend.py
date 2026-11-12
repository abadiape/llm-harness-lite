"""Anthropic backend, translated to/from normalized types."""
from __future__ import annotations
import time
import anthropic
from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, ToolSpec, Usage
from llm_harness_lite.utils.costs import estimate_cost

class AnthropicBackend:
    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic()

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse:
        t0 = time.time()
        system = "\n".join(m.content for m in messages if m.role == "system")
        conv: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                conv.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id,
                     "content": m.content}]})
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name,
                            "input": tc.arguments} for tc in m.tool_calls]
                conv.append({"role": "assistant", "content": blocks})
            else:
                conv.append({"role": m.role, "content": m.content})
        kwargs: dict = {"model": self.model, "max_tokens": 1024,
                        "temperature": temperature, "messages": conv}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description,
                                "input_schema": t.parameters} for t in tools]
        resp = self.client.messages.create(**kwargs)
        text, tcs = "", []
        for b in resp.content:
            if b.type == "text":
                text += b.text
            elif b.type == "tool_use":
                tcs.append(ToolCallRequest(id=b.id, name=b.name, arguments=dict(b.input)))
        return ChatResponse(
            message=Message(role="assistant", content=text, tool_calls=tcs),
            finish="tool_calls" if resp.stop_reason == "tool_use" else "stop",
            usage=Usage(input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                        cost_usd=estimate_cost(self.model, resp.usage.input_tokens,
                                              resp.usage.output_tokens),
                        model=self.model, latency_s=time.time() - t0))
