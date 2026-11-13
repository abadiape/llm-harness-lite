"""One backend for OpenAI + Ollama + vLLM + OpenRouter (all OpenAI-compatible)."""
from __future__ import annotations
import json, time
from urllib.parse import urlsplit
from openai import OpenAI
from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, ToolSpec, Usage
from llm_harness_lite.utils.costs import CostManager, estimate_cost

class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, cost_manager: CostManager | None = None,
                 local: bool | None = None):
        import os
        self.model = model
        self.cost_manager = cost_manager
        host = urlsplit(base_url or "").hostname
        self.local = local if local is not None else host in {"localhost", "127.0.0.1", "::1"}
        self.client = OpenAI(base_url=base_url,
                             api_key=api_key or os.getenv("OPENAI_API_KEY", "dummy"))

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse:
        t0 = time.time()
        oai_msgs = []
        for m in messages:
            if m.role == "tool":
                oai_msgs.append({"role": "tool", "tool_call_id": m.tool_call_id,
                                 "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                oai_msgs.append({"role": "assistant", "content": m.content or None,
                    "tool_calls": [{"id": tc.id, "type": "function",
                        "function": {"name": tc.name,
                                     "arguments": json.dumps(tc.arguments)}} for tc in m.tool_calls]})
            else:
                oai_msgs.append({"role": m.role, "content": m.content})
        kwargs: dict = {"model": self.model, "messages": oai_msgs, "temperature": temperature}
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        out = choice.message
        tcs = [ToolCallRequest(id=tc.id, name=tc.function.name,
               arguments=json.loads(tc.function.arguments or "{}"))
               for tc in (out.tool_calls or [])]
        u = resp.usage
        in_t, out_t = (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
        return ChatResponse(
            message=Message(role="assistant", content=out.content or "", tool_calls=tcs),
            finish="tool_calls" if tcs else ("stop" if choice.finish_reason == "stop" else "stop"),
            usage=Usage(input_tokens=in_t, output_tokens=out_t,
                        cost_usd=estimate_cost(self.model, in_t, out_t,
                                               manager=self.cost_manager, local=self.local),
                        model=self.model, latency_s=time.time() - t0))
