"""Framework-agnostic agent loop. Works with any LLMBackend."""
from __future__ import annotations
from dataclasses import dataclass, field
from llm_harness_lite.llm.types import Message, Usage

SYSTEM = ("You are a careful agent. Think step by step. "
          "Call one tool at a time. When finished, answer WITHOUT further tool calls. "
          "If a tool errors, try to recover or explain.")

@dataclass
class AgentResult:
    answer: str
    steps: int
    usage: Usage
    history: list[Message] = field(default_factory=list)

class Agent:
    def __init__(self, llm, registry, max_steps: int = 8,
                 budget_usd: float = 0.05, system: str = SYSTEM):
        self.llm, self.registry = llm, registry
        self.max_steps, self.budget = max_steps, budget_usd
        self.system = system

    def run(self, goal: str, temperature: float = 0.2) -> AgentResult:
        msgs = [Message(role="system", content=self.system),
                Message(role="user", content=goal)]
        total = Usage(model=getattr(self.llm, "model", ""))
        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(msgs, tools=self.registry.specs(), temperature=temperature)
            total.input_tokens += resp.usage.input_tokens
            total.output_tokens += resp.usage.output_tokens
            total.cost_usd += resp.usage.cost_usd
            total.latency_s += resp.usage.latency_s
            msgs.append(resp.message)
            if resp.finish != "tool_calls" or not resp.message.tool_calls:
                return AgentResult(answer=resp.message.content, steps=step,
                                   usage=total, history=msgs)
            for tc in resp.message.tool_calls:
                result = self.registry.call(tc.name, tc.arguments)
                msgs.append(Message(role="tool", content=result[:8000],
                                    tool_call_id=tc.id, name=tc.name))
            if total.cost_usd > self.budget:
                return AgentResult(answer="[BUDGET EXCEEDED] " + msgs[-2].content,
                                   steps=step, usage=total, history=msgs)
        return AgentResult(answer="[MAX STEPS] " + msgs[-1].content,
                           steps=self.max_steps, usage=total, history=msgs)
