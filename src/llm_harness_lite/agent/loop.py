"""Async-first, framework-agnostic agent loop."""
from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import dataclass, field
from enum import Enum

from llm_harness_lite.llm.types import Message, Usage
from llm_harness_lite.tools.registry import ToolExecutionError

SYSTEM = (
    "You are a careful agent. Think step by step. "
    "Call one tool at a time. When finished, answer WITHOUT further tool calls. "
    "If a tool errors, try to recover or explain."
)


class AgentStopReason(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    INCOMPLETE = "incomplete"
    MODEL_ERROR = "model_error"
    INVALID_RESPONSE = "invalid_response"


@dataclass
class AgentResult:
    answer: str
    steps: int
    usage: Usage
    history: list[Message] = field(default_factory=list)
    stop_reason: AgentStopReason = AgentStopReason.COMPLETED


class Agent:
    """Run a bounded tool-using loop; await ``arun`` in async applications."""

    def __init__(
        self,
        llm,
        registry,
        max_steps: int = 8,
        budget_usd: float = 0.05,
        system: str = SYSTEM,
    ):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if (isinstance(budget_usd, bool) or not isinstance(budget_usd, (int, float))
                or not math.isfinite(budget_usd) or budget_usd < 0):
            raise ValueError("budget_usd must be finite and non-negative")
        self.llm, self.registry = llm, registry
        self.max_steps, self.budget = max_steps, float(budget_usd)
        self.system = system

    async def _chat(self, messages, tools, temperature):
        async_chat = getattr(self.llm, "achat", None)
        if async_chat is None:
            return await asyncio.to_thread(
                self.llm.chat, messages, tools=tools, temperature=temperature
            )
        if inspect.iscoroutinefunction(async_chat):
            return await async_chat(messages, tools=tools, temperature=temperature)
        result = await asyncio.to_thread(
            async_chat, messages, tools=tools, temperature=temperature
        )
        return await result if inspect.isawaitable(result) else result

    async def _call_tool(self, name, arguments):
        try:
            async_call = getattr(self.registry, "acall", None)
            if async_call is None:
                result = await asyncio.to_thread(self.registry.call, name, arguments)
            elif inspect.iscoroutinefunction(async_call):
                result = await async_call(name, arguments)
            else:
                result = await asyncio.to_thread(async_call, name, arguments)
                if inspect.isawaitable(result):
                    result = await result
        except ToolExecutionError as exc:
            result = f"Error executing {name}: {exc}"
        return result if isinstance(result, str) else str(result)

    @staticmethod
    def _result(answer, steps, usage, history, stop_reason):
        return AgentResult(
            answer=answer,
            steps=steps,
            usage=usage,
            history=history,
            stop_reason=stop_reason,
        )

    async def arun(self, goal: str, temperature: float = 0.2) -> AgentResult:
        """Run the agent asynchronously; sync backends and tools run in threads."""
        messages = [
            Message(role="system", content=self.system),
            Message(role="user", content=goal),
        ]
        total = Usage(model=getattr(self.llm, "model", ""))
        last_answer = ""

        for step in range(1, self.max_steps + 1):
            if step > 1 and total.cost_usd >= self.budget:
                return self._result(
                    last_answer or "Budget exhausted before the agent could complete.",
                    step - 1,
                    total,
                    messages,
                    AgentStopReason.BUDGET_EXCEEDED,
                )

            response = await self._chat(
                messages, self.registry.specs(), temperature
            )
            total.input_tokens += response.usage.input_tokens
            total.output_tokens += response.usage.output_tokens
            total.cost_usd += response.usage.cost_usd
            total.latency_s += response.usage.latency_s
            messages.append(response.message)
            last_answer = response.message.content

            if response.finish == "error":
                return self._result(
                    last_answer, step, total, messages, AgentStopReason.MODEL_ERROR
                )
            if response.finish == "length":
                return self._result(
                    last_answer, step, total, messages, AgentStopReason.INCOMPLETE
                )
            if response.finish == "tool_calls" and not response.message.tool_calls:
                return self._result(
                    last_answer or "Model requested tools but returned no tool calls.",
                    step,
                    total,
                    messages,
                    AgentStopReason.INVALID_RESPONSE,
                )
            if response.finish != "tool_calls":
                reason = (
                    AgentStopReason.BUDGET_EXCEEDED
                    if total.cost_usd > self.budget
                    else AgentStopReason.COMPLETED
                )
                return self._result(last_answer, step, total, messages, reason)

            if total.cost_usd >= self.budget:
                for tool_call in response.message.tool_calls:
                    messages.append(
                        Message(
                            role="tool",
                            content="Tool call skipped because the agent budget is exhausted.",
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                    )
                return self._result(
                    last_answer or "Budget exhausted before requested tools could run.",
                    step,
                    total,
                    messages,
                    AgentStopReason.BUDGET_EXCEEDED,
                )

            for tool_call in response.message.tool_calls:
                result = await self._call_tool(tool_call.name, tool_call.arguments)
                messages.append(
                    Message(
                        role="tool",
                        content=result[:8000],
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        return self._result(
            last_answer or "Maximum agent steps reached before a final answer.",
            self.max_steps,
            total,
            messages,
            AgentStopReason.MAX_STEPS,
        )

    def run(self, goal: str, temperature: float = 0.2) -> AgentResult:
        """Synchronous convenience wrapper; use ``await arun`` inside async code."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(goal, temperature))
        raise RuntimeError("Agent.run() cannot run inside an event loop; await Agent.arun()")
