"""Async-first, framework-agnostic agent loop."""
from __future__ import annotations

import asyncio
import time
import uuid

from llm_harness_lite.agent.execution import call_tool, chat_backend
from llm_harness_lite.agent.prompts import SYSTEM
from llm_harness_lite.agent.telemetry import AgentObserver, completed_result, emit_event
from llm_harness_lite.agent.types import AgentResult, AgentStopReason
from llm_harness_lite.agent.validation import validate_agent_config, validate_response
from llm_harness_lite.llm.types import Message, Usage


class Agent:
    """Run a bounded tool-using loop; await ``arun`` in async applications."""

    def __init__(
        self,
        llm,
        registry,
        max_steps: int = 8,
        budget_usd: float = 0.05,
        system: str = SYSTEM,
        observer: AgentObserver | None = None,
    ):
        self.llm, self.registry = llm, registry
        self.max_steps = max_steps
        self.budget = validate_agent_config(max_steps, budget_usd, observer)
        self.system = system
        self.observer = observer

    def _emit(self, name: str, run_id: str, **attributes) -> None:
        emit_event(self.observer, name, run_id, **attributes)

    async def arun(self, goal: str, temperature: float = 0.2) -> AgentResult:
        """Run the agent asynchronously; sync backends and tools run in threads."""
        messages = [
            Message(role="system", content=self.system),
            Message(role="user", content=goal),
        ]
        run_id = uuid.uuid4().hex
        total = Usage(model=getattr(self.llm, "model", ""))
        last_answer = ""
        self._emit("agent.started", run_id, max_steps=self.max_steps, budget_usd=self.budget)

        for step in range(1, self.max_steps + 1):
            if step > 1 and total.cost_usd >= self.budget:
                return completed_result(
                    self.observer,
                    run_id,
                    last_answer or "Budget exhausted before the agent could complete.",
                    step - 1,
                    total,
                    messages,
                    AgentStopReason.BUDGET_EXCEEDED,
                )

            started = time.perf_counter()
            try:
                response = await chat_backend(
                    self.llm, messages, self.registry.specs(), temperature
                )
            except Exception as exc:
                self._emit("llm.failed", run_id, step=step, error_type=type(exc).__name__)
                raise
            try:
                validate_response(response)
            except (TypeError, ValueError) as exc:
                self._emit(
                    "llm.invalid_response",
                    run_id,
                    step=step,
                    error_type=type(exc).__name__,
                )
                raise
            self._emit(
                "llm.completed",
                run_id,
                step=step,
                model=response.usage.model,
                finish=response.finish,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.usage.cost_usd,
                latency_s=time.perf_counter() - started,
            )
            total.input_tokens += response.usage.input_tokens
            total.output_tokens += response.usage.output_tokens
            total.cost_usd += response.usage.cost_usd
            total.latency_s += response.usage.latency_s
            messages.append(response.message)
            last_answer = response.message.content

            if response.finish == "error":
                return completed_result(
                    self.observer, run_id, last_answer, step, total, messages,
                    AgentStopReason.MODEL_ERROR,
                )
            if response.finish == "length":
                return completed_result(
                    self.observer, run_id, last_answer, step, total, messages,
                    AgentStopReason.INCOMPLETE,
                )
            if response.finish == "tool_calls" and not response.message.tool_calls:
                return completed_result(
                    self.observer,
                    run_id,
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
                return completed_result(self.observer, run_id, last_answer, step, total,
                                        messages, reason)

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
                return completed_result(
                    self.observer,
                    run_id,
                    last_answer or "Budget exhausted before requested tools could run.",
                    step,
                    total,
                    messages,
                    AgentStopReason.BUDGET_EXCEEDED,
                )

            for tool_call in response.message.tool_calls:
                tool_started = time.perf_counter()
                try:
                    result = await call_tool(
                        self.registry,
                        tool_call.name,
                        tool_call.arguments,
                        self.observer,
                        run_id,
                    )
                except Exception as exc:
                    self._emit(
                        "tool.failed",
                        run_id,
                        step=step,
                        tool=tool_call.name,
                        error_type=type(exc).__name__,
                    )
                    raise
                self._emit(
                    "tool.completed",
                    run_id,
                    step=step,
                    tool=tool_call.name,
                    duration_s=time.perf_counter() - tool_started,
                    result_length=len(result),
                )
                messages.append(
                    Message(
                        role="tool",
                        content=result[:8000],
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        return completed_result(
            self.observer,
            run_id,
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
