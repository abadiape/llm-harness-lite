"""LLM Harness Lite — lightweight universal agent + RAG + MCP for any model."""
from llm_harness_lite.agent.loop import Agent, AgentResult, AgentStopReason
from llm_harness_lite.llm.factory import create_llm, register_provider
from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, ToolSpec
from llm_harness_lite.tools.registry import ToolExecutionError, ToolRegistry, tool

# Short alias for code: `import llm_harness_lite as harness`
__all__ = [
    "Agent",
    "AgentResult",
    "AgentStopReason",
    "ChatResponse",
    "Message",
    "ToolCallRequest",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolSpec",
    "create_llm",
    "register_provider",
    "tool",
]
__version__ = "0.1.0a1"
