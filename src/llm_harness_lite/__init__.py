"""LLM Harness Lite — lightweight universal agent + RAG + MCP for any model."""
from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, ToolSpec
from llm_harness_lite.llm.factory import create_llm
from llm_harness_lite.agent.loop import Agent, AgentResult
from llm_harness_lite.tools.registry import ToolRegistry, tool

# Short alias for code: `import llm_harness_lite as harness`
__all__ = ["ChatResponse", "Message", "ToolCallRequest", "ToolSpec",
           "create_llm", "Agent", "AgentResult", "ToolRegistry", "tool"]
__version__ = "0.1.0a1"
