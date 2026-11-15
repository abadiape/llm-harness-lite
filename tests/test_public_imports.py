import llm_harness_lite
from llm_harness_lite.agent.types import AgentResult
from llm_harness_lite.llm.router import LLMRouter


def test_current_namespace_exports_core_objects():
    assert llm_harness_lite.LLMRouter is LLMRouter
    assert llm_harness_lite.AgentResult is AgentResult
