from harness.agent.types import AgentResult as CompatAgentResult
from harness.llm.router import LLMRouter as CompatRouter

import harness
import llm_harness_lite
from llm_harness_lite.agent.types import AgentResult


def test_legacy_namespace_reuses_public_objects():
    assert harness is llm_harness_lite
    assert CompatRouter is llm_harness_lite.LLMRouter
    assert CompatAgentResult is AgentResult
