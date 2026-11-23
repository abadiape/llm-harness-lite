import llm_harness_lite
from llm_harness_lite.agent.types import AgentResult
from llm_harness_lite.artifacts import Artifact, ArtifactRegistry, ArtifactType, SkillResult
from llm_harness_lite.llm.router import LLMRouter


def test_current_namespace_exports_core_objects():
    assert llm_harness_lite.LLMRouter is LLMRouter
    assert llm_harness_lite.AgentResult is AgentResult


def test_artifact_types_are_exported_from_public_namespace():
    assert llm_harness_lite.Artifact is Artifact
    assert llm_harness_lite.ArtifactRegistry is ArtifactRegistry
    assert llm_harness_lite.ArtifactType is ArtifactType
    assert llm_harness_lite.SkillResult is SkillResult
