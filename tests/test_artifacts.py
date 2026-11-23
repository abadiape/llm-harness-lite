import json
from datetime import datetime

from llm_harness_lite.artifacts import Artifact, ArtifactRegistry, ArtifactType, SkillResult


def test_registry_registers_and_lists_artifacts():
    artifact = Artifact(
        name="Architecture Explorer",
        type=ArtifactType.HTML,
        path="outputs/architecture.html",
        description="Interactive architecture report",
    )
    registry = ArtifactRegistry()

    assert registry.register(artifact) is artifact
    assert registry.list_artifacts() == [artifact]
    assert registry.count() == 1
    assert SkillResult("Analysis complete", [artifact]).artifacts == [artifact]


def test_artifact_accepts_custom_type_strings():
    artifact = Artifact(name="Custom output", type="parquet", path="outputs/data.parquet")

    assert artifact.to_dict()["type"] == "parquet"


def test_registry_exports_manifest(tmp_path):
    registry = ArtifactRegistry()
    registry.register(
        Artifact(
            name="Repository Map",
            type=ArtifactType.JSON,
            path=tmp_path / "repository_map.json",
            metadata={"files": 12},
        )
    )

    manifest_path = registry.export_manifest(tmp_path / "outputs" / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    datetime.fromisoformat(manifest["generated_at"])
    assert manifest["artifacts"] == [
        {
            "name": "Repository Map",
            "type": "json",
            "path": str(tmp_path / "repository_map.json"),
            "description": "",
            "metadata": {"files": 12},
        }
    ]
