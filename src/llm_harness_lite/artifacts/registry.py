"""In-memory registry for artifact metadata."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Artifact


class ArtifactRegistry:
    """Track artifact metadata and export it as a JSON manifest."""

    def __init__(self) -> None:
        self._artifacts: list[Artifact] = []

    def register(self, artifact: Artifact) -> Artifact:
        self._artifacts.append(artifact)
        return artifact

    def list_artifacts(self) -> list[Artifact]:
        return list(self._artifacts)

    def count(self) -> int:
        return len(self._artifacts)

    def export_manifest(self, path: str | Path = "outputs/manifest.json") -> Path:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [artifact.to_dict() for artifact in self._artifacts],
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
            manifest_file.write("\n")
        return manifest_path
