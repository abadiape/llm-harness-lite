"""Metadata models for generated artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    SVG = "svg"
    CSV = "csv"
    TEXT = "text"
    OTHER = "other"


@dataclass
class Artifact:
    """Descriptive metadata for a generated output file."""

    name: str
    type: ArtifactType | str
    path: str | Path
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        artifact_type = self.type.value if isinstance(self.type, ArtifactType) else self.type
        return {
            "name": self.name,
            "type": artifact_type,
            "path": str(self.path),
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class SkillResult:
    """Summary and artifacts produced by a skill execution."""

    summary: str
    artifacts: list[Artifact] = field(default_factory=list)
