"""Metadata models and registry for generated artifacts."""
from .models import Artifact, ArtifactType, SkillResult
from .registry import ArtifactRegistry

__all__ = ["Artifact", "ArtifactRegistry", "ArtifactType", "SkillResult"]
