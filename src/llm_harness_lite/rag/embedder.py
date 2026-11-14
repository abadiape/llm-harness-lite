"""Embedder protocol + OpenAI + Ollama implementations."""
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int
    model: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()
        self.dim = 1536 if "small" in model else 3072
    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in r.data]

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434"):
        import ollama
        self.model, self.dim = model, 768
        self.client = ollama.Client(host=base_url)
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.client.embeddings(model=self.model, prompt=t)["embedding"]
                for t in texts]
