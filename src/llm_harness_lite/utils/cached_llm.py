"""Synchronous LLM wrapper with exact and optional semantic caching."""
from __future__ import annotations

import json
import math

from llm_harness_lite.llm.base import LLMBackend
from llm_harness_lite.llm.types import ChatResponse, Message, ToolSpec
from llm_harness_lite.utils.cache_core import (
    ChatCache,
    messages_payload,
    prompt_key,
    validate_vector,
)


class CachedLLM:
    def __init__(
        self,
        backend: LLMBackend,
        cache: ChatCache | None = None,
        embedder=None,
        semantic_threshold: float = 0.93,
        cache_tools: bool = False,
    ):
        if (isinstance(semantic_threshold, bool)
                or not isinstance(semantic_threshold, (int, float))
                or not math.isfinite(semantic_threshold)
                or not 0 <= semantic_threshold <= 1):
            raise ValueError("semantic_threshold must be between 0 and 1")
        self.backend = backend
        self.cache = cache if cache is not None else ChatCache()
        self.embedder = embedder
        self.threshold = float(semantic_threshold)
        self.cache_tools = cache_tools
        self.saved_usd = 0.0

    @property
    def model(self) -> str:
        return getattr(self.backend, "model", "cached") + "+cache"

    def _model_for_request(
        self, messages: list[Message], tools: list[ToolSpec] | None, temperature: float
    ) -> str:
        resolver = getattr(self.backend, "model_for_request", None)
        if callable(resolver):
            return resolver(messages, tools=tools, temperature=temperature)
        return self.backend.model

    @staticmethod
    def _semantic_prompt(messages: list[Message]) -> str:
        return json.dumps(messages_payload(messages), sort_keys=True, separators=(",", ":"))

    def _semantic_scope(self, model: str, temperature: float) -> str:
        embedder_model = getattr(self.embedder, "model", None)
        if not isinstance(embedder_model, str) or not embedder_model:
            embedder_model = f"{type(self.embedder).__module__}.{type(self.embedder).__qualname__}"
        return json.dumps(
            {"model": model, "temperature": temperature, "embedder": embedder_model},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
    ) -> ChatResponse:
        model = self._model_for_request(messages, tools, temperature)
        key = prompt_key(model, messages, tools, temperature)
        if not tools or self.cache_tools:
            hit = self.cache.get_exact(key)
            if hit:
                return hit
        vector, prompt, scope = None, "", ""
        if self.embedder is not None and not tools:
            prompt = self._semantic_prompt(messages)
            scope = self._semantic_scope(model, temperature)
            vectors = self.embedder.embed([prompt])
            if len(vectors) != 1:
                raise ValueError(f"embedder returned {len(vectors)} vectors for one prompt")
            vector = validate_vector(vectors[0])
            hit = self.cache.get_semantic(prompt, vector, self.threshold, scope)
            if hit:
                self.saved_usd += 0.001
                return hit
        response = self.backend.chat(messages, tools, temperature)
        if not response.message.tool_calls or self.cache_tools:
            self.cache.put(
                key, model, prompt or json.dumps(messages_payload(messages)),
                response, vector, scope,
            )
        else:
            self.cache.misses += 1
        return response

    def stats(self) -> dict:
        return {
            "hits_exact": self.cache.hits_exact,
            "hits_sem": self.cache.hits_sem,
            "saved_usd_est": round(self.saved_usd, 4),
        }
