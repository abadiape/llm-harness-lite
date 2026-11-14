"""Add new providers here. Keep `create_llm` as the single entry point."""
from __future__ import annotations

import os
from collections.abc import Callable

from llm_harness_lite.llm.base import LLMBackend

_BUILTIN_PROVIDERS = {
    "anthropic", "openai", "openai_compat", "openrouter",
    "ollama", "vllm", "lmstudio",
}
_PROVIDER_FACTORIES: dict[str, Callable[..., LLMBackend]] = {}


def register_provider(provider: str, factory: Callable[..., LLMBackend]) -> None:
    """Register a custom backend factory accepting ``model`` and provider kwargs."""
    name = provider.strip().casefold()
    if not name:
        raise ValueError("Provider name must not be empty")
    if name in _BUILTIN_PROVIDERS:
        raise ValueError(f"Cannot replace built-in LLM provider {name!r}")
    if name in _PROVIDER_FACTORIES:
        raise ValueError(f"LLM provider {name!r} is already registered")
    _PROVIDER_FACTORIES[name] = factory


def create_llm(provider: str, model: str, **kw) -> LLMBackend:
    """Create a backend by provider name; provider SDKs are optional extras."""
    provider = provider.strip().casefold()
    cost_manager = kw.get("cost_manager")
    if provider == "anthropic":
        from llm_harness_lite.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model, api_key=kw.get("api_key"),
                                cost_manager=cost_manager,
                                timeout=kw.get("timeout", 60.0))
    if provider == "openai":
        from llm_harness_lite.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model, base_url=kw.get("base_url"),
                                   api_key=kw.get("api_key", os.getenv("OPENAI_API_KEY")),
                                   cost_manager=cost_manager,
                                   timeout=kw.get("timeout", 60.0))
    if provider == "openrouter":
        from llm_harness_lite.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(
            model, base_url=kw.get("base_url", "https://openrouter.ai/api/v1"),
            api_key=kw.get("api_key", os.getenv("OPENROUTER_API_KEY")),
            cost_manager=cost_manager,
            timeout=kw.get("timeout", 60.0),
        )
    if provider in {"openai_compat", "ollama", "vllm", "lmstudio"}:
        from llm_harness_lite.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model, base_url=kw.get("base_url"),
                                   api_key=kw.get("api_key", "dummy"),
                                   cost_manager=cost_manager,
                                   local=kw.get("local"),
                                   timeout=kw.get("timeout", 60.0))
    if provider in _PROVIDER_FACTORIES:
        return _PROVIDER_FACTORIES[provider](model, **kw)
    supported = ", ".join(sorted(_BUILTIN_PROVIDERS | _PROVIDER_FACTORIES.keys()))
    raise ValueError(f"Unknown LLM provider {provider!r}. Supported providers: {supported}")
