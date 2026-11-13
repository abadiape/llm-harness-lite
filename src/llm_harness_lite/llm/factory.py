"""Add new providers here. Keep `create_llm` as the single entry point."""
from __future__ import annotations
import os

def create_llm(provider: str, model: str, **kw):
    """provider: openai_compat | openai | openrouter | ollama | anthropic"""
    cost_manager = kw.get("cost_manager")
    if provider == "anthropic":
        from llm_harness_lite.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model, cost_manager=cost_manager)
    if provider == "openai":
        from llm_harness_lite.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model, api_key=os.getenv("OPENAI_API_KEY"),
                                   cost_manager=cost_manager)
    if provider == "openrouter":
        from llm_harness_lite.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model, base_url="https://openrouter.ai/api/v1",
                                   api_key=os.getenv("OPENROUTER_API_KEY"),
                                   cost_manager=cost_manager)
    # openai_compat covers ollama / vllm / lmstudio via base_url
    from llm_harness_lite.llm.openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(model, base_url=kw.get("base_url"),
                               api_key=kw.get("api_key", "dummy"),
                               cost_manager=cost_manager,
                               local=kw.get("local"))
