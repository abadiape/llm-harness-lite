"""Back-compat: `import harness` is an alias of `llm_harness_lite`.

New code should use `import llm_harness_lite as harness`.
Old deep imports (`from harness.llm.types import X`) keep working and
resolve to the SAME objects (no duplicate-class bugs).
"""
import importlib as _il
import sys as _sys

import llm_harness_lite as _m

_sys.modules[__name__] = _m

# Alias submodules so harness.X is llm_harness_lite.X (same object)
_SUBS = [
    "llm",
    "llm.types",
    "llm.base",
    "llm.factory",
    "llm.router",
    "llm.openai_compat",
    "llm.anthropic_backend",
    "agent",
    "agent.loop",
    "agent.types",
    "agent.prompts",
    "agent.telemetry",
    "agent.execution",
    "agent.validation",
    "tools",
    "tools.registry",
    "rag",
    "rag.chunker",
    "rag.embedder",
    "rag.store_base",
    "rag.store_pgvector",
    "rag.store_qdrant",
    "rag.retriever",
    "mcp",
    "mcp.client",
    "mcp.client_async",
    "mcp.client_sync",
    "utils",
    "utils.costs",
    "utils.cache",
    "utils.cache_core",
    "utils.cached_llm",
]
for _s in _SUBS:
    try:
        _mod = _il.import_module(f"llm_harness_lite.{_s}")
        _sys.modules[f"harness.{_s}"] = _mod
    except ImportError:
        pass
