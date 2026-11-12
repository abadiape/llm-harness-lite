# LLM Harness Lite — lightweight universal agent + RAG + MCP for any LLM

Local-first, vendor-neutral, cost-aware. Not the next LangChain: ~1k LOC core you can read in an afternoon.

```
config.yaml → llm/factory → agent/loop → tools/registry (+ RAG as tool, MCP tools)
```

## Install
```bash
pip install llm-harness-lite
# or local dev:
# python -m venv .venv && source .venv/bin/activate
# pip install -e ".[dev]"
# docker compose up -d db qdrant
```

## Use (short alias supported)
```python
import llm_harness_lite as harness
llm = harness.create_llm("openai_compat", "qwen3:8b", base_url="http://localhost:11434/v1")
```

See `LOCAL_SETUP.md` + `examples/`:
- `01_basic_chat.py` — normalized `chat()` across providers
- `02_agent_with_tools.py` — agent loop + budget guard
- `03_rag_ingest_search.py` — pgvector RAG
- `04_cost_compare.py` — bang-for-buck across models
- `05_qdrant_mcp_cache.py` — Qdrant + persistent MCP + semantic cache

## Layout (for contributors)
- `src/llm_harness_lite/llm/` — `types.py`, `base.py` (Protocol), `openai_compat.py`, `anthropic_backend.py`, `factory.py`
- `src/llm_harness_lite/agent/loop.py` — framework-agnostic ReAct loop
- `src/llm_harness_lite/tools/registry.py` — `@tool` + JSON Schema from signature
- `src/llm_harness_lite/rag/` — `chunker.py`, `embedder.py`, `store_base.py`, `store_pgvector.py`, `store_qdrant.py`, `retriever.py`
- `src/llm_harness_lite/mcp/client.py` — `MCPClient` + `PersistentMCP` background-loop bridge
- `src/llm_harness_lite/utils/` — `costs.py`, `cache.py` (exact + semantic)
- `src/harness/` — back-compat shim (`import harness` still works)
