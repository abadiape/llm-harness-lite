# LLM Harness Lite — lightweight universal agent + RAG + MCP for any LLM

Local-first, vendor-neutral, cost-aware. Not the next LangChain: ~1k LOC core you can read in an afternoon.

```
config.yaml → llm/factory → agent/loop → tools/registry (+ RAG as tool, MCP tools)
```

## Install
```bash
pip install llm-harness-lite                 # dependency-free core
pip install "llm-harness-lite[openai]"       # OpenAI-compatible providers
pip install "llm-harness-lite[anthropic]"    # Anthropic
pip install "llm-harness-lite[pgvector]"     # PostgreSQL + pgvector RAG
pip install "llm-harness-lite[qdrant]"       # Qdrant RAG
pip install "llm-harness-lite[mcp]"          # MCP client
pip install "llm-harness-lite[full]"         # all integrations + example config deps
# or local dev:
# python -m venv .venv && source .venv/bin/activate
# pip install -e ".[dev,examples,openai,ollama,pgvector]"
# docker compose up -d db
```

The core package uses only the Python standard library. Provider SDKs, vector
database clients, and the MCP SDK are optional; install only the extras your
application uses. OpenAI-compatible providers (including Ollama and vLLM) use
the `openai` extra.

## RAG portability

The initial supported vector stores are intentionally limited to **pgvector**
and **Qdrant**. The default local setup uses pgvector; Qdrant is the production
deployment option. Both implement the same `VectorStore` contract and plug into
the same `Retriever`. Embedding generation is separate from both storage and
chat-model choice, so users own their vectors and can change LLM providers
without binding retrieval to a model vendor.

Run `docker compose up -d qdrant` only when testing the Qdrant path. The Qdrant
client is available through the `qdrant` extra.

Custom providers can implement the `LLMBackend` protocol and register a factory
without changing the library:

```python
harness.register_provider("my-runtime", MyBackend)
llm = harness.create_llm("my-runtime", "model-name", endpoint="http://localhost")
```

## Use (short alias supported)
```python
import llm_harness_lite as harness
llm = harness.create_llm("openai_compat", "qwen3:8b", base_url="http://localhost:11434/v1")
```

The agent API is async-first: use `await agent.arun(goal)` in async applications.
`agent.run(goal)` is a synchronous convenience for scripts and must not be called
from an already-running event loop. Agent results expose a structured
`stop_reason` and usage totals. Tools can raise `harness.ToolExecutionError` for
recoverable failures to return to the model; unexpected tool exceptions propagate
to the caller.

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
