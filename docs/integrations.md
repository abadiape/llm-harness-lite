# RAG and MCP integrations

Start with the model and tool registry from the
[first-agent guide](guide.md). Add these integrations only when your
application needs them. RAG and MCP are not required to create an agent.

## Add retrieval as a tool

RAG has three application-level pieces: an embedder, a vector store, and a
`Retriever`. The agent does not need to know about the database; expose the
retrieval operation as a regular tool.

This example uses Ollama embeddings and local PostgreSQL with pgvector:

```bash
pip install "llm-harness-lite[ollama-embed,pgvector]"
docker compose up -d db
ollama pull nomic-embed-text
```

Create and populate the retriever once when preparing your application:

```python
import os
from pathlib import Path

from llm_harness_lite.rag.embedder import OllamaEmbedder
from llm_harness_lite.rag.retriever import Retriever
from llm_harness_lite.rag.store_base import create_store

embedder = OllamaEmbedder("nomic-embed-text")
store = create_store(
    "pgvector",
    dsn=os.getenv(
        "DATABASE_URL",
        "postgresql://harness:harness@localhost:5432/harness",
    ),
    table="rag_chunks",
    dim=768,
)
store.init()
retriever = Retriever(embedder, store, chunk_size=600, chunk_overlap=80)
retriever.ingest(
    Path("knowledge/handbook.md").read_text(encoding="utf-8"),
    source="handbook.md",
)
```

Register a retrieval function in the same tool registry from the first-agent
guide. The standalone snippet below creates that registry explicitly:

```python
import llm_harness_lite

tools = llm_harness_lite.ToolRegistry()

@llm_harness_lite.tool(tools)
def search_handbook(query: str) -> str:
    """Find relevant passages in the support handbook."""
    return retriever.context_block(query, top_k=4)
```

The vector-store and embedding dimensions must match. `create_store()` also
supports Qdrant; see [`examples/03_rag_ingest_search.py`](../examples/03_rag_ingest_search.py)
and [`examples/05_qdrant_mcp_cache.py`](../examples/05_qdrant_mcp_cache.py) for
the repository's complete setup patterns. Keep retrieved content within the
user's authorization scope, and treat document text as untrusted input.

## Add remote tools with MCP

Install the MCP extra with `pip install "llm-harness-lite[mcp]"`. The async
`MCPClient` can discover a trusted server's tools and register them into the
same registry as local Python tools:

```python
import asyncio

import llm_harness_lite
from llm_harness_lite.mcp.client import MCPClient

async def main():
    llm = llm_harness_lite.create_llm("openai", "gpt-4o-mini")
    tools = llm_harness_lite.ToolRegistry()
    # Register any local or RAG tools in this registry before connecting MCP.

    try:
        async with MCPClient(
            command="your-mcp-server",
            args=["--server-option"],
        ) as mcp:
            await mcp.register_into(tools)
            agent = llm_harness_lite.Agent(
                llm,
                tools,
                system="Use the available tools to complete the user's request.",
                max_steps=5,
                budget_usd=0.02,
            )
            result = await agent.arun("Ask the connected service for the requested information.")
            print(result.answer)
    finally:
        await llm.aclose()

asyncio.run(main())
```

Replace the command and arguments with those required by an MCP server you
trust. For network transports, configure `MCPClient` with the server URL and
supported transport. MCP tools can have real effects: review the server,
restrict its permissions, and register only tools the agent should use.

### Optional no-API-key web search

For a concrete stdio example, the community-maintained
[DuckDuckGo Search MCP Server](https://github.com/nickclyde/duckduckgo-mcp-server)
provides web search without an API token. It requires `uvx` (provided by `uv`)
and outbound internet access. Search availability can be affected by
rate-limits or upstream blocking; the upstream recommends the `browser` extra
for its search fallback.

The server can be configured in `config.yaml` like this:

```yaml
mcp:
  servers:
    - command: uvx
      args:
        - --with
        - "duckduckgo-mcp-server[browser]"
        - duckduckgo-mcp-server
```

The example in `examples/05_qdrant_mcp_cache.py` uses the first configured
server. Keep `servers: []` in the shared sample config unless this integration
is wanted by default; personal GitHub tokens and other credentials belong in
local environment variables, not the checked-in YAML.

## Other built-in pieces

- **Cost-aware estimates:** Create a `CostManager`, initialize its public
  pricing catalog when appropriate, and pass it to `create_llm()`. See
  [`examples/04_cost_compare.py`](../examples/04_cost_compare.py).
- **Caching:** `CachedLLM` can wrap a backend with exact or optional semantic
  caching. See [`examples/05_qdrant_mcp_cache.py`](../examples/05_qdrant_mcp_cache.py).
- **Routing:** `LLMRouter` selects among configured backends through a
  caller-supplied policy; it does not choose a route automatically. See the
  routing example in the [README](../README.md#explicit-model-routing).
- **Observability and custom providers:** `Agent` accepts an observer callback,
  and custom backends can be registered with `register_provider()`. The
  [README](../README.md) has brief overviews of both features.
