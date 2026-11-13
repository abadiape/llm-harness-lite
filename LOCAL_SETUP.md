# Local setup (recommended: venv + compose for infra only)

You do NOT need a Dockerfile for daily dev. `docker-compose.yml` pulls the
prebuilt pgvector image by default. Your Python code runs on the host in a venv
for fast iteration.

## 1) Local RAG with pgvector
```bash
docker compose up -d db
docker ps   # 5432 = pgvector
```

## 2) Python in venv (host)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,examples,openai,ollama,pgvector]"
cp .env.example .env
```

For a minimal local environment, install only what is needed; for example,
`pip install -e ".[openai,ollama]"` is enough for Ollama chat and embeddings.
Set the relevant API key in `.env` when using a hosted provider, such as
`OPENAI_API_KEY` for OpenAI. Ollama does not require an API key.

Qdrant is the production vector-store option and is also available for local
testing. Start it only when needed with `docker compose up -d qdrant` and add
the `qdrant` extra (`pip install -e ".[qdrant]"`). Both stores use the same
`Retriever`; switch `store.kind` in `config.yaml` between `pgvector` (the local
default) and `qdrant`.

## Cost estimates

`CostManager.initialize()` refreshes OpenRouter's public model catalog when its
local cache expires (one hour by default). It is explicit, so imports and normal
chat calls do not make network requests; if refresh fails, cached or built-in
prices remain available. `examples/04_cost_compare.py` demonstrates setup.
You can override prices in `config.yaml`:

```yaml
costs:
  overrides:
    company/model-name:
      input_per_million: 0.25
      output_per_million: 1.0
```

Pass the configured manager to `create_llm(..., cost_manager=cost_manager)` to
use its refreshed prices and overrides:

```python
cost_manager = CostManager.from_config(config.get("costs"))
cost_manager.initialize()
llm = create_llm(provider, model, cost_manager=cost_manager)
```

Loopback OpenAI-compatible endpoints are treated as free unless an override
matches their model. An unknown remote model warns and estimates as $0 until
you add an override. Prices cover input and output tokens; cached tokens,
images, and provider-specific billing adjustments are not included.

## 3) Local models (cheapest path)
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## 4) Run
```bash
python examples/01_basic_chat.py
python examples/03_rag_ingest_search.py
python examples/05_qdrant_mcp_cache.py
```

## When to use the Dockerfile?
Only for prod deploy or sharing an exact env:
```bash
docker build -t universal-harness .
docker run --network host --env-file .env universal-harness
```
