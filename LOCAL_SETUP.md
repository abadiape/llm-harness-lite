# Local setup (recommended: venv + compose for infra only)

You do NOT need a Dockerfile for daily dev. `docker-compose.yml` pulls the
prebuilt pgvector image by default. Your Python code runs on the host in a venv
for fast iteration. This guide is for local development, not production
deployment.

## 1) Local RAG with pgvector
```bash
docker compose up -d db
docker ps   # 5432 = pgvector
```

The Compose database ports bind to `127.0.0.1`. These services are local-only:
PostgreSQL uses sample credentials and Qdrant has no authentication configured.
Do not expose them to a network or use this Compose setup in production.

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

The Qdrant adapter can connect to a secured production Qdrant deployment. For
local testing, start the unauthenticated Compose service only when needed with
`docker compose up -d qdrant` and add the `qdrant` extra
(`pip install -e ".[qdrant]"`). Both stores use the same `Retriever`; switch
`store.kind` in `config.yaml` between `pgvector` (the local default) and
`qdrant`.

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

## 3) Local models with Ollama

Install Ollama for your operating system from the
[official download page](https://ollama.com/download), then launch the app. If
the server is not already running, start it in a separate terminal and leave
that terminal open:

```bash
ollama serve
```

In another terminal, download the models used by the sample configuration:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama list
```

The sample config uses Ollama's OpenAI-compatible endpoint at
`http://localhost:11434/v1` for chat and its native endpoint at
`http://localhost:11434` for embeddings. Local Ollama needs no API key; the
sample's `dummy` key is only a placeholder for the OpenAI-compatible client.
See [Choosing an LLM and configuring your app](docs/configuration.md) for the
provider settings.

## 4) Run
```bash
python examples/01_basic_chat.py
python examples/03_rag_ingest_search.py
python examples/05_qdrant_mcp_cache.py
```

The fifth example requires the Qdrant extra and service:
`pip install -e ".[qdrant]"` and `docker compose up -d qdrant`. Install the MCP
extra only if you add an MCP server to `config.yaml`.

## Container example

The Dockerfile builds a sample runner, not a production-hardened service. It
does not run Ollama or a database, and it has no API server, health checks, or
deployment configuration. The command below uses Linux host networking so the
sample can reach Ollama at the configured localhost URL:

```bash
docker build -t llm-harness-lite .
docker run --rm --network host --env-file .env llm-harness-lite
```

For production, deploy your application with secured provider and data-service
configuration, secrets management, monitoring, backups, and capacity settings
appropriate to your workload. The project does not prescribe a production
platform or fixed CPU/memory requirements.
