# Local setup (recommended: venv + compose for infra only)

You do NOT need a Dockerfile for daily dev. `docker-compose.yml` already pulls
prebuilt images (pgvector, qdrant). Your Python code runs on the host in a venv
for fast iteration.

## 1) Infra with Docker (one command)
```bash
docker compose up -d db qdrant
docker ps   # 5432 = pgvector, 6333 = qdrant
```

## 2) Python in venv (host)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
pip install -e ".[full]" 2>/dev/null || pip install qdrant-client ollama
cp .env.example .env
```
Set the relevant API key in `.env` when using a hosted provider, such as
`OPENAI_API_KEY` for OpenAI. Ollama does not require an API key.

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
