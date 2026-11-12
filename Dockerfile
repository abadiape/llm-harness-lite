# App image for prod / collaborators who want full-docker. For daily dev, use venv + compose db only.
FROM python:3.15-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]" && pip install --no-cache-dir qdrant-client ollama
COPY config.yaml .env.example ./
COPY examples/ examples/
CMD ["python", "examples/01_basic_chat.py"]
