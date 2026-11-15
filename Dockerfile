# Sample runner image for local use; this is not production-hardened.
FROM python:3.15-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[openai,examples]"
COPY config.yaml ./
COPY examples/ examples/
CMD ["python", "examples/01_basic_chat.py"]
