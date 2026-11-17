# Choosing an LLM and configuring your app

The harness does not automatically read `config.yaml`. Your application can
choose to read a config file, environment variables, command-line options, or
another settings source, then pass the values to `create_llm()` and `Agent`.
For a first test, passing the provider and model directly is the simplest choice.

## Provider choices

| Provider | Extra | Typical setup |
| --- | --- | --- |
| `openai` | `openai` | Set `OPENAI_API_KEY`. |
| `anthropic` | `anthropic` | Set `ANTHROPIC_API_KEY`. |
| `openrouter` | `openai` | Set `OPENROUTER_API_KEY`. |
| `openai_compat` | `openai` | Set `base_url`; pass `api_key` if the endpoint requires one. Local endpoints can use a dummy key. |
| `ollama`, `vllm`, `lmstudio` | `openai` | Set `base_url` to the endpoint; for example, Ollama commonly uses the url: `http://localhost:11434/v1`. |

Use the model identifier expected by the provider. `create_llm()` accepts
provider-specific options such as `base_url`, `api_key`, `timeout`, and
`cost_manager`. The request timeout defaults to 60 seconds.

For example, to use Ollama's OpenAI-compatible endpoint:

```python
import llm_harness_lite

llm = llm_harness_lite.create_llm(
    "openai_compat",
    "qwen3:8b",
    base_url="http://localhost:11434/v1",
    timeout=300,
)
```

Install Ollama and pull the model separately; see
[Local setup](../LOCAL_SETUP.md) for download, startup, and model commands.
For hosted providers, keep the API keys outside the YAML file and source them from
environment variables or a secret manager.

## Optional YAML configuration

If you want users to change settings without editing Python, your application
can load a small YAML file. Install the example dependencies with
`pip install "llm-harness-lite[examples]"` or install PyYAML separately.

`config.yaml`:

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  timeout: 60
agent:
  max_steps: 5
  budget_usd: 0.02
```

The application code reads the file and passes the relevant values to the
harness:

```python
from pathlib import Path

import yaml
import llm_harness_lite

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
llm_cfg = cfg["llm"]
agent_cfg = cfg["agent"]

llm = llm_harness_lite.create_llm(
    llm_cfg["provider"],
    llm_cfg["model"],
    timeout=llm_cfg.get("timeout", 60),
)
agent = llm_harness_lite.Agent(
    llm,
    tools,
    max_steps=agent_cfg.get("max_steps", 8),
    budget_usd=agent_cfg.get("budget_usd", 0.05),
)
```

This is your application's configuration convention, not a harness-specific
file format. Add fields only when your application uses them. For example,
RAG settings belong in config only if your application creates an embedder,
vector store, and retriever; see [RAG and MCP integrations](integrations.md).

## Secrets and local environment

Do not commit API keys, database passwords, or other credentials. Set provider
keys in the process environment. A `.env` file can be convenient for local
development if your application loads it with `python-dotenv`; it should stay
out of version control.

The example [`config.yaml`](../config.yaml) includes settings for several
optional features. It is a sample for the repository's examples, not a
required configuration for every harness user.
