"""Bang-for-buck: same prompt across backends, print tokens/$/latency."""
import yaml
from dotenv import load_dotenv
from openai import APIError

import llm_harness_lite
from llm_harness_lite.utils.costs import CostManager

load_dotenv()
with open("config.yaml") as config_file:
    cfg = yaml.safe_load(config_file)
cost_manager = CostManager.from_config(cfg.get("costs"))
catalog_refreshed = cost_manager.initialize()
print(f"Pricing catalog: {'refreshed' if catalog_refreshed else 'cached/fallback'}")

MODELS = [
    ("openai_compat", "qwen3:8b", {
        "base_url": "http://localhost:11434/v1",
        "timeout": 300.0,
    }),
    ("openai", "gpt-4o-mini", {}),
    # ("anthropic", "claude-3-5-haiku-latest", {}),
]
prompt = "Explain pgvector vs Qdrant in 3 bullets for a Python dev."
for provider, model, kw in MODELS:
    try:
        llm = llm_harness_lite.create_llm(provider, model, cost_manager=cost_manager, **kw)
        r = llm.chat([llm_harness_lite.Message(role="user", content=prompt)])
        print(f"\n== {model} ==\n{r.message.content[:400]}")
        print(
            f"tok {r.usage.input_tokens}/{r.usage.output_tokens} | "
            f"${r.usage.cost_usd:.5f} | {r.usage.latency_s:.1f}s"
        )
    except (APIError, ConnectionError, TimeoutError, ValueError) as e:
        print(f"\n== {model} FAILED: {e}")
