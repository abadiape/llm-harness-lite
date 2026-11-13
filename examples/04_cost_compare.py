"""Bang-for-buck: same prompt across backends, print tokens/$/latency."""
import yaml
from dotenv import load_dotenv
import llm_harness_lite as harness
from llm_harness_lite.utils.costs import CostManager

load_dotenv()
cfg = yaml.safe_load(open("config.yaml"))
cost_manager = CostManager.from_config(cfg.get("costs"))
catalog_refreshed = cost_manager.initialize()
print(f"Pricing catalog: {'refreshed' if catalog_refreshed else 'cached/fallback'}")

MODELS = [
    ("openai_compat", "qwen3:8b", {"base_url": "http://localhost:11434/v1"}),
    ("openai", "gpt-4o-mini", {}),
    # ("anthropic", "claude-3-5-haiku-latest", {}),
]
prompt = "Explain pgvector vs Qdrant in 3 bullets for a Python dev."
for provider, model, kw in MODELS:
    try:
        llm = harness.create_llm(provider, model, cost_manager=cost_manager, **kw)
        r = llm.chat([harness.Message(role="user", content=prompt)])
        print(f"\n== {model} ==\n{r.message.content[:400]}")
        print(
            f"tok {r.usage.input_tokens}/{r.usage.output_tokens} | "
            f"${r.usage.cost_usd:.5f} | {r.usage.latency_s:.1f}s"
        )
    except Exception as e:
        print(f"\n== {model} FAILED: {e}")
