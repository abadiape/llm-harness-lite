"""Agent loop with local python tools + RAG-as-tool pattern."""
import yaml
from dotenv import load_dotenv

import llm_harness_lite

load_dotenv()

cfg = yaml.safe_load(open("config.yaml"))  # noqa: SIM115
llm = llm_harness_lite.create_llm(cfg["llm"]["provider"], cfg["llm"]["model"],
                 base_url=cfg["llm"].get("base_url"))
reg = llm_harness_lite.ToolRegistry()

@llm_harness_lite.tool(reg)
def calc(expr: str) -> str:
    """Evaluate a math expression. Args: expr like '12*7'."""
    allowed = set("0123456789+-*/(). ")
    assert set(expr) <= allowed, "unsafe chars"
    return str(eval(expr))

@llm_harness_lite.tool(reg)
def search_docs(query: str) -> str:
    """Search ingested docs. Replace body with Retriever.context_block()."""
    return f"[stub] would retrieve for: {query}"

agent = llm_harness_lite.Agent(llm, reg, max_steps=cfg["agent"]["max_steps"],
              budget_usd=cfg["agent"]["budget_usd"])
res = agent.run("What is 18*23? Then search_docs for 'refunds'. Answer concisely.")
print(res.answer)
print(f"[steps {res.steps} | ${res.usage.cost_usd:.5f} | {res.usage.input_tokens} input tokens | {res.usage.output_tokens} output tokens | {res.usage.latency_s:.2f}s latency]")
