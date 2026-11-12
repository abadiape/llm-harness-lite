"""Agent loop with local python tools + RAG-as-tool pattern."""
import yaml
from dotenv import load_dotenv
load_dotenv()
import llm_harness_lite as harness

cfg = yaml.safe_load(open("config.yaml"))
llm = harness.create_llm(cfg["llm"]["provider"], cfg["llm"]["model"],
                 base_url=cfg["llm"].get("base_url"))
reg = harness.ToolRegistry()

@harness.tool(reg)
def calc(expr: str) -> str:
    """Evaluate a math expression. Args: expr like '12*7'."""
    allowed = set("0123456789+-*/(). ")
    assert set(expr) <= allowed, "unsafe chars"
    return str(eval(expr))

@harness.tool(reg)
def search_docs(query: str) -> str:
    """Search ingested docs. Replace body with Retriever.context_block()."""
    return f"[stub] would retrieve for: {query}"

agent = harness.Agent(llm, reg, max_steps=cfg["agent"]["max_steps"],
              budget_usd=cfg["agent"]["budget_usd"])
res = agent.run("What is 18*23? Then search_docs for 'refunds'. Answer concisely.")
print(res.answer)
print(f"[steps {res.steps} | ${res.usage.cost_usd:.5f} | {res.usage.input_tokens+res.usage.output_tokens} tok]")
