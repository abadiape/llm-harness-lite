"""Normalized chat() — swap provider via config.yaml / env."""
import yaml
from dotenv import load_dotenv

load_dotenv()
import llm_harness_lite

with open("config.yaml") as config_file:
    cfg = yaml.safe_load(config_file)
llm = llm_harness_lite.create_llm(cfg["llm"]["provider"], cfg["llm"]["model"],
                 base_url=cfg["llm"].get("base_url"))
resp = llm.chat([llm_harness_lite.Message(role="user", content="Say hello in one line + name your model.")])
print(resp.message.content)
print(f"[tokens {resp.usage.input_tokens}/{resp.usage.output_tokens} | ${resp.usage.cost_usd:.5f}]")
