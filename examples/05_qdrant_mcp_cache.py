"""All three add-ons together: Qdrant store + persistent MCP + semantic cache.

Run: docker compose up -d qdrant ; python examples/05_qdrant_mcp_cache.py
"""
import yaml
from dotenv import load_dotenv
load_dotenv()

import llm_harness_lite as harness
from llm_harness_lite.rag.embedder import OllamaEmbedder
from llm_harness_lite.rag.store_base import create_store
from llm_harness_lite.rag.retriever import Retriever
from llm_harness_lite.utils.cache import ChatCache, CachedLLM
from llm_harness_lite.mcp.client import PersistentMCP

cfg = yaml.safe_load(open("config.yaml"))

# 1) Qdrant store. Both supported stores work with the same Retriever.
s = cfg["store"]["qdrant"]
store = create_store("qdrant", url=s["url"], collection=s["collection"], dim=s["dim"])
store.init()
emb = OllamaEmbedder(cfg["embed"]["model"], cfg["embed"]["base_url"])
ret = Retriever(emb, store)
ret.ingest("Qdrant is a Rust vector DB with hybrid dense+sparse search.", source="docs")
print("Qdrant search:", ret.context_block("What is Qdrant?", top_k=2)[:200])

# 2) Cached LLM (exact + semantic)
base = harness.create_llm(cfg["llm"]["provider"], cfg["llm"]["model"], base_url=cfg["llm"].get("base_url"))
llm = CachedLLM(base, ChatCache(cfg["cache"]["path"]), embedder=emb,
                semantic_threshold=cfg["cache"]["semantic_threshold"])
for i in range(2):  # second call should hit cache
    r = llm.chat([harness.Message(role="user", content="What is Qdrant in one line?")])
    print(f"call{i}: {r.message.content[:120]} | model={r.usage.model}")
print("cache stats:", llm.stats())

# 3) Persistent MCP (needs an MCP server; skipped if none configured)
servers = cfg.get("mcp", {}).get("servers", [])
if servers:
    reg = harness.ToolRegistry()
    m = PersistentMCP(command=servers[0]["command"], args=servers[0].get("args", [])).start()
    print("MCP tools:", m.register_into(reg))
    m.stop()
else:
    print("MCP: no servers in config.yaml -> skipped (add one to test).")
