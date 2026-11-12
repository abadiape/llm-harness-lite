"""RAG starter: ingest a file → configured vector store → search → inject into prompt."""
import yaml
from dotenv import load_dotenv
load_dotenv()
import llm_harness_lite as harness
from llm_harness_lite.rag.embedder import OllamaEmbedder, OpenAIEmbedder
from llm_harness_lite.rag.store_base import create_store
from llm_harness_lite.rag.retriever import Retriever

cfg = yaml.safe_load(open("config.yaml"))
ecfg = cfg["embed"]
emb = OllamaEmbedder(ecfg["model"], ecfg.get("base_url", "http://localhost:11434")) \
      if ecfg["provider"] == "ollama" else OpenAIEmbedder(ecfg["model"])
store_cfg = cfg["store"][cfg["store"]["kind"]]
store = create_store(cfg["store"]["kind"], **store_cfg)
store.init()
ret = Retriever(emb, store, cfg["rag"]["chunk_size"], cfg["rag"]["chunk_overlap"])

text = open("README.md").read()
print(f"ingested {ret.ingest(text, source='README.md')} chunks")
q = "What does this harness do?"
ctx = ret.context_block(q, top_k=cfg["rag"]["top_k_final"])
llm = harness.create_llm(cfg["llm"]["provider"], cfg["llm"]["model"],
                 base_url=cfg["llm"].get("base_url"))
resp = llm.chat([harness.Message(role="system", content="Answer ONLY from CONTEXT. Cite [1][2]."),
                 harness.Message(role="user", content=f"CONTEXT:\n{ctx}\n\nQ: {q}")])
print(resp.message.content)
