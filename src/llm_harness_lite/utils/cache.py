"""Two-level LLM cache to cut token spend: exact-match + semantic (embedding) match.

- Exact: SHA of (model, messages, tools) → response. Zero-cost hits.
- Semantic: cosine(prompt embedding, cached embedding) >= threshold → reuse.
- SQLite persistent (default .harness_cache.db), shared across runs.
- CachedLLM wraps ANY LLMBackend (same .chat signature) + tracks saved $.

Contributors: add Redis backend implementing ChatCache.store/lookup.
"""
from __future__ import annotations
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, ToolSpec, Usage

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def _prompt_key(model: str, messages: list[Message],
                tools: list[ToolSpec] | None) -> str:
    payload = {"m": model,
               "msgs": [(x.role, x.content) for x in messages],
               "tools": sorted(t.name for t in (tools or []))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

class ChatCache:
    def __init__(self, path: str = ".harness_cache.db"):
        self.path = path
        self.hits_exact = self.hits_sem = self.misses = 0
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE IF NOT EXISTS llm_cache(
          key TEXT PRIMARY KEY, model TEXT, prompt TEXT, embedding TEXT,
          content TEXT, toolcalls TEXT, tokens_in INT, tokens_out INT, ts REAL)""")
        con.commit()
        con.close()

    def _con(self):
        return sqlite3.connect(self.path)

    def get_exact(self, key: str) -> ChatResponse | None:
        con = self._con()
        row = con.execute("SELECT content, toolcalls, tokens_in, tokens_out FROM llm_cache WHERE key=?",
                          (key,)).fetchone()
        con.close()
        if not row:
            return None
        content, tc_json, ti, to = row
        tcs = [ToolCallRequest(**d) for d in json.loads(tc_json or "[]")]
        self.hits_exact += 1
        return ChatResponse(Message("assistant", content, tcs), "tool_calls" if tcs else "stop",
                            Usage(ti, to, 0.0, "cached", 0.0))

    def get_semantic(self, prompt: str, vec: list[float],
                     threshold: float) -> ChatResponse | None:
        con = self._con()
        rows = con.execute("SELECT content, toolcalls, tokens_in, tokens_out, embedding FROM llm_cache").fetchall()
        con.close()
        best, best_s = None, threshold
        for content, tc_json, ti, to, ejson in rows:
            if not ejson:
                continue
            s = _cosine(vec, json.loads(ejson))
            if s >= best_s:
                best_s, best = s, (content, tc_json, ti, to)
        if best is None:
            return None
        content, tc_json, ti, to = best
        tcs = [ToolCallRequest(**d) for d in json.loads(tc_json or "[]")]
        self.hits_sem += 1
        return ChatResponse(Message("assistant", content, tcs), "tool_calls" if tcs else "stop",
                            Usage(ti, to, 0.0, "cached-sem", 0.0))

    def put(self, key: str, model: str, prompt: str, resp: ChatResponse,
            vec: list[float] | None = None):
        con = self._con()
        con.execute("INSERT OR REPLACE INTO llm_cache VALUES(?,?,?,?,?,?,?,?,?)",
            (key, model, prompt, json.dumps(vec) if vec else None,
             resp.message.content,
             json.dumps([vars(t) for t in resp.message.tool_calls]),
             resp.usage.input_tokens, resp.usage.output_tokens, time.time()))
        con.commit()
        con.close()

class CachedLLM:
    """Drop-in wrapper: CachedLLM(backend, cache).chat(...) with same signature."""
    def __init__(self, backend, cache: ChatCache | None = None,
                 embedder=None, semantic_threshold: float = 0.93,
                 cache_tools: bool = False):
        self.backend = backend
        self.cache = cache or ChatCache()
        self.embedder = embedder
        self.threshold = semantic_threshold
        self.cache_tools = cache_tools
        self.saved_usd = 0.0

    @property
    def model(self) -> str:
        return getattr(self.backend, "model", "cached") + "+cache"

    def _last_prompt(self, messages: list[Message]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return messages[-1].content if messages else ""

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             temperature: float = 0.2) -> ChatResponse:
        # Never serve tool-calls from semantic cache (args matter); exact only if enabled.
        key = _prompt_key(self.backend.model, messages, tools)
        if not tools or self.cache_tools:
            hit = self.cache.get_exact(key)
            if hit:
                return hit
        if self.embedder is not None and not tools:
            prompt = self._last_prompt(messages)
            try:
                vec = self.embedder.embed([prompt])[0]
                hit = self.cache.get_semantic(prompt, vec, self.threshold)
                if hit:
                    self.saved_usd += 0.001  # rough; real saving computed by caller diff
                    return hit
            except Exception:
                vec = None
        else:
            vec = None
        resp = self.backend.chat(messages, tools, temperature)
        if not resp.message.tool_calls or self.cache_tools:
            try:
                self.cache.put(key, self.backend.model, self._last_prompt(messages), resp, vec)
            except Exception:
                pass
        else:
            self.cache.misses += 1
        return resp

    def stats(self) -> dict:
        return {"hits_exact": self.cache.hits_exact, "hits_sem": self.cache.hits_sem,
                "saved_usd_est": round(self.saved_usd, 4)}
