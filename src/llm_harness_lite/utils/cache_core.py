"""SQLite persistence and key construction for cached LLM responses."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import closing

from llm_harness_lite.llm.types import ChatResponse, Message, ToolCallRequest, ToolSpec, Usage


def validate_vector(vector: list[float]) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("semantic cache embeddings must be non-empty vectors")
    values = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("semantic cache embeddings must contain finite numbers")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("semantic cache embeddings must contain finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError("semantic cache embeddings must contain finite numbers")
        values.append(number)
    return values


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a, b = validate_vector(left), validate_vector(right)
    if len(a) != len(b):
        raise ValueError("semantic cache embedding dimensions do not match")
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def messages_payload(messages: list[Message]) -> list[dict]:
    return [
        {
            "role": message.role,
            "content": message.content,
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ],
        }
        for message in messages
    ]


def prompt_key(
    model: str,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    temperature: float = 0.2,
) -> str:
    payload = {
        "model": model,
        "messages": messages_payload(messages),
        "tools": [tool.to_openai() for tool in (tools or [])],
        "temperature": temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ChatCache:
    def __init__(self, path: str = ".harness_cache.db"):
        self.path = path
        self.hits_exact = self.hits_sem = self.misses = 0
        with closing(sqlite3.connect(path)) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS llm_cache(
              key TEXT PRIMARY KEY, model TEXT, prompt TEXT, embedding TEXT,
              content TEXT, toolcalls TEXT, tokens_in INT, tokens_out INT, ts REAL,
              scope TEXT NOT NULL DEFAULT '', finish TEXT NOT NULL DEFAULT 'stop')""")
            columns = {row[1] for row in con.execute("PRAGMA table_info(llm_cache)")}
            if "scope" not in columns:
                con.execute("ALTER TABLE llm_cache ADD COLUMN scope TEXT NOT NULL DEFAULT ''")
            if "finish" not in columns:
                con.execute(
                    "ALTER TABLE llm_cache ADD COLUMN finish TEXT NOT NULL DEFAULT 'stop'"
                )
            con.commit()

    def _con(self):
        return sqlite3.connect(self.path)

    def get_exact(self, key: str) -> ChatResponse | None:
        with closing(self._con()) as con:
            row = con.execute(
                "SELECT content, toolcalls, tokens_in, tokens_out, finish "
                "FROM llm_cache WHERE key=?",
                (key,),
            ).fetchone()
        if not row:
            return None
        content, tc_json, ti, to, finish = row
        calls = [ToolCallRequest(**item) for item in json.loads(tc_json or "[]")]
        self.hits_exact += 1
        return ChatResponse(
            Message("assistant", content, calls),
            finish,
            Usage(ti, to, 0.0, "cached", 0.0),
        )

    def get_semantic(
        self,
        prompt: str,
        vector: list[float],
        threshold: float,
        scope: str = "",
    ) -> ChatResponse | None:
        vector = validate_vector(vector)
        with closing(self._con()) as con:
            rows = con.execute(
                "SELECT content, toolcalls, tokens_in, tokens_out, embedding, finish "
                "FROM llm_cache WHERE scope=?",
                (scope,),
            ).fetchall()
        best, best_score = None, threshold
        for content, tc_json, ti, to, embedding, finish in rows:
            if embedding:
                score = cosine_similarity(vector, json.loads(embedding))
                if score >= best_score:
                    best_score, best = score, (content, tc_json, ti, to, finish)
        if best is None:
            return None
        content, tc_json, ti, to, finish = best
        calls = [ToolCallRequest(**item) for item in json.loads(tc_json or "[]")]
        self.hits_sem += 1
        return ChatResponse(
            Message("assistant", content, calls),
            finish,
            Usage(ti, to, 0.0, "cached-sem", 0.0),
        )

    def put(
        self,
        key: str,
        model: str,
        prompt: str,
        response: ChatResponse,
        vector: list[float] | None = None,
        scope: str = "",
    ) -> None:
        embedding = json.dumps(validate_vector(vector)) if vector is not None else None
        toolcalls = json.dumps([vars(call) for call in response.message.tool_calls])
        with closing(self._con()) as con, con:
            con.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(key, model, prompt, embedding, content, toolcalls, tokens_in, "
                "tokens_out, ts, scope, finish) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key, model, prompt, embedding, response.message.content, toolcalls,
                    response.usage.input_tokens, response.usage.output_tokens,
                    time.time(), scope, response.finish,
                ),
            )
