"""Central pricing table. Contributors: add OSS/local as 0.0 here."""
from __future__ import annotations

# $ per 1M tokens (input, output) — update as prices change
PRICES = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.0),
    "claude-3-5-haiku": (0.80, 4.0), "claude-sonnet-4": (3.0, 15.0),
    "gemini-2.0-flash": (0.10, 0.40),
}

def estimate_cost(model: str, inp: int, out: int) -> float:
    key = next((k for k in PRICES if k in model.lower()), None)
    if key is None:  # local / OSS / unknown → treat as free
        return 0.0
    pi, po = PRICES[key]
    return (inp * pi + out * po) / 1_000_000
