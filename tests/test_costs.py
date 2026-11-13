import json
from urllib.error import URLError

import pytest

from llm_harness_lite.utils import costs
from llm_harness_lite.utils.costs import CostManager


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


def test_override_precedes_static_fallback(tmp_path):
    manager = CostManager(
        overrides={"gpt-4o-mini": {
            "input_per_million": 2.0,
            "output_per_million": 4.0,
        }},
        cache_path=tmp_path / "prices.json",
    )

    assert manager.estimate("gpt-4o-mini", 1000, 500) == pytest.approx(0.004)
    assert manager.lookup("gpt-4o-mini").source == "override"


def test_refresh_loads_openrouter_prices_and_persists_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        costs,
        "urlopen",
        lambda request, timeout: _Response({
            "data": [{
                "id": "vendor/model-name-001",
                "pricing": {"prompt": "0.000002", "completion": "0.000003"},
            }],
        }),
    )
    cache_path = tmp_path / "prices.json"
    manager = CostManager(cache_path=cache_path)

    assert manager.initialize()
    assert manager.estimate("model-name", 1_000_000, 1_000_000) == pytest.approx(5.0)
    assert manager.lookup("vendor/model-name").source == "openrouter"

    def unexpected_request(*args, **kwargs):
        raise AssertionError("fresh cache should avoid a network request")

    monkeypatch.setattr(costs, "urlopen", unexpected_request)
    cached = CostManager(cache_path=cache_path)
    assert not cached.initialize()
    assert cached.estimate("model-name", 1_000_000, 1_000_000) == pytest.approx(5.0)


def test_refresh_failure_warns_and_uses_static_fallback(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(costs, "urlopen", offline)
    manager = CostManager(cache_path=tmp_path / "prices.json")

    with pytest.warns(RuntimeWarning, match="using cached/fallback prices"):
        assert not manager.initialize()
    assert manager.estimate("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_unknown_remote_model_warns_and_local_cost_is_explicit(tmp_path):
    manager = CostManager(
        overrides={"local-paid": {
            "input_per_million": 1.0,
            "output_per_million": 2.0,
        }},
        cache_path=tmp_path / "prices.json",
    )

    with pytest.warns(RuntimeWarning, match="No price found"):
        assert manager.estimate("private/model", 100, 100) == 0.0
    assert manager.estimate("ollama/qwen", 100, 100, local=True) == 0.0
    assert manager.estimate("local-paid", 1000, 500, local=True) == pytest.approx(0.002)
