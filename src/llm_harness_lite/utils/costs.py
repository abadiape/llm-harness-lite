"""Model price lookup with OpenRouter refreshes, disk cache, and local overrides."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "llm-harness-lite" / "openrouter-prices.json"
DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_REQUEST_TIMEOUT_SECONDS = 2.0

# Fallback prices in USD per 1M tokens (input, output).
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-sonnet-4": (3.0, 15.0),
    "gemini-2.0-flash": (0.10, 0.40),
}


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    source: str
    model_id: str


def _price(model_id: str, raw: Any, source: str) -> ModelPricing:
    if isinstance(raw, Mapping):
        input_rate = raw.get("input_per_million")
        output_rate = raw.get("output_per_million")
    elif isinstance(raw, (tuple, list)) and len(raw) == 2:
        input_rate, output_rate = raw
    else:
        raise ValueError(f"Invalid price for {model_id!r}")

    try:
        input_rate = float(input_rate)
        output_rate = float(output_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid price for {model_id!r}") from exc

    if (not math.isfinite(input_rate) or not math.isfinite(output_rate)
            or input_rate < 0 or output_rate < 0):
        raise ValueError(f"Prices for {model_id!r} must be finite and non-negative")

    return ModelPricing(input_rate, output_rate, source, model_id)


def _match_price(model: str, prices: Mapping[str, ModelPricing],
                 allow_version_suffix: bool = False,
                 allow_model_variant: bool = False) -> ModelPricing | None:
    requested = model.strip().casefold()
    if not requested:
        return None

    exact = prices.get(requested)
    if exact:
        return exact

    requested_tail = requested.rsplit("/", 1)[-1]
    matches = [
        price for model_id, price in prices.items()
        if model_id.rsplit("/", 1)[-1] == requested_tail
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None

    if allow_model_variant:
        variants = [
            price for model_id, price in prices.items()
            if model_id.rsplit("/", 1)[-1].startswith(requested_tail)
            and model_id.rsplit("/", 1)[-1][len(requested_tail):].startswith(("-", ":"))
        ]
        if len(variants) == 1:
            return variants[0]
        if variants:
            return None

    if allow_version_suffix:
        versioned = [
            (model_id, price) for model_id, price in prices.items()
            if requested_tail.startswith(model_id.rsplit("/", 1)[-1])
            and requested_tail[len(model_id.rsplit("/", 1)[-1]):].startswith(("-", ":"))
        ]
        if versioned:
            longest = max(len(model_id.rsplit("/", 1)[-1]) for model_id, _ in versioned)
            best = [price for model_id, price in versioned
                    if len(model_id.rsplit("/", 1)[-1]) == longest]
            if len(best) == 1:
                return best[0]
    return None


class CostManager:
    """Resolve prices from overrides, cached OpenRouter data, then static fallbacks."""

    def __init__(
        self,
        overrides: Mapping[str, Any] | None = None,
        cache_path: str | Path | None = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ):
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if overrides is not None and not isinstance(overrides, Mapping):
            raise ValueError("pricing overrides must be a mapping")

        self.cache_path = Path(cache_path).expanduser() if cache_path else DEFAULT_CACHE_PATH
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._overrides = {
            str(model).strip().casefold(): _price(str(model), value, "override")
            for model, value in (overrides or {}).items()
        }
        self._fallback = {
            model.casefold(): _price(model, value, "fallback")
            for model, value in PRICES.items()
        }
        self._catalog: dict[str, ModelPricing] = {}
        self._fetched_at = 0.0
        self._warned_unknown: set[str] = set()
        self._load_cache()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> CostManager:
        """Create a manager from the optional `costs` section in config.yaml."""
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("costs configuration must be a mapping")
        return cls(
            overrides=config.get("overrides"),
            cache_path=config.get("cache_path"),
            cache_ttl_seconds=config.get("cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS),
            request_timeout_seconds=config.get(
                "request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS
            ),
        )

    def _load_cache(self) -> None:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            warnings.warn(f"Could not read pricing cache {self.cache_path}: {exc}",
                          RuntimeWarning, stacklevel=2)
            return

        if not isinstance(raw, Mapping) or not isinstance(raw.get("prices"), Mapping):
            warnings.warn(f"Ignoring invalid pricing cache {self.cache_path}",
                          RuntimeWarning, stacklevel=2)
            return

        try:
            fetched_at = float(raw["fetched_at"])
            catalog = {
                str(model).casefold(): _price(str(model), value, "openrouter")
                for model, value in raw["prices"].items()
            }
            if not math.isfinite(fetched_at) or fetched_at <= 0 or not catalog:
                raise ValueError("cache timestamp or model list is invalid")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(f"Ignoring invalid pricing cache {self.cache_path}: {exc}",
                          RuntimeWarning, stacklevel=2)
            return

        self._catalog = catalog
        self._fetched_at = fetched_at

    def _cache_is_fresh(self) -> bool:
        age = time.time() - self._fetched_at
        return bool(self._catalog) and 0 <= age < self.cache_ttl_seconds

    @staticmethod
    def _parse_catalog(payload: Any) -> dict[str, ModelPricing]:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise ValueError("OpenRouter response did not contain a model list")

        catalog = {}
        for entry in payload["data"]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                continue
            pricing = entry.get("pricing")
            if not isinstance(pricing, Mapping):
                continue
            try:
                input_rate = float(pricing["prompt"]) * 1_000_000
                output_rate = float(pricing["completion"]) * 1_000_000
                catalog[entry["id"].casefold()] = _price(
                    entry["id"],
                    {"input_per_million": input_rate, "output_per_million": output_rate},
                    "openrouter",
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not catalog:
            raise ValueError("OpenRouter response contained no usable model prices")
        return catalog

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": self._fetched_at,
            "prices": {
                model_id: {
                    "input_per_million": price.input_per_million,
                    "output_per_million": price.output_per_million,
                }
                for model_id, price in self._catalog.items()
            },
        }
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.cache_path.parent,
                delete=False
            ) as cache_file:
                temp_path = Path(cache_file.name)
                json.dump(payload, cache_file)
            os.replace(temp_path, self.cache_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def initialize(self) -> bool:
        """Load a fresh cache or fetch an updated catalog; return whether fetched."""
        if self._cache_is_fresh():
            return False
        return self.refresh()

    def refresh(self, force: bool = False) -> bool:
        """Fetch OpenRouter prices. On failure, retain cached and static prices."""
        if not force and self._cache_is_fresh():
            return False

        request = Request(
            OPENROUTER_MODELS_URL,
            headers={"User-Agent": "llm-harness-lite"},
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                catalog = self._parse_catalog(json.loads(response.read()))
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            warnings.warn(
                f"Could not refresh OpenRouter prices; using cached/fallback prices: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        self._catalog = catalog
        self._fetched_at = time.time()
        try:
            self._save_cache()
        except OSError as exc:
            warnings.warn(f"Could not save pricing cache {self.cache_path}: {exc}",
                          RuntimeWarning, stacklevel=2)
        return True

    def lookup(self, model: str) -> ModelPricing | None:
        """Return the matching price and its source, or None if unresolved."""
        return (
            _match_price(model, self._overrides)
            or _match_price(model, self._catalog, allow_model_variant=True)
            or _match_price(model, self._fallback, allow_version_suffix=True)
        )

    def estimate(self, model: str, inp: int, out: int, *, local: bool = False) -> float:
        if inp < 0 or out < 0:
            raise ValueError("token counts must be non-negative")
        if local:
            pricing = _match_price(model, self._overrides)
            if pricing is None:
                return 0.0
        else:
            pricing = self.lookup(model)

        if pricing is None:
            model_key = model.casefold()
            if model_key not in self._warned_unknown:
                warnings.warn(
                    f"No price found for model {model!r}; reporting $0. "
                    "Add a costs.overrides entry if this model has a usage cost.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_unknown.add(model_key)
            return 0.0

        return (
            inp * pricing.input_per_million + out * pricing.output_per_million
        ) / 1_000_000


_DEFAULT_MANAGER: CostManager | None = None


def estimate_cost(
    model: str,
    inp: int,
    out: int,
    *,
    manager: CostManager | None = None,
    local: bool = False,
) -> float:
    """Estimate USD from token counts using the default or supplied manager."""
    global _DEFAULT_MANAGER
    if manager is None:
        if _DEFAULT_MANAGER is None:
            _DEFAULT_MANAGER = CostManager()
        manager = _DEFAULT_MANAGER
    return manager.estimate(model, inp, out, local=local)
