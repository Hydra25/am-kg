"""Free cloud APIs first, local Ollama last: an automatic fallback chain.

    llm.provider: fallback            (or LLM_PROVIDER=fallback in .env)

Each call goes to the first provider in `llm.fallback_chain` that is usable:

  * a provider without an API key or model in .env is simply skipped;
  * HTTP 429 (rate limit / daily quota) -> that provider "cools down" until its
    Retry-After time (or 60 s; 1 h if the message says a daily limit is used up)
    and the SAME call goes to the next provider immediately;
  * HTTP 401/403 (bad key) -> provider disabled for 24 h;
  * network errors / 5xx  -> 60 s cool-down;
  * the last entry is normally local Ollama, which has no quota.

Cool-downs are saved in data/cache/llm_quota.json, so the next run does not
waste time on a provider whose daily quota is still used up.

Every call records which provider answered (`last_provider`); the extraction
stats show calls per provider. For the BENCHMARK use ONE fixed provider, not
the chain: mixing models inside one run makes results hard to compare.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .client import LLMError, LLMHTTPError

log = logging.getLogger(__name__)

_DAILY = re.compile(r"(per day|daily|tokens per day|requests per day|quota|RESOURCE_EXHAUSTED|TPD|RPD)", re.I)


class FallbackLLM:
    provider = "fallback"

    def __init__(self, members: list[tuple[str, Any]], state_file: Optional[Path] = None,
                 clock: Callable[[], float] = time.time) -> None:
        if not members:
            raise LLMError("fallback chain is empty: add API keys to .env or keep ollama in llm.fallback_chain")
        self.members = members
        self.state_file = Path(state_file) if state_file else None
        self.clock = clock
        self.cooldown_until: dict[str, float] = self._load_state()
        self.last_provider: Optional[str] = None
        self.last_usage: dict[str, Any] = {}
        self.calls_by_provider: dict[str, int] = {}
        self.temperature = 0.0
        self.num_ctx = None
        self.think = None
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------------ settings
    @classmethod
    def from_settings(cls, settings: Any) -> "FallbackLLM":
        from .client import OllamaClient
        from .openai_compat import OpenAICompatClient
        members: list[tuple[str, Any]] = []
        for spec in settings.llm.fallback_chain:
            prov = str(spec.get("provider", "")).lower()
            name = spec.get("name") or prov
            try:
                if prov == "ollama":
                    client = OllamaClient.from_settings(settings)
                    if spec.get("model"):
                        client.model = spec["model"]
                else:
                    # fail over fast: at most one short retry inside a provider
                    client = OpenAICompatClient.from_spec(spec, settings, max_rate_limit_retries=0)
            except LLMError as exc:
                log.info("Fallback chain: skipping %s (%s)", name, exc)
                continue
            members.append((name, client))
        return cls(members, settings.path("cache_dir") / "llm_quota.json")

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict[str, float]:
        if self.state_file and self.state_file.exists():
            try:
                return {k: float(v) for k, v in json.loads(self.state_file.read_text(encoding="utf-8")).items()}
            except (OSError, ValueError):
                pass
        return {}

    def _save_state(self) -> None:
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            now = self.clock()
            live = {k: v for k, v in self.cooldown_until.items() if v > now}
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(live, indent=1), encoding="utf-8")
            os.replace(tmp, self.state_file)
        except OSError as exc:
            log.debug("Could not save fallback state: %s", exc)

    def _cool_down(self, name: str, seconds: float, why: str) -> None:
        self.cooldown_until[name] = self.clock() + seconds
        self._save_state()
        log.warning("LLM provider %s unavailable (%s); paused for %s, trying the next one",
                    name, why, _human(seconds))

    def status(self) -> list[dict[str, Any]]:
        now = self.clock()
        return [{"name": n, "model": getattr(c, "model", "?"),
                 "paused_for_s": max(0, round(self.cooldown_until.get(n, 0) - now))} for n, c in self.members]

    # ------------------------------------------------------------------ client API
    @property
    def model(self) -> str:
        return "fallback(" + ",".join(f"{n}:{getattr(c, 'model', '?')}" for n, c in self.members) + ")"

    @property
    def num_predict(self) -> int:
        return getattr(self.members[0][1], "num_predict", 1500)

    @num_predict.setter
    def num_predict(self, value: int) -> None:       # per-stage caps from InstrumentedLLM
        for _, c in self.members:
            if hasattr(c, "num_predict"):
                c.num_predict = value

    def is_available(self) -> bool:
        ok = False
        for name, c in self.members:
            if self.cooldown_until.get(name, 0) > self.clock():
                continue
            try:
                up = c.is_available()
            except Exception:  # noqa: BLE001 - any failure means "not now"
                up = False
            if up:
                ok = True
            else:
                self.cooldown_until[name] = self.clock() + 600
                log.info("Fallback chain: %s not reachable at start-up; paused 10 min", name)
        self._save_state()
        return ok

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]] = None, json_mode: bool = True) -> str:
        self.calls += 1
        errors = []
        for name, client in self.members:
            if self.cooldown_until.get(name, 0) > self.clock():
                continue
            try:
                out = client.chat(system, user, schema=schema, json_mode=json_mode)
            except LLMHTTPError as exc:
                errors.append(f"{name}: HTTP {exc.status}")
                if exc.status == 429:
                    daily = bool(_DAILY.search(exc.body or str(exc)))
                    self._cool_down(name, exc.retry_after or (3600 if daily else 60),
                                    "daily quota used up" if daily else "rate limit")
                elif exc.status in (401, 403):
                    self._cool_down(name, 24 * 3600, "API key rejected")
                elif exc.status == 404:
                    self._cool_down(name, 24 * 3600, "model not found - check the model name")
                else:
                    self._cool_down(name, 60, f"HTTP {exc.status}")
                continue
            except LLMError as exc:
                errors.append(f"{name}: {exc}")
                self._cool_down(name, 60, "request failed")
                continue
            self.last_provider = name
            self.last_usage = getattr(client, "last_usage", {}) or {}
            self.calls_by_provider[name] = self.calls_by_provider.get(name, 0) + 1
            return out
        self.failures += 1
        raise LLMError("All LLM providers are unavailable right now: " + "; ".join(errors or ["all paused"]))

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        for _, c in self.members:
            if hasattr(c, "embed") and c.__class__.__name__ == "OllamaClient":
                return c.embed(texts, model)
        raise LLMError("Embeddings need an Ollama entry in the fallback chain")


def _human(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{int(seconds)} s"
