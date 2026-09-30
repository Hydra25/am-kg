"""Measure and cache LLM calls, for any provider.

`InstrumentedLLM` wraps an LLM client (Ollama, FakeLLM, a future cloud client)
and adds two things:

1. **Metrics per stage** ("entities", "relations", "findings", "normalization"):
   number of calls, wall-clock seconds, prompt/output tokens and cache hits.
   This answers "where does the time go?" before we optimise anything.

2. **A disk cache** (data/cache/llm/): the same model + prompt + settings returns
   the saved answer instead of calling the model again. With temperature 0 the
   model is deterministic, so a re-run of an unchanged paper costs ~0 seconds.
   Changing the prompt, the model or its options changes the cache key, so stale
   answers are never reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)


@dataclass
class StageStats:
    calls: int = 0            # real model calls (cache misses)
    cache_hits: int = 0
    errors: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


class InstrumentedLLM:
    """Drop-in replacement for an LLM client: same `chat()` signature."""

    def __init__(self, inner: Any, cache_dir: Optional[Path] = None,
                 stage_max_tokens: Optional[dict[str, int]] = None) -> None:
        self.inner = inner
        # Per-stage output caps (e.g. relations: 800). A small model sometimes
        # starts repeating itself; the cap stops that early instead of after
        # thousands of tokens. Truncated JSON is repaired downstream.
        self.stage_max_tokens = dict(stage_max_tokens or {})
        self.model = getattr(inner, "model", "unknown")
        self.stage = "other"
        self.stats: dict[str, StageStats] = {}
        self.by_provider: dict[str, int] = {}     # which member of a fallback chain answered
        # Only cache deterministic generations (temperature 0).
        temperature = getattr(inner, "temperature", 0.0) or 0.0
        self.cache_dir = Path(cache_dir) / "llm" if cache_dir and temperature == 0 else None
        # Everything that changes the model's answer must be part of the cache key.
        self._key_extra = json.dumps({
            "model": self.model,
            "provider": getattr(inner, "provider", "ollama"),
            "temperature": temperature,
            "num_ctx": getattr(inner, "num_ctx", None),
            "num_predict": getattr(inner, "num_predict", None),
            "think": getattr(inner, "think", None),
        }, sort_keys=True)

    # --- behave like the wrapped client --------------------------------------
    def __getattr__(self, name: str) -> Any:          # embed(), list_models(), ...
        return getattr(self.inner, name)

    def is_available(self) -> bool:
        return self.inner.is_available()

    # --- stage bookkeeping -----------------------------------------------------
    @contextmanager
    def use_stage(self, stage: str) -> Iterator[None]:
        previous, self.stage = self.stage, stage
        try:
            yield
        finally:
            self.stage = previous

    def _stage_stats(self) -> StageStats:
        return self.stats.setdefault(self.stage, StageStats())

    def reset_stats(self) -> None:
        self.stats = {}
        self.by_provider = {}

    def summary(self) -> dict[str, Any]:
        """Totals plus a per-stage breakdown (JSON-serialisable)."""
        total = StageStats()
        for st in self.stats.values():
            for f in total.__dataclass_fields__:
                setattr(total, f, getattr(total, f) + getattr(st, f))
        out = asdict(total)
        out["seconds"] = round(out["seconds"], 2)
        out["by_stage"] = {k: {**asdict(v), "seconds": round(v.seconds, 2)} for k, v in sorted(self.stats.items())}
        out["by_provider"] = dict(self.by_provider)
        return out

    # --- the actual call -------------------------------------------------------
    def _cache_path(self, system: str, user: str, schema: Optional[dict], json_mode: bool) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        blob = json.dumps([self._key_extra, self._max_tokens(), system, user, schema, json_mode],
                          sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def _max_tokens(self) -> Optional[int]:
        return self.stage_max_tokens.get(self.stage)

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]] = None, json_mode: bool = True) -> str:
        st = self._stage_stats()
        path = self._cache_path(system, user, schema, json_mode)
        if path is not None and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))["response"]
                st.cache_hits += 1
                return cached
            except (OSError, ValueError, KeyError):
                log.debug("Ignoring unreadable cache entry %s", path)

        cap = self._max_tokens()
        default_cap = getattr(self.inner, "num_predict", None)
        if cap and default_cap is not None:
            self.inner.num_predict = min(cap, default_cap)
        t0 = time.perf_counter()
        try:
            response = self.inner.chat(system, user, schema=schema, json_mode=json_mode)
        except Exception:
            st.errors += 1
            st.seconds += time.perf_counter() - t0
            raise
        finally:
            if cap and default_cap is not None:
                self.inner.num_predict = default_cap
        st.calls += 1
        st.seconds += time.perf_counter() - t0
        prov = getattr(self.inner, "last_provider", None) or getattr(self.inner, "provider", "ollama")
        self.by_provider[prov] = self.by_provider.get(prov, 0) + 1
        usage = getattr(self.inner, "last_usage", None) or {}
        st.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        st.output_tokens += int(usage.get("output_tokens", 0) or 0)

        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"model": self.model, "stage": self.stage, "response": response},
                                          ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)      # atomic: a crash never leaves a half-written entry
            except OSError as exc:
                log.warning("Could not write LLM cache entry: %s", exc)
        return response


@contextmanager
def llm_stage(llm: Any, stage: str) -> Iterator[None]:
    """`with llm_stage(llm, "entities"):` works for wrapped and plain clients."""
    if isinstance(llm, InstrumentedLLM):
        with llm.use_stage(stage):
            yield
    else:
        yield
