"""Optional cloud / server LLMs through the OpenAI-compatible chat API.

Many providers speak the same `/chat/completions` protocol, several with a free
tier: Groq, Cerebras, Google Gemini (OpenAI endpoint), OpenRouter - and local
servers such as LM Studio or vLLM. Use them when the laptop is too slow (or too
hot); Ollama stays the default and the project never needs a paid API.

Rules this client follows:
  * the API key is read from an environment variable only (never from a file in git)
  * HTTP 429 (rate limit) -> wait (Retry-After header or exponential back-off) and retry
  * `last_usage` reports tokens so InstrumentedLLM can meter cloud calls too

Note: free tiers may log prompts or use them for training. Only send published
papers, never unpublished manuscripts.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import requests

from .client import LLMError, LLMHTTPError

log = logging.getLogger(__name__)

# provider name -> OpenAI-compatible base URL (the model name is always set by the user,
# because the free model catalogues change often)
PROVIDER_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "lmstudio": "http://localhost:1234/v1",
}


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.0,
                 timeout_s: int = 120, num_predict: int = 1500, provider: str = "openai_compat",
                 max_rate_limit_retries: int = 6, session: Optional[requests.Session] = None,
                 extra_body: Optional[dict[str, Any]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.num_predict = num_predict
        self.provider = provider
        self.max_rate_limit_retries = max_rate_limit_retries
        self.http = session or requests.Session()
        self.use_json_format = True
        self.daily_limit_until = 0.0     # set when the provider reports a per-day quota as used up
        # provider-specific request fields, e.g. reasoning controls for reasoning models
        self.extra_body: dict[str, Any] = dict(extra_body or {})
        self.last_usage: dict[str, Any] = {}
        self.calls = 0
        self.failures = 0
        # attributes InstrumentedLLM puts in its cache key
        self.num_ctx = None
        self.think = None

    @classmethod
    def from_settings(cls, settings: Any) -> "OpenAICompatClient":
        c = settings.llm
        return cls.from_spec({"provider": c.provider, "model": c.model, "api_base": c.api_base,
                              "api_key_env": c.api_key_env}, settings)

    @classmethod
    def from_spec(cls, spec: dict[str, Any], settings: Any, max_rate_limit_retries: int = 6) -> "OpenAICompatClient":
        """spec: provider, model | model_env, api_key_env, api_base (optional)."""
        c = settings.llm
        provider = str(spec.get("provider", "openai_compat")).lower()
        base = spec.get("api_base") or PROVIDER_URLS.get(provider, "")
        if not base:
            raise LLMError(f"provider {provider!r} needs an api_base (or LLM_API_BASE in .env)")
        model = spec.get("model") or os.environ.get(spec.get("model_env") or "", "")
        if not model:
            raise LLMError(f"provider {provider!r}: no model set ({spec.get('model_env') or 'model'} in .env)")
        key_env = spec.get("api_key_env") or "LLM_API_KEY"
        key = os.environ.get(key_env, "")
        if not key and provider != "lmstudio":
            raise LLMError(f"No API key for {provider}: put {key_env}=... in your .env file (never in settings.yaml)")
        extra = dict(spec.get("extra_body") or reasoning_defaults(provider, model))
        num_predict = int(spec.get("num_predict") or c.num_predict)
        return cls(base, model, key, c.temperature, c.timeout_s, num_predict, provider,
                   max_rate_limit_retries=max_rate_limit_retries, extra_body=extra)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def is_available(self) -> bool:
        try:
            r = self.http.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        r = self.http.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]] = None,
             json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
            "seed": 42,
        }
        if json_mode and self.use_json_format:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        if self.daily_limit_until and time.time() < self.daily_limit_until:
            self.failures += 1                         # fail fast: no point waiting for hours per chunk
            raise LLMHTTPError(f"{self.provider}: daily limit reached earlier in this run", 429,
                               self.daily_limit_until - time.time(), "daily limit")
        self.calls += 1
        wait = 2.0
        for attempt in range(self.max_rate_limit_retries + 3 + len(self.extra_body)):   # + fallbacks
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=payload,
                                   headers=self._headers(), timeout=self.timeout_s)
            except requests.RequestException as exc:
                self.failures += 1
                raise LLMError(f"{self.provider} request failed: {exc}") from exc
            if r.status_code == 429 and is_daily_limit(r.text):
                # e.g. Groq: "Rate limit reached for model ... on tokens per day (TPD): Limit 200000, Used ..."
                self.daily_limit_until = time.time() + (_retry_after(r) or 3600.0)
                self.failures += 1
                log.warning("%s: DAILY LIMIT REACHED for %s -> %s | Finished calls are cached: run the same "
                            "command again after the reset and only the missing chunks are sent.",
                            self.provider, self.model, _error_message(r.text))
                raise LLMHTTPError(f"{self.provider} daily limit: {_error_message(r.text)}", 429,
                                   _retry_after(r), r.text[:500])
            if r.status_code == 429 and attempt < self.max_rate_limit_retries:
                delay = _retry_after(r) or wait
                log.info("%s rate limit (429); waiting %.1fs | %s", self.provider, delay,
                         _error_message(r.text)[:160])
                time.sleep(min(delay, 60.0))
                wait = min(wait * 2, 60.0)
                continue
            if r.status_code == 400 and "seed" in payload and "seed" in r.text.lower():
                payload.pop("seed")                    # some providers reject the seed field
                continue
            bad_extra = [k for k in self.extra_body if k in payload and k in r.text] if r.status_code == 400 else []
            if bad_extra:                              # provider does not know this reasoning field
                for k in bad_extra:
                    payload.pop(k)
                    self.extra_body.pop(k, None)
                continue
            if r.status_code == 400 and "response_format" in payload:
                # some models do not support JSON mode -> plain text (our JSON repair handles it)
                payload.pop("response_format")
                self.use_json_format = False
                continue
            if r.status_code >= 400:
                self.failures += 1
                raise LLMHTTPError(f"{self.provider} HTTP {r.status_code}: {r.text[:300]}", r.status_code,
                                   _retry_after(r), r.text[:500])
            data = r.json()
            usage = data.get("usage") or {}
            self.last_usage = {"prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                               "output_tokens": usage.get("completion_tokens", 0) or 0}
            choices = data.get("choices") or [{}]
            return (choices[0].get("message") or {}).get("content") or ""
        self.failures += 1
        raise LLMHTTPError(f"{self.provider}: rate-limited (HTTP 429)", 429, _retry_after(r), r.text[:500])

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        raise LLMError("Embeddings are only supported through Ollama (set llm.provider: ollama for Chroma)")


_DAILY_RE = re.compile(r"per day|\bTPD\b|\bRPD\b|daily|quota exceeded|RESOURCE_EXHAUSTED", re.I)


def is_daily_limit(body: str) -> bool:
    """True if a 429 says a per-day quota is used up (not the short per-minute limit)."""
    return bool(_DAILY_RE.search(body or ""))


def _error_message(body: str) -> str:
    try:
        err = json.loads(body).get("error")
        msg = err.get("message") if isinstance(err, dict) else err
        return str(msg or body)[:300]
    except (ValueError, AttributeError):
        return (body or "")[:300]


def reasoning_defaults(provider: str, model: str) -> dict[str, Any]:
    """Keep reasoning models short and their thinking out of the JSON answer (Groq API fields).
    Unknown fields are dropped automatically if a provider rejects them (HTTP 400)."""
    m = (model or "").lower()
    if provider == "groq" and m.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    if provider == "groq" and "qwen3" in m:
        return {"reasoning_effort": "low", "reasoning_format": "hidden"}
    return {}


def parse_llm_spec(text: str, settings: Any, num_predict: int = 3000) -> OpenAICompatClient:
    """'groq:openai/gpt-oss-120b' -> client (key from the provider's env variable, e.g. GROQ_API_KEY).
    Used for a separate relation-extraction model (`extract --relations-llm`)."""
    provider, _, model = text.partition(":")
    provider = provider.strip().lower()
    if not model:
        raise LLMError("relation model must look like provider:model, e.g. groq:openai/gpt-oss-120b")
    key_env = next((e.get("api_key_env") for e in settings.llm.fallback_chain
                    if str(e.get("provider", "")).lower() == provider and e.get("api_key_env")),
                   f"{provider.upper()}_API_KEY")
    # a fixed model for a whole run: wait on rate limits (Retry-After) instead of switching providers
    return OpenAICompatClient.from_spec({"provider": provider, "model": model.strip(), "api_key_env": key_env,
                                         "num_predict": num_predict}, settings, max_rate_limit_retries=12)


def _retry_after(r: Any) -> Optional[float]:
    v = (getattr(r, "headers", None) or {}).get("retry-after")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def make_llm_client(settings: Any) -> Any:
    """The LLM client selected by `llm.provider` (default: local Ollama).
    `fallback` = free cloud APIs in order, then local Ollama (see src/llm/fallback.py)."""
    from .client import OllamaClient
    provider = (settings.llm.provider or "ollama").lower()
    if provider == "ollama":
        return OllamaClient.from_settings(settings)
    if provider == "fallback":
        from .fallback import FallbackLLM
        return FallbackLLM.from_settings(settings)
    return OpenAICompatClient.from_settings(settings)
