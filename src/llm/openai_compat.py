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

import logging
import os
import time
from typing import Any, Optional

import requests

from .client import LLMError

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
                 max_rate_limit_retries: int = 6, session: Optional[requests.Session] = None) -> None:
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
        self.last_usage: dict[str, Any] = {}
        self.calls = 0
        self.failures = 0
        # attributes InstrumentedLLM puts in its cache key
        self.num_ctx = None
        self.think = None

    @classmethod
    def from_settings(cls, settings: Any) -> "OpenAICompatClient":
        c = settings.llm
        provider = c.provider.lower()
        base = c.api_base or PROVIDER_URLS.get(provider, "")
        if not base:
            raise LLMError(f"llm.provider={provider!r} needs llm.api_base (or LLM_API_BASE in .env)")
        key = os.environ.get(c.api_key_env, "")
        if not key and provider != "lmstudio":
            raise LLMError(f"No API key: put {c.api_key_env}=... in your .env file (never in settings.yaml)")
        return cls(base, c.model, key, c.temperature, c.timeout_s, c.num_predict, provider)

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
        self.calls += 1
        wait = 2.0
        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=payload,
                                   headers=self._headers(), timeout=self.timeout_s)
            except requests.RequestException as exc:
                self.failures += 1
                raise LLMError(f"{self.provider} request failed: {exc}") from exc
            if r.status_code == 429 and attempt < self.max_rate_limit_retries:
                retry_after = r.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else wait
                log.info("%s rate limit (429); waiting %.1fs", self.provider, delay)
                time.sleep(min(delay, 60.0))
                wait = min(wait * 2, 60.0)
                continue
            if r.status_code == 400 and "response_format" in payload:
                # some models do not support JSON mode -> plain text (our JSON repair handles it)
                payload.pop("response_format")
                self.use_json_format = False
                continue
            if r.status_code >= 400:
                self.failures += 1
                raise LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            usage = data.get("usage") or {}
            self.last_usage = {"prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                               "output_tokens": usage.get("completion_tokens", 0) or 0}
            choices = data.get("choices") or [{}]
            return (choices[0].get("message") or {}).get("content") or ""
        self.failures += 1
        raise LLMError(f"{self.provider}: still rate-limited after {self.max_rate_limit_retries} retries")

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        raise LLMError("Embeddings are only supported through Ollama (set llm.provider: ollama for Chroma)")


def make_llm_client(settings: Any) -> Any:
    """The LLM client selected by `llm.provider` (default: local Ollama)."""
    from .client import OllamaClient
    provider = (settings.llm.provider or "ollama").lower()
    if provider == "ollama":
        return OllamaClient.from_settings(settings)
    return OpenAICompatClient.from_settings(settings)
