"""LLM provider abstraction. Default: local Ollama (qwen2.5:3b). No paid APIs.

To use another local model, only change OLLAMA_MODEL in .env (e.g.
qwen2.5:7b, llama3.1:8b, mistral). To add a different local runtime
(llama.cpp server, LM Studio, vLLM), implement the `LLMProvider` protocol.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Protocol

import requests

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    model: str

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]] = None,
             json_mode: bool = True) -> str: ...

    def is_available(self) -> bool: ...


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b",
                 temperature: float = 0.0, num_ctx: int = 8192, timeout_s: int = 240,
                 structured_output: bool = True, embed_model: str = "nomic-embed-text",
                 think: Optional[bool] = None, num_predict: int = 1500, keep_alive: str = "30m") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.structured_output = structured_output
        self.embed_model = embed_model
        self.think = think
        self.num_predict = num_predict    # cap on output tokens: stops runaway generations
        self.keep_alive = keep_alive      # keep the model in memory between calls/runs
        self.last_usage: dict[str, Any] = {}
        self.calls = 0
        self.failures = 0

    @classmethod
    def from_settings(cls, settings: Any) -> "OllamaClient":
        c = settings.llm
        return cls(c.base_url, c.model, c.temperature, c.num_ctx, c.timeout_s, c.structured_output, c.embed_model,
                   getattr(c, "think", None), getattr(c, "num_predict", 1500), getattr(c, "keep_alive", "30m"))

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    def is_available(self) -> bool:
        try:
            models = self.list_models()
        except requests.RequestException:
            return False
        if self.model in models or f"{self.model}:latest" in models:
            return True
        return ":" not in self.model and any(m.startswith(self.model + ":") for m in models)

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]] = None,
             json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "seed": 42,
                        "num_predict": self.num_predict},
            "keep_alive": self.keep_alive,
        }
        if json_mode:
            payload["format"] = schema if (schema and self.structured_output) else "json"
        if self.think is not None:
            payload["think"] = self.think
        self.calls += 1
        t0 = time.time()
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
            if r.status_code == 400 and "think" in payload:
                # model does not support the think flag (e.g. qwen2.5) -> retry without it
                payload.pop("think")
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
            if r.status_code == 400 and schema is not None and payload.get("format") != "json":
                # older Ollama versions do not accept a JSON schema in `format`
                payload["format"] = "json"
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.RequestException as exc:
            self.failures += 1
            raise LLMError(f"Ollama request failed: {exc}") from exc
        data = r.json()
        content = data.get("message", {}).get("content", "")
        # Ollama reports token counts and timings (nanoseconds) with every response.
        self.last_usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
            "output_tokens": data.get("eval_count", 0) or 0,
            "load_seconds": round((data.get("load_duration", 0) or 0) / 1e9, 2),
        }
        log.debug("LLM call %.1fs, %d chars, usage=%s", time.time() - t0, len(content), self.last_usage)
        return content

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        r = requests.post(f"{self.base_url}/api/embed", json={"model": model or self.embed_model, "input": texts},
                          timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["embeddings"]


class FakeLLM:
    """Deterministic stand-in used by tests and offline demos.

    `responses` is a list of strings returned in order (cycled), or a callable
    (system, user) -> str.
    """

    def __init__(self, responses: Any, model: str = "fake-llm") -> None:
        self.responses = responses
        self.model = model
        self.calls = 0
        self.failures = 0
        self.prompts: list[tuple[str, str]] = []

    def is_available(self) -> bool:
        return True

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]] = None,
             json_mode: bool = True) -> str:
        self.prompts.append((system, user))
        self.calls += 1
        if callable(self.responses):
            return self.responses(system, user)
        return self.responses[(self.calls - 1) % len(self.responses)]
