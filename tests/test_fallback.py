"""Fallback chain: free cloud APIs in order, local Ollama last."""
import json

import pytest

from src.llm.client import LLMError, LLMHTTPError
from src.llm.fallback import FallbackLLM


class Member:
    """Fake provider: answers, or raises the errors it is given (one per call)."""

    def __init__(self, name, errors=(), answer=None, available=True):
        self.model = f"{name}-model"
        self.errors = list(errors)
        self.answer = answer or f"answer from {name}"
        self.calls = 0
        self.num_predict = 1500
        self.available = available
        self.last_usage = {"prompt_tokens": 10, "output_tokens": 5}

    def is_available(self):
        return self.available

    def chat(self, system, user, schema=None, json_mode=True):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.answer


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def test_switches_to_next_provider_on_rate_limit_and_comes_back_later(tmp_path):
    clock = Clock()
    groq = Member("groq", errors=[LLMHTTPError("429", 429, retry_after=30)])
    cerebras = Member("cerebras")
    chain = FallbackLLM([("groq", groq), ("cerebras", cerebras)], tmp_path / "q.json", clock)
    assert chain.chat("s", "u") == "answer from cerebras"      # same call, next provider
    assert chain.last_provider == "cerebras"
    assert chain.chat("s", "u") == "answer from cerebras"      # groq still paused
    assert groq.calls == 1
    clock.t += 31                                              # Retry-After has passed
    assert chain.chat("s", "u") == "answer from groq"
    assert chain.calls_by_provider == {"cerebras": 2, "groq": 1}


def test_daily_quota_pauses_long_and_survives_restart(tmp_path):
    clock = Clock()
    state = tmp_path / "q.json"
    body = '{"error": {"message": "Rate limit reached: tokens per day (TPD)"}}'
    groq = Member("groq", errors=[LLMHTTPError("429", 429, body=body)])
    chain = FallbackLLM([("groq", groq), ("ollama", Member("ollama"))], state, clock)
    assert chain.chat("s", "u") == "answer from ollama"
    saved = json.loads(state.read_text())
    assert saved["groq"] - clock.t == pytest.approx(3600)
    # a new run (new object) still skips groq
    again = FallbackLLM([("groq", Member("groq")), ("ollama", Member("ollama"))], state, clock)
    assert again.chat("s", "u") == "answer from ollama"


def test_bad_key_disables_provider_and_all_down_raises(tmp_path):
    clock = Clock()
    a = Member("a", errors=[LLMHTTPError("401", 401)])
    b = Member("b", errors=[LLMError("connection refused")])
    chain = FallbackLLM([("a", a), ("b", b)], None, clock)
    with pytest.raises(LLMError, match="All LLM providers are unavailable"):
        chain.chat("s", "u")
    assert chain.cooldown_until["a"] - clock.t == 24 * 3600
    assert chain.cooldown_until["b"] - clock.t == 60


def test_token_caps_reach_every_member_and_availability(tmp_path):
    a, b = Member("a", available=False), Member("b")
    chain = FallbackLLM([("a", a), ("b", b)], None, Clock())
    chain.num_predict = 800
    assert a.num_predict == b.num_predict == 800
    assert chain.is_available()                    # b is up
    assert chain.cooldown_until["a"] > 0          # a paused because unreachable at start-up
    assert chain.chat("s", "u") == "answer from b"


def test_chain_from_settings_skips_providers_without_keys(tmp_project, monkeypatch):
    for v in ("GROQ_API_KEY", "GROQ_MODEL", "CEREBRAS_API_KEY", "CEREBRAS_MODEL", "GEMINI_API_KEY",
              "GEMINI_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    chain = FallbackLLM.from_settings(tmp_project)
    assert [n for n, _ in chain.members] == ["groq", "ollama"]
    assert "groq:some-model" in chain.model


def test_pipeline_records_calls_per_provider(tmp_project):
    from src.ingestion.loader import parse_document
    from src.pipeline import ExtractionPipeline
    from tests.test_speed import _llm_answer

    class Answering(Member):
        def chat(self, system, user, schema=None, json_mode=True):
            super().chat(system, user)
            return _llm_answer(system, user)

    tmp_project.llm.cache = False
    first = Answering("groq", errors=[LLMHTTPError("429", 429, retry_after=9999)])
    chain = FallbackLLM([("groq", first), ("ollama", Answering("ollama"))], None, Clock())
    doc = parse_document(tmp_project.path("demo_dir") / "demo_01_lpbf_ti64.md", tmp_project)
    res = ExtractionPipeline(tmp_project, llm=chain, mode="hybrid").extract(doc)
    assert res.stats.llm_by_provider.get("ollama", 0) > 0
    assert "groq" not in res.stats.llm_by_provider          # its only call failed -> not counted
