"""Phase 1 speed features: LLM metering, disk cache and rule-first gating."""
import json

from src.llm.client import FakeLLM
from src.llm.instrumented import InstrumentedLLM, llm_stage


class CountingLLM(FakeLLM):
    """FakeLLM that also reports token usage like the Ollama client does."""

    def chat(self, system, user, schema=None, json_mode=True):
        out = super().chat(system, user, schema, json_mode)
        self.last_usage = {"prompt_tokens": len(user.split()), "output_tokens": len(out.split())}
        return out


def test_cache_returns_saved_answer_without_calling_model(tmp_path):
    inner = CountingLLM(['{"a": 1}'])
    llm = InstrumentedLLM(inner, tmp_path)
    with llm_stage(llm, "entities"):
        first = llm.chat("sys", "same prompt")
        second = llm.chat("sys", "same prompt")
    assert first == second == '{"a": 1}'
    assert inner.calls == 1                      # the model ran only once
    st = llm.summary()
    assert st["calls"] == 1 and st["cache_hits"] == 1
    assert st["by_stage"]["entities"]["prompt_tokens"] == 2
    # the cache survives a new wrapper (= a new run of the pipeline)
    again = InstrumentedLLM(CountingLLM(["different"]), tmp_path)
    assert again.chat("sys", "same prompt") == '{"a": 1}'


def test_cache_key_depends_on_prompt_and_model(tmp_path):
    llm = InstrumentedLLM(CountingLLM(["x", "y"]), tmp_path)
    assert llm.chat("sys", "prompt A") == "x"
    assert llm.chat("sys", "prompt B") == "y"    # different prompt -> new call
    other_model = InstrumentedLLM(CountingLLM(["z"], model="other-model"), tmp_path)
    assert other_model.chat("sys", "prompt A") == "z"


def test_no_cache_when_disabled(tmp_path):
    inner = CountingLLM(["1", "2"])
    llm = InstrumentedLLM(inner, cache_dir=None)
    llm.chat("s", "p")
    llm.chat("s", "p")
    assert inner.calls == 2 and llm.summary()["cache_hits"] == 0


def _llm_answer(system, user):
    if "Extract Additive Manufacturing entities" in user:
        return json.dumps({"entities": []})
    if "Extract relations" in user:
        return json.dumps({"relations": []})
    return json.dumps({"findings": []})


def test_gating_skips_chunks_without_entities(tmp_project):
    from src.ingestion.loader import parse_document
    from src.pipeline import ExtractionPipeline

    doc_path = tmp_project.path("demo_dir") / "demo_01_lpbf_ti64.md"
    doc = parse_document(doc_path, tmp_project)

    tmp_project.extraction.llm_gate = False
    ungated = ExtractionPipeline(tmp_project, llm=FakeLLM(_llm_answer), mode="hybrid").extract(doc)
    tmp_project.extraction.llm_gate = True
    tmp_project.llm.cache = False
    gated = ExtractionPipeline(tmp_project, llm=FakeLLM(_llm_answer), mode="hybrid").extract(doc)

    assert gated.stats.llm_chunks == ungated.stats.llm_chunks > 0
    assert gated.stats.llm_chunks_skipped <= gated.stats.llm_chunks
    assert gated.stats.llm_calls <= ungated.stats.llm_calls
    # rule-based results are identical: gating only saves LLM calls
    assert gated.stats.num_mentions == ungated.stats.num_mentions
    assert "entities" in gated.stats.llm_by_stage


def test_second_run_is_served_from_cache(tmp_project):
    from src.ingestion.loader import parse_document
    from src.pipeline import ExtractionPipeline

    doc = parse_document(tmp_project.path("demo_dir") / "demo_01_lpbf_ti64.md", tmp_project)
    first = ExtractionPipeline(tmp_project, llm=FakeLLM(_llm_answer), mode="hybrid").extract(doc)
    second = ExtractionPipeline(tmp_project, llm=FakeLLM(_llm_answer), mode="hybrid").extract(doc)
    assert first.stats.llm_cache_hits == 0 and first.stats.llm_calls > 0
    assert second.stats.llm_cache_hits == first.stats.llm_calls
    assert sum(v["calls"] for v in second.stats.llm_by_stage.values()) == 0   # no model calls at all
    assert second.stats.num_relations == first.stats.num_relations


def test_ollama_client_reads_token_usage(monkeypatch):
    from src.llm import client as client_mod

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}, "prompt_eval_count": 120, "eval_count": 30,
                    "load_duration": 2_000_000_000}

    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return Resp()

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = client_mod.OllamaClient(num_predict=800, keep_alive="10m")
    assert c.chat("s", "u") == "{}"
    assert c.last_usage == {"prompt_tokens": 120, "output_tokens": 30, "load_seconds": 2.0}
    assert sent["options"]["num_predict"] == 800 and sent["keep_alive"] == "10m"


def test_stage_token_cap_is_applied_and_restored(tmp_path):
    seen = []

    class CapLLM(CountingLLM):
        num_predict = 1500

        def chat(self, system, user, schema=None, json_mode=True):
            seen.append(self.num_predict)
            return super().chat(system, user, schema, json_mode)

    inner = CapLLM(["{}"])
    llm = InstrumentedLLM(inner, None, stage_max_tokens={"relations": 800})
    with llm_stage(llm, "relations"):
        llm.chat("s", "u1")
    llm.chat("s", "u2")                         # stage "other": no cap
    assert seen == [800, 1500] and inner.num_predict == 1500
