"""GLiNER entity backend (with a fake model - no download) and the junk-entity filter."""
import json
import re

import pytest

from src.llm.client import FakeLLM
from src.quality.junk import junk_reason


@pytest.mark.parametrize("text,etype,reason", [
    ("fiera milano", "CharacterizationMethod", "not_a_method"),   # seen on a real paper
    ("blue", "Microstructure", "generic_word"),
    ("Fig. 4 (e", "Defect", "figure_or_table_reference"),
    ("LT1", "Geometry", "specimen_id"),
    ("12.5 ± 0.3", "Property", "number_only"),
    ("x", "Material", "too_short"),
    ("MT3", "Material", "specimen_id"),        # specimen label typed as a material by the LLM
    ("1000 °C", "ExperimentalCondition", "value_not_entity"),
])
def test_junk_is_rejected(text, etype, reason):
    assert junk_reason(text, etype, in_lexicon=False) == reason


@pytest.mark.parametrize("text,etype", [
    ("digital image correlation", "CharacterizationMethod"),
    ("micro-CT", "CharacterizationMethod"),
    ("Printdur HSA", "Material"),
    ("S460NL", "Material"),
    ("H13", "Material"),                      # steel grade, not a specimen id
    ("316L", "Material"),
    ("IN718", "Material"),
    ("interlayer temperature", "ExperimentalCondition"),
])
def test_real_terms_pass(text, etype):
    assert junk_reason(text, etype, in_lexicon=False) is None


def test_lexicon_terms_are_always_trusted():
    assert junk_reason("SEM", "CharacterizationMethod", in_lexicon=True) is None


class FakeGliner:
    """Mimics gliner.GLiNER.batch_predict_entities: finds fixed phrases by regex."""
    name = "fake-gliner"
    PHRASES = {"Printdur HSA": "material or alloy", "interlayer temperature": "experimental condition",
               "fiera milano": "characterization or testing method", "laser power": "process parameter"}

    def __init__(self):
        self.calls = 0

    def batch_predict_entities(self, texts, labels, threshold=0.5):
        self.calls += 1
        out = []
        for t in texts:
            ents = []
            for phrase, label in self.PHRASES.items():
                for m in re.finditer(re.escape(phrase), t, flags=re.I):
                    ents.append({"start": m.start(), "end": m.end(), "text": m.group(), "label": label, "score": 0.8})
            out.append(ents)
        return out


def _doc(tmp_project, text):
    p = tmp_project.path("papers_dir") / "gliner_test.md"
    p.write_text(f"# GLiNER test paper\n\n## 2. Materials and methods\n\n{text}\n\n## 3. Results\n\n"
                 "The tensile strength was 700 MPa.\n", encoding="utf-8")
    from src.ingestion.loader import parse_document
    return parse_document(p, tmp_project)


def _llm(system, user):
    if "Extract Additive Manufacturing entities" in user:
        raise AssertionError("the LLM entity prompt must not run when GLiNER is the entity backend")
    if "Extract relations" in user:
        return json.dumps({"relations": []})
    return json.dumps({"findings": []})


def test_gliner_backend_replaces_llm_entity_prompt(tmp_project):
    from src.entity_extraction.gliner_extractor import GlinerEntityExtractor
    from src.pipeline import ExtractionPipeline

    doc = _doc(tmp_project, "Printdur HSA was deposited at an interlayer temperature of 150 °C. "
                            "The parts were shown at fiera milano. A laser power of 200 W was used.")
    tmp_project.extraction.entity_backend = "gliner"
    probe = ExtractionPipeline(tmp_project, mode="rule")
    # inject a fake model (no Hugging Face download in tests)
    fake = GlinerEntityExtractor(FakeGliner(), probe.normalizer, probe.rule_entities)
    pipe = ExtractionPipeline(tmp_project, llm=FakeLLM(_llm), mode="hybrid", gliner=fake)
    fake.normalizer, fake.rule_extractor = pipe.normalizer, pipe.rule_entities
    assert pipe.entity_backend == "gliner"
    res = pipe.extract(doc)

    by_text = {m.text: m for m in res.entities}
    assert by_text["Printdur HSA"].method == "gliner" and by_text["Printdur HSA"].entity_type == "Material"
    assert "fiera milano" not in by_text                       # junk filtered
    lp = by_text["laser power"]                                # lexicon term found by both
    assert lp.method == "rule+gliner" and lp.quantity and lp.quantity.normalized_value == 200
    assert res.provenance.entity_backend == "gliner"
    assert res.stats.gliner_mentions >= 1 and "interlayer temperature" in by_text
    assert "entities" not in res.stats.llm_by_stage


def test_gliner_mode_needs_no_llm(tmp_project):
    from src.entity_extraction.gliner_extractor import GlinerEntityExtractor
    from src.pipeline import ExtractionPipeline

    doc = _doc(tmp_project, "Printdur HSA specimens were produced.")
    probe = ExtractionPipeline(tmp_project, mode="rule")
    fake = GlinerEntityExtractor(FakeGliner(), probe.normalizer, probe.rule_entities)
    pipe = ExtractionPipeline(tmp_project, mode="gliner", gliner=fake)
    res = pipe.extract(doc)
    assert pipe.llm is None and res.stats.llm_calls == 0
    assert any(m.text == "Printdur HSA" for m in res.entities)


def test_auto_backend_falls_back_to_llm_without_gliner(tmp_project, monkeypatch):
    from src.entity_extraction import gliner_extractor
    from src.pipeline import ExtractionPipeline

    def boom(*a, **k):
        raise ImportError("No module named 'gliner'")
    monkeypatch.setattr(gliner_extractor.GlinerEntityExtractor, "load", classmethod(lambda cls, *a, **k: boom()))
    tmp_project.extraction.entity_backend = "auto"
    pipe = ExtractionPipeline(tmp_project, llm=FakeLLM(['{"entities": []}']), mode="hybrid")
    assert pipe.entity_backend == "llm"
    tmp_project.extraction.entity_backend = "gliner"
    with pytest.raises(RuntimeError, match="pip install gliner"):
        ExtractionPipeline(tmp_project, llm=FakeLLM(['{}']), mode="hybrid")


@pytest.mark.parametrize("term,candidate,ok", [
    ("mechanical properties", "hardness", False),      # wrong LLM mappings seen on a real paper
    ("titanium", "Ti-6Al-4V", False),
    ("This defect", "porosity", False),
    ("treatment at 800 °C", "heat treatment", True),
    ("laser powers", "laser power", True),
])
def test_llm_normalisation_candidates_need_matching_head_noun(term, candidate, ok):
    from src.normalization.entity_normalizer import head_compatible
    assert head_compatible(term, candidate) is ok


def test_gliner_strips_determiners_and_skips_new_entities_in_introduction(tmp_project):
    from src.entity_extraction.gliner_extractor import GlinerEntityExtractor
    from src.ingestion.loader import parse_document
    from src.pipeline import ExtractionPipeline

    class DetGliner(FakeGliner):
        PHRASES = {"The Printdur HSA": "material or alloy", "architected materials": "material or alloy"}

    p = tmp_project.path("papers_dir") / "det.md"
    p.write_text("# T\n\n## 1. Introduction\n\nArchitected materials are popular.\n\n"
                 "## 2. Materials and methods\n\nThe Printdur HSA powder was used.\n", encoding="utf-8")
    doc = parse_document(p, tmp_project)
    probe = ExtractionPipeline(tmp_project, mode="rule")
    fake = GlinerEntityExtractor(DetGliner(), probe.normalizer, probe.rule_entities, skip_new_in={"introduction"})
    res = ExtractionPipeline(tmp_project, mode="gliner", gliner=fake).extract(doc)
    texts = {m.text for m in res.entities}
    assert "Printdur HSA" in texts and "The Printdur HSA" not in texts
    assert not any("rchitected" in t for t in texts)
    # a new entity mentioned once is not turned into a paper-level STUDIES claim
    assert not any(r.relation == "STUDIES" and "printdur" in r.object.id.lower() for r in res.relations)
