import json

from conftest import ROOT

from src.entity_extraction.llm_extractor import LLMEntities, LLMExtractor
from src.entity_extraction.rule_based import RuleEntityExtractor
from src.llm.client import FakeLLM
from src.llm.json_utils import call_llm_json, parse_json_lenient
from src.llm.prompts import PromptLibrary
from src.ontology.models import PaperMetadata, ParsedDocument, Sentence


def test_json_repair():
    assert parse_json_lenient('```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_json_lenient("Sure! Here it is: {'a': True, 'b': None}") == {"a": True, "b": None}
    assert parse_json_lenient('{"entities": [{"text": "LPBF"') == {"entities": [{"text": "LPBF"}]}  # truncated
    assert parse_json_lenient('{a: 1}') == {"a": 1}


def test_retry_with_feedback():
    llm = FakeLLM(["not json at all", json.dumps({"entities": [{"text": "LPBF", "type": "Process", "sentence": "S1"}]})])
    res, log = call_llm_json(llm, "sys", "user", LLMEntities, max_retries=2)
    assert res is not None and res.entities[0].text == "LPBF"
    assert log.attempts == 2 and "INVALID" in llm.prompts[1][1]


def test_gives_up_gracefully():
    res, log = call_llm_json(FakeLLM(["nope"]), "s", "u", LLMEntities, max_retries=1)
    assert res is None and log.attempts == 2


def _doc():
    s = [Sentence(sentence_id="p_s0000", text="Inconel 718 walls were deposited by WAAM with a wire feed rate of 6 m/min and an arc current of 160 A.",
                  page=2, section_id="sec", section_type="materials_methods", section_title="2. Methods", index=0),
         Sentence(sentence_id="p_s0001", text="The travel speed reduced the grain size.", page=3,
                  section_id="sec", section_type="results", section_title="3. Results", index=1)]
    return ParsedDocument(paper=PaperMetadata(paper_id="p"), parser="test", sentences=s)


def test_llm_extractor_verifies_everything(lexicon, normalizer):
    def responder(system, user):
        if "Extract Additive Manufacturing entities" in user:
            return json.dumps({"entities": [
                {"text": "WAAM", "type": "AMProcess", "sentence": "S1", "confidence": 0.9},
                {"text": "wire feed rate", "type": "Parameter", "sentence": "S1", "value": "6 m/min", "confidence": 0.9},
                {"text": "laser power", "type": "Parameter", "sentence": "S1", "value": "200 W", "confidence": 0.9},  # hallucinated
                {"text": "wire feed rate", "type": "Parameter", "sentence": "S1", "value": "160 A", "confidence": 0.9}]})
        if "Extract relations" in user:
            return json.dumps({"relations": [
                {"subject": "Inconel 718", "relation": "PROCESSED_BY", "object": "WAAM", "sentence": "S1",
                 "evidence": "Inconel 718 walls were deposited by WAAM", "confidence": 0.9},
                {"subject": "travel speed", "relation": "AFFECTS", "object": "grain size", "sentence": "S2",
                 "evidence": "The travel speed reduced the grain size", "effect": "decrease", "confidence": 0.8},
                {"subject": "laser power", "relation": "INFLUENCES", "object": "porosity", "sentence": "S2",
                 "evidence": "laser power increases porosity", "confidence": 0.9}]})       # not in text
        return json.dumps({"findings": [{"sentence": "S2", "finding_type": "trend", "confidence": 0.8}]})

    doc = _doc()
    rex = RuleEntityExtractor(lexicon, normalizer)
    analyses = {s.sentence_id: rex.analyze(s, "p") for s in doc.sentences}
    llm = LLMExtractor(FakeLLM(responder), lexicon, normalizer, PromptLibrary(ROOT / "prompts"))
    out = llm.extract(doc, analyses, [], 2000)
    reasons = [d["reason"] for d in out.dropped]
    assert "text_not_in_sentence" in reasons                       # hallucinated entity dropped
    assert any(r.startswith("dimension_mismatch") for r in reasons)   # 5 MPa is not a speed
    assert "subject_not_in_sentence" in reasons                    # hallucinated relation dropped
    t = {(r.subject.name, r.relation, r.object.name) for r in out.relations}
    assert ("WAAM", "PROCESSES", "Inconel 718") in t               # inverse relation canonicalised
    assert ("travel speed", "AFFECTS", "grain size") in t
    assert all(r.evidence.text in [s.text for s in doc.sentences] for r in out.relations)  # verbatim evidence
    assert out.findings and out.findings[0].text == doc.sentences[1].text
    wfr = [m for m in out.mentions if m.canonical_name == "wire feed rate"][0]
    assert wfr.quantity.normalized_value == 100.0 and wfr.method == "rule+llm"
