"""GLiNER precision controls: new junk rules, per-type thresholds, long-sentence windows, support filter."""
import pytest

from src.entity_extraction.gliner_extractor import LABEL_SETS, MAX_WORDS, GlinerEntityExtractor
from src.ontology.models import ClaimStatus, EntityMention, EntityRef, Quantity, Relation
from src.quality.entity_support import support_filter
from src.quality.junk import junk_reason


# ------------------------------------------------------------------ junk rules (from the DEV false positives)
@pytest.mark.parametrize("text,etype,reason", [
    ("length of 26 mm", "Geometry", "contains_value"),
    ("p-value", "Property", "statistics_term"),
    ("ANOVA", "CharacterizationMethod", "statistics_term"),
    ("FC", "Microstructure", "unknown_abbreviation"),
    ("qi", "Parameter", "unknown_abbreviation"),
    ("different process parameter", "Parameter", "generic_phrase"),
    ("various operating conditions", "ExperimentalCondition", "generic_phrase"),
])
def test_new_junk_rules(text, etype, reason):
    assert junk_reason(text, etype, in_lexicon=False) == reason


@pytest.mark.parametrize("text,etype", [
    ("pore", "Defect"), ("lath", "Microstructure"), ("Ti", "Material"),
    ("lack of fusion", "Defect"), ("melt pool depth", "Property"),
])
def test_real_short_terms_still_pass(text, etype):
    assert junk_reason(text, etype, in_lexicon=False) is None


# ------------------------------------------------------------------ extractor mechanics (no model download)
class RecordingModel:
    name = "rec"

    def __init__(self, score=0.6):
        self.texts, self.thresholds, self.score = [], [], score

    def predict_entities(self, text, labels, threshold=0.5):
        self.texts.append(text)
        self.thresholds.append(threshold)
        i = text.find("keyhole")
        return [{"start": i, "end": i + 7, "label": labels[0], "score": self.score}] if i >= 0 else []


def _extractor(model, **kw):
    return GlinerEntityExtractor(model, normalizer=None, rule_extractor=None, **kw)


def test_type_thresholds_and_floor():
    g = _extractor(RecordingModel(), threshold=0.5, type_thresholds={"Property": 0.7, "Defect": 0.4})
    assert g.threshold_for("Property") == 0.7 and g.threshold_for("Material") == 0.5
    assert g._floor() == 0.4
    g.predict_floor = 0.3
    assert g._floor() == 0.3


def test_long_sentences_are_windowed_and_offsets_mapped():
    model = RecordingModel()
    g = _extractor(model)
    words = ["w"] * (MAX_WORDS + 60) + ["keyhole"]
    text = " ".join(words)
    preds = g._predict([text])[0]
    assert len(model.texts) >= 2 and all(len(t.split()) <= MAX_WORDS for t in model.texts)
    assert any(text[p["start"]:p["end"]] == "keyhole" for p in preds)


def test_predictions_are_cached():
    model = RecordingModel()
    g = _extractor(model)
    g._predict(["a keyhole pore"])
    g._predict(["a keyhole pore"])
    assert len(model.texts) == 1
    g.labels = dict(LABEL_SETS["lpbf"])          # new label set -> new prediction
    g._predict(["a keyhole pore"])
    assert len(model.texts) == 2


# ------------------------------------------------------------------ support filter
def _m(cid, method="gliner", q=None, sid="s1"):
    return EntityMention(mention_id=f"{cid}-{sid}-{method}", text=cid.split(":")[1], entity_type=cid.split(":")[0],
                         canonical_id=cid, sentence_id=sid, method=method, quantity=q)


def _rel(s, o, status=ClaimStatus.ACCEPTED):
    return Relation(claim_id=f"{s}-{o}", subject=EntityRef(id=s, name=s, type="x"), relation="AFFECTS",
                    object=EntityRef(id=o, name=o, type="x"), status=status)


def test_support_filter_keeps_only_useful_new_entities():
    lex = {"Defect:porosity"}
    ms = [
        _m("Defect:porosity"),                                   # lexicon -> kept
        _m("Defect:spatter particles"),                          # relation endpoint -> kept
        _m("Parameter:spot size", q=Quantity(raw="80 μm")),      # has a value -> kept
        _m("Microstructure:colony", sid="s1"), _m("Microstructure:colony", sid="s2"),   # repeated -> kept
        _m("Property:utility degree"),                           # once, no relation -> dropped
        _m("Defect:crack tip"),                                  # only in a rejected relation -> dropped
        _m("Property:odd thing", method="rule"),                 # not a GLiNER mention -> kept
    ]
    rels = [_rel("Defect:spatter particles", "Defect:porosity"),
            _rel("Defect:crack tip", "Defect:porosity", ClaimStatus.REJECTED)]
    kept, dropped = support_filter(ms, rels, lex, min_mentions=2)
    names = {m.canonical_id for m in kept}
    assert names == {"Defect:porosity", "Defect:spatter particles", "Parameter:spot size",
                     "Microstructure:colony", "Property:odd thing"}
    assert {d["canonical_id"] for d in dropped} == {"Property:utility degree", "Defect:crack tip"}
    assert all(d["reason"] == "no_support" for d in dropped)


def test_nuner_gets_lowercase_labels_and_results_map_back():
    class LowerOnly:
        name = "nuner"

        def __init__(self):
            self.labels = None

        def predict_entities(self, text, labels, threshold=0.5):
            self.labels = labels
            i = text.find("keyhole")
            return [{"start": i, "end": i + 7, "label": labels[0], "score": 0.9}]

    model = LowerOnly()
    g = GlinerEntityExtractor(model, normalizer=None, rule_extractor=None, labels=dict(LABEL_SETS["lpbf"]),
                              model_name="numind/NuNER_Zero")
    preds = g._predict(["a keyhole pore"])[0]
    assert all(lab == lab.lower() for lab in model.labels)
    assert preds[0]["label"] in g.labels          # mapped back to the original label -> entity type works


def test_neighbour_context_keeps_only_spans_of_the_target_sentence():
    from types import SimpleNamespace as NS

    model = RecordingModel()
    g = _extractor(model)
    g.context = "neighbours"
    sents = [NS(sentence_id="a", text="A keyhole formed.", section_id="s1", source_type="text"),
             NS(sentence_id="b", text="Porosity rose.", section_id="s1", source_type="text"),
             NS(sentence_id="c", text="Another keyhole here.", section_id="s1", source_type="text")]
    doc = NS(sentences=sents)
    preds = g._predict_with_neighbours(doc, sents)
    # the model saw "previous + sentence + next" for the middle sentence
    assert any("A keyhole formed. Porosity rose. Another keyhole here." == t for t in model.texts)
    assert preds[1] == []                                  # 'keyhole' in a neighbour is not given to sentence b
    assert sents[0].text[preds[0][0]["start"]:preds[0][0]["end"]] == "keyhole"
    assert sents[2].text[preds[2][0]["start"]:preds[2][0]["end"]] == "keyhole"


def test_lpbf_multi_labels_cover_all_types():
    assert set(LABEL_SETS["lpbf_multi"].values()) == set(LABEL_SETS["lpbf"].values())


# ------------------------------------------------------------------ way 2: LLM checker
def test_verifier_keeps_drops_and_retypes():
    import json as _json

    from src.entity_extraction.gliner_verifier import Candidate, GlinerVerifier
    from src.llm.client import FakeLLM

    answer = _json.dumps({"decisions": [{"id": 1, "keep": True, "type": "Defect"},
                                        {"id": 2, "keep": False, "type": "Property"},
                                        {"id": 3, "keep": "yes", "type": "Nonsense"}]})
    v = GlinerVerifier(FakeLLM([answer]))
    out = v.verify([Candidate("Spatter particles raised porosity.", "spatter particles", "Microstructure"),
                    Candidate("The utility degree was 0.8.", "utility degree", "Property"),
                    Candidate("Dimples were seen.", "dimples", "Microstructure"),
                    Candidate("x", "y", "Defect")])          # id 4 missing -> kept (fail open)
    assert [x.keep for x in out] == [True, False, True, True]
    assert out[0].entity_type == "Defect"                     # corrected type
    assert out[2].entity_type == "Microstructure"             # unknown type -> proposed type kept
    assert out[3].reason == "no_answer"


def test_verifier_fails_open_on_bad_llm():
    from src.entity_extraction.gliner_verifier import Candidate, GlinerVerifier
    from src.llm.client import FakeLLM

    def boom(system, user):
        raise RuntimeError("ollama down")

    v = GlinerVerifier(FakeLLM(boom))
    out = v.verify([Candidate("s", "keyhole", "Defect")])
    assert out[0].keep and v.failures == 1


def test_make_verifier_off_by_default():
    from src.entity_extraction.gliner_verifier import make_verifier
    assert make_verifier("", None) is None


# ------------------------------------------------------------------ junk rules from the DEV error analysis
@pytest.mark.parametrize("text,etype,reason", [
    ("AB sample", "Geometry", "specimen_id"),
    ("sample P2", "Geometry", "specimen_id"),
    ("OC1 specimens", "Geometry", "specimen_id"),
    ("specimens from OC5", "Defect", "specimen_id"),
    ("strength35", "Property", "citation_artifact"),
    ("pores44", "Defect", "citation_artifact"),
    ("27 conditions", "ExperimentalCondition", "generic_phrase"),
    ("Five representative conditions", "ExperimentalCondition", "generic_phrase"),
    ("as-built samples", "Geometry", "generic_phrase"),
    ("supplied powder", "Material", "generic_phrase"),
    ("desired shape", "Geometry", "generic_phrase"),
    ("average values", "Property", "generic_phrase"),
    ("powder", "Material", "generic_word"),
    ("grains", "Microstructure", "generic_word"),
    ("Pre-testing", "PostProcessing", "generic_word"),
])
def test_error_analysis_junk(text, etype, reason):
    assert junk_reason(text, etype, in_lexicon=False) == reason


@pytest.mark.parametrize("text,etype", [
    ("surface tension", "Property"), ("Marangoni flow", "Microstructure"), ("grain boundaries", "Microstructure"),
    ("gas bubbles", "Defect"), ("improper fusion", "Defect"), ("Weak fusion defects", "Defect"),
    ("tensile test specimen", "Geometry"), ("Inconel718", "Material"), ("powder morphology", "Microstructure"),
    ("stainless-steel substrate", "Material"), ("ASTM 1045", "Material"), ("strain", "Property"),
])
def test_error_analysis_real_terms_kept(text, etype):
    assert junk_reason(text, etype, in_lexicon=False) is None
