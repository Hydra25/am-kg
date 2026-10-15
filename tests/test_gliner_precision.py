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
