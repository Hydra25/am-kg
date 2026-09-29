"""The worked example from the project brief must be extracted exactly."""
from src.entity_extraction.rule_based import RuleEntityExtractor
from src.ontology.models import Sentence
from src.relation_extraction.rule_based import RuleRelationExtractor

BRIEF = ("Ti-6Al-4V specimens were manufactured using laser powder bed fusion with a laser power of 200 W, "
         "scanning speed of 800 mm/s and layer thickness of 30 μm.")
BRIEF2 = "The resulting tensile strength was 950 MPa."


def _sent(i, text, section="materials_methods"):
    return Sentence(sentence_id=f"p_s{i:04d}", text=text, page=5, section_id="sec1", section_type=section,
                    section_title="Experimental Methods", index=i)


def _extract(lexicon, normalizer, texts, section="results"):
    ex = RuleEntityExtractor(lexicon, normalizer)
    rx = RuleRelationExtractor("p")
    prev, rels = [], []
    for i, t in enumerate(texts):
        an = ex.analyze(_sent(i, t, section), "p")
        rels += rx.effect_relations(an) + rx.structural_relations(an, prev, [])
        prev.append(an)
    return prev, rels


def triples(rels):
    return {(r.subject.name, r.relation, r.object.name) for r in rels}


def test_brief_example(lexicon, normalizer):
    ans, rels = _extract(lexicon, normalizer, [BRIEF, BRIEF2], "materials_methods")
    ents = {(m.entity_type, m.canonical_name) for m in ans[0].mentions + ans[1].mentions}
    assert {("Material", "Ti-6Al-4V"), ("Process", "LPBF"), ("Parameter", "laser power"),
            ("Parameter", "scanning speed"), ("Parameter", "layer thickness"), ("Property", "tensile strength")} <= ents
    t = triples(rels)
    assert ("LPBF", "PROCESSES", "Ti-6Al-4V") in t            # == Ti-6Al-4V PROCESSED_BY LPBF
    for p, v in [("laser power", "200 W"), ("scanning speed", "800 mm/s"), ("layer thickness", "30 μm")]:
        assert ("LPBF", "HAS_PARAMETER", p) in t
        assert (p, "HAS_VALUE", v) in t
    assert ("tensile strength", "HAS_VALUE", "950 MPa") in t
    assert ("LPBF", "PRODUCES", "tensile strength") in t
    # every relation carries evidence
    for r in rels:
        assert r.evidence and r.evidence.paper_id == "p" and r.evidence.page == 5
        assert r.evidence.section == "Experimental Methods" and r.evidence.text
    produces = [r for r in rels if r.relation == "PRODUCES"][0]
    assert produces.evidence.context_sentence_ids == ["p_s0000"]     # uses the previous sentence, says so
    value = [r for r in rels if r.object.name == "30 μm"][0].object.quantity
    assert value.normalized_value == 30 and value.normalized_unit == "μm" and value.raw == "30 μm"


def test_causal_direction_negation_hedging(lexicon, normalizer):
    _, rels = _extract(lexicon, normalizer, [
        "Increasing the laser power from 150 W to 250 W reduced the porosity from 1.1 % to 0.3 %.",
        "The build orientation did not significantly affect the tensile strength.",
        "Keyhole porosity may be promoted by excessive laser power.",
        "HIP closed the internal pores and improved the elongation to 14 %.",
    ])
    by = {(r.subject.name, r.relation, r.object.name): r for r in rels}
    r = by[("laser power", "INFLUENCES", "porosity")]
    assert r.attributes["trend"] == "negative" and r.knowledge_type.value == "EXPLICIT"
    r = by[("build orientation", "AFFECTS", "tensile strength")]
    assert r.attributes["negated"] is True
    r = by[("laser power", "INFLUENCES", "keyhole porosity")]            # passive voice, hedged
    assert r.attributes["hedged"] is True and r.confidence < 0.85
    assert ("HIP", "IMPROVES", "elongation") in by                        # coordinated predicate
    assert ("porosity", "AFFECTS", "elongation") not in by


def test_no_relation_without_textual_support(lexicon, normalizer):
    _, rels = _extract(lexicon, normalizer, ["Porosity and laser power are discussed in many reviews."])
    assert not [r for r in rels if r.relation in ("INFLUENCES", "AFFECTS")]


def test_characterization_pairing(lexicon, normalizer):
    _, rels = _extract(lexicon, normalizer, ["Porosity was quantified with micro-CT and residual stress was measured by XRD."])
    t = triples(rels)
    assert ("CT", "CHARACTERIZES", "porosity") in t and ("XRD", "CHARACTERIZES", "residual stress") in t
    assert ("CT", "CHARACTERIZES", "residual stress") not in t


def test_background_claims_are_downgraded(lexicon, normalizer):
    _, rels = _extract(lexicon, normalizer, ["Earlier studies reported that the volumetric energy density influences porosity [3]."],
                       section="introduction")
    r = [r for r in rels if r.relation == "INFLUENCES"][0]
    assert r.attributes["claim_origin"] == "background" and r.confidence < 0.85


def test_dimension_check_blocks_wrong_values(lexicon, normalizer):
    ans, rels = _extract(lexicon, normalizer, ["The laser power was 200 W and the porosity was 0.5 %."])
    t = triples(rels)
    assert ("laser power", "HAS_VALUE", "200 W") in t and ("porosity", "HAS_VALUE", "0.5 %") in t
    assert ("laser power", "HAS_VALUE", "0.5 %") not in t
