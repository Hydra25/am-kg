from src.config import QualityConfig
from src.ontology.models import EntityRef, Evidence, Relation
from src.ontology.schema import KnowledgeType
from src.quality.validator import QualityController


def rel(conf=0.9, evidence=True, subj=("Parameter:laser power", "laser power", "Parameter"),
        obj=("Defect:porosity", "porosity", "Defect"), r="INFLUENCES", norm=1.0, **attrs):
    return Relation(claim_id="c", subject=EntityRef(id=subj[0], name=subj[1], type=subj[2], normalization_confidence=norm),
                    relation=r, object=EntityRef(id=obj[0], name=obj[1], type=obj[2]),
                    evidence=Evidence(paper_id="p", text="Laser power affected porosity.") if evidence else None,
                    confidence=conf, knowledge_type=KnowledgeType.EXPLICIT, attributes=attrs)


def qc(**kw):
    return QualityController(QualityConfig(**kw))


def test_thresholds():
    assert qc().assess(rel(0.9), None).status.value == "accepted"
    assert qc().assess(rel(0.7), None).status.value == "review"
    assert qc().assess(rel(0.4), None).status.value == "rejected"
    assert qc(accept_threshold=0.95).assess(rel(0.9), None).status.value == "review"   # configurable


def test_hard_rules():
    r = qc().assess(rel(evidence=False), None)
    assert r.status.value == "rejected" and "missing_evidence" in r.qc_reasons
    r = qc().assess(rel(subj=("Material:x", "x", "Material")), None)
    assert r.status.value == "rejected" and any(q.startswith("invalid_signature") for q in r.qc_reasons)
    r = qc().assess(rel(subj=("", "", "Parameter")), None)
    assert "missing_entity" in r.qc_reasons


def test_soft_rules_cap_at_review():
    assert qc().assess(rel(0.95, norm=0.5), None).status.value == "review"          # new entity
    assert qc().assess(rel(0.95, claim_origin="background"), None).status.value == "review"
