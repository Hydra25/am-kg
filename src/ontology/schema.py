"""AM ontology: entity types, relation types and the allowed relation signatures.

This module is the single source of truth for what may enter the Knowledge
Graph. Any relation whose (subject type, relation, object type) triple is not
listed in RELATION_SIGNATURES is rejected by the quality controller.
"""
from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    MATERIAL = "Material"
    PROCESS = "Process"                      # "AMProcess" in the project brief
    PARAMETER = "Parameter"                  # process / machine parameter
    PROPERTY = "Property"                    # mechanical, thermal, surface, physical
    DEFECT = "Defect"
    MICROSTRUCTURE = "Microstructure"
    POST_PROCESSING = "PostProcessing"
    CHARACTERIZATION = "CharacterizationMethod"
    GEOMETRY = "Geometry"
    CONDITION = "ExperimentalCondition"
    # value nodes (one node per reported observation, never shared across papers)
    PARAMETER_VALUE = "ParameterValue"
    PROPERTY_VALUE = "PropertyValue"
    # document-level nodes
    PAPER = "Paper"
    FINDING = "Finding"
    CLAIM = "Claim"

    @classmethod
    def concept_types(cls) -> list["EntityType"]:
        """Types that are canonicalised and shared across papers."""
        return [
            cls.MATERIAL, cls.PROCESS, cls.PARAMETER, cls.PROPERTY, cls.DEFECT,
            cls.MICROSTRUCTURE, cls.POST_PROCESSING, cls.CHARACTERIZATION,
            cls.GEOMETRY, cls.CONDITION,
        ]

    @classmethod
    def value_types(cls) -> list["EntityType"]:
        return [cls.PARAMETER_VALUE, cls.PROPERTY_VALUE]


class RelationType(str, Enum):
    # paper-level
    STUDIES = "STUDIES"
    USES_PROCESS = "USES_PROCESS"
    MENTIONS = "MENTIONS"
    REPORTS = "REPORTS"
    SUPPORTS = "SUPPORTS"
    # structural (constructed from a single evidence sentence / table row)
    PROCESSES = "PROCESSES"
    HAS_PARAMETER = "HAS_PARAMETER"
    HAS_VALUE = "HAS_VALUE"
    HAS_CONDITION = "HAS_CONDITION"
    FABRICATES = "FABRICATES"
    CHARACTERIZES = "CHARACTERIZES"
    # effect / causal
    PRODUCES = "PRODUCES"
    AFFECTS = "AFFECTS"
    INFLUENCES = "INFLUENCES"
    CAUSES = "CAUSES"
    IMPROVES = "IMPROVES"
    CHANGES = "CHANGES"


# Relation names that are accepted as input but stored in canonical direction.
# PROCESSED_BY(Material, Process) == PROCESSES(Process, Material)
INVERSE_ALIASES: dict[str, str] = {
    "PROCESSED_BY": "PROCESSES",
    "MANUFACTURED_BY": "PROCESSES",
    "FABRICATED_BY": "PROCESSES",
    "CHARACTERIZED_BY": "CHARACTERIZES",
    "PARAMETER_OF": "HAS_PARAMETER",
}
SYNONYM_ALIASES: dict[str, str] = {
    "INFLUENCE": "INFLUENCES", "AFFECT": "AFFECTS", "CAUSE": "CAUSES",
    "IMPROVE": "IMPROVES", "CHANGE": "CHANGES", "PRODUCE": "PRODUCES",
    "HAS_PARAM": "HAS_PARAMETER", "USES": "USES_PROCESS",
}

E = EntityType
R = RelationType

RELATION_SIGNATURES: dict[RelationType, set[tuple[EntityType, EntityType]]] = {
    R.STUDIES: {(E.PAPER, E.MATERIAL)},
    R.USES_PROCESS: {(E.PAPER, E.PROCESS)},
    R.REPORTS: {(E.PAPER, E.FINDING)},
    R.SUPPORTS: {(E.PAPER, E.CLAIM)},
    R.PROCESSES: {(E.PROCESS, E.MATERIAL)},
    R.HAS_PARAMETER: {(E.PROCESS, E.PARAMETER)},
    R.HAS_CONDITION: {(E.PROCESS, E.CONDITION)},
    R.FABRICATES: {(E.PROCESS, E.GEOMETRY)},
    R.HAS_VALUE: {
        (E.PARAMETER, E.PARAMETER_VALUE),
        (E.CONDITION, E.PARAMETER_VALUE),
        (E.PROPERTY, E.PROPERTY_VALUE),
        (E.DEFECT, E.PROPERTY_VALUE),
        (E.MICROSTRUCTURE, E.PROPERTY_VALUE),
    },
    R.CHARACTERIZES: {
        (E.CHARACTERIZATION, E.MATERIAL),
        (E.CHARACTERIZATION, E.MICROSTRUCTURE),
        (E.CHARACTERIZATION, E.DEFECT),
        (E.CHARACTERIZATION, E.PROPERTY),
    },
    R.PRODUCES: {(E.PROCESS, E.PROPERTY), (E.PROCESS, E.MICROSTRUCTURE)},
    R.AFFECTS: {
        (E.PARAMETER, E.PROPERTY),
        (E.PARAMETER, E.MICROSTRUCTURE),
        (E.MICROSTRUCTURE, E.PROPERTY),
        (E.POST_PROCESSING, E.PROPERTY),
        (E.DEFECT, E.PROPERTY),
        (E.CONDITION, E.PROPERTY),
        (E.CONDITION, E.MICROSTRUCTURE),
        (E.GEOMETRY, E.PROPERTY),
    },
    R.INFLUENCES: {
        (E.PARAMETER, E.DEFECT),
        (E.CONDITION, E.DEFECT),
        (E.POST_PROCESSING, E.DEFECT),
        (E.MICROSTRUCTURE, E.DEFECT),
    },
    R.CAUSES: {(E.PROCESS, E.DEFECT), (E.DEFECT, E.DEFECT)},
    R.IMPROVES: {(E.POST_PROCESSING, E.PROPERTY)},
    R.CHANGES: {(E.POST_PROCESSING, E.MICROSTRUCTURE)},
}

EFFECT_RELATIONS = {R.AFFECTS, R.INFLUENCES, R.CAUSES, R.IMPROVES, R.CHANGES, R.PRODUCES}
STRUCTURAL_RELATIONS = {R.PROCESSES, R.HAS_PARAMETER, R.HAS_VALUE, R.HAS_CONDITION,
                        R.FABRICATES, R.CHARACTERIZES}
PAPER_RELATIONS = {R.STUDIES, R.USES_PROCESS, R.MENTIONS, R.REPORTS, R.SUPPORTS}


class KnowledgeType(str, Enum):
    """Epistemic status of a fact. These are stored separately and never mixed.

    EXPLICIT    - the evidence sentence itself states the relation with a
                  relational/causal predicate ("laser power increased porosity").
    EXTRACTED   - the extractor constructed the relation from entities that
                  co-occur in one evidence sentence / table row, without an
                  explicit predicate ("LPBF with a laser power of 200 W").
    DERIVED     - computed by the system across evidence (paper-level
                  STUDIES/USES_PROCESS, aggregated trends, gap statistics).
    INTERPRETED - LLM-generated explanation. Never written to the graph as a
                  fact; only shown in assistant answers, clearly labelled.
    """
    EXPLICIT = "EXPLICIT"
    EXTRACTED = "EXTRACTED"
    DERIVED = "DERIVED"
    INTERPRETED = "INTERPRETED"


class ClaimStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class SectionType(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    MATERIALS_METHODS = "materials_methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    RESULTS_DISCUSSION = "results_discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGEMENTS = "acknowledgements"
    OTHER = "other"
    FRONT_MATTER = "front_matter"


# Which relation to create when a causal predicate links two typed entities.
def infer_effect_relation(subj: EntityType, obj: EntityType, trigger_class: str) -> RelationType | None:
    """Map (subject type, object type, trigger class) to an ontology relation.

    trigger_class is one of: increase, decrease, improve, deteriorate, affect, cause.
    Returns None when the ontology has no valid effect relation for the pair.
    """
    if subj == E.POST_PROCESSING:
        if obj == E.PROPERTY:
            return R.IMPROVES if trigger_class == "improve" else R.AFFECTS
        if obj == E.MICROSTRUCTURE:
            return R.CHANGES
        if obj == E.DEFECT:
            return R.INFLUENCES
        return None
    if subj == E.PROCESS:
        if obj == E.DEFECT:
            return R.CAUSES
        if obj in (E.PROPERTY, E.MICROSTRUCTURE):
            return R.PRODUCES
        return None
    if subj == E.DEFECT and obj == E.DEFECT:
        return R.CAUSES
    for rel in (R.AFFECTS, R.INFLUENCES):
        if (subj, obj) in RELATION_SIGNATURES[rel]:
            return rel
    return None


def is_valid_signature(relation: str, subj: str, obj: str) -> bool:
    try:
        rel = RelationType(relation)
        return (EntityType(subj), EntityType(obj)) in RELATION_SIGNATURES.get(rel, set()) or rel == R.MENTIONS
    except ValueError:
        return False


def canonicalize_relation(relation: str, subj_type: str, obj_type: str) -> tuple[str, bool] | None:
    """Return (canonical relation name, swapped?) or None if unknown.

    Handles inverse aliases (PROCESSED_BY -> PROCESSES with swapped direction)
    and fixes direction when only the reversed signature is valid.
    """
    name = relation.strip().upper().replace(" ", "_").replace("-", "_")
    name = SYNONYM_ALIASES.get(name, name)
    if name in INVERSE_ALIASES:
        return INVERSE_ALIASES[name], True
    if name not in RelationType.__members__.values() and name not in {r.value for r in RelationType}:
        return None
    if is_valid_signature(name, subj_type, obj_type):
        return name, False
    if is_valid_signature(name, obj_type, subj_type):
        return name, True
    return name, False  # invalid signature; QC will reject it with a reason
