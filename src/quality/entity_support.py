"""Keep only GLiNER entities that are useful for the knowledge graph ("kaam ki entities").

GLiNER proposes many new (non-lexicon) spans. Most of them appear once and take part in nothing.
A new GLiNER entity is kept only if it has *support* in the paper:

* it is an endpoint of a relation that QC did not reject, or
* a value is attached to it ("laser power of 200 W"), or
* the paper mentions it at least `min_mentions` times.

Lexicon entities and entities found by rules or the LLM are never touched.
"""
from __future__ import annotations

from collections import Counter
from typing import Container, Iterable

from ..ontology.models import ClaimStatus, EntityMention, Relation


def is_unsupported_candidate(m: EntityMention, lexicon_ids: Container[str]) -> bool:
    return m.method == "gliner" and bool(m.canonical_id) and m.canonical_id not in lexicon_ids


def support_filter(mentions: list[EntityMention], relations: Iterable[Relation], lexicon_ids: Container[str],
                   min_mentions: int = 2) -> tuple[list[EntityMention], list[dict]]:
    """-> (kept mentions, dropped records for the audit file)."""
    endpoints: set[str] = set()
    for r in relations:
        if r.status != ClaimStatus.REJECTED:
            endpoints.add(r.subject.id)
            endpoints.add(r.object.id)
    counts = Counter(m.canonical_id for m in mentions if m.canonical_id)
    with_value = {m.canonical_id for m in mentions if m.canonical_id and m.quantity is not None}

    kept: list[EntityMention] = []
    dropped: list[dict] = []
    for m in mentions:
        if (not is_unsupported_candidate(m, lexicon_ids) or m.canonical_id in endpoints
                or m.canonical_id in with_value or counts[m.canonical_id] >= min_mentions):
            kept.append(m)
        else:
            dropped.append({"stage": "gliner_support", "text": m.text, "type": m.entity_type,
                            "canonical_id": m.canonical_id, "sentence_id": m.sentence_id,
                            "reason": "no_support"})
    return kept, dropped
