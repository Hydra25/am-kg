"""Generic process mentions -> the paper's main process.

Real papers say "the printing process", "fabrication", "the metal printer" or
"the deposition process" when they mean the paper's own LPBF process. Without
help every such phrase became its own new Process entity, and rule relations
then attached parameters to it ("deposition process -HAS_PARAMETER-> laser power").
On the 10-paper LPBF silver standard these were a large share of relation
false positives.

Two rules:
  1. A generic AM-process phrase is mapped to the paper's main process: the
     lexicon process (LPBF, PBF, EBM ...) the paper mentions most outside the
     introduction, if it clearly dominates.
  2. Other Process mentions that are not lexicon processes (e.g. "remelting",
     "Marangoni flow", "powder compaction") stay as entities but are not used
     as the subject of rule-based structural relations (see
     `RuleRelationExtractor(process_ok=...)`).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

from .text import normalize_chars, singularize

# words that may appear in a generic phrase for "the AM process of this paper"
_GENERIC_WORDS = {
    "the", "this", "our", "a", "an", "whole", "entire", "overall",
    "additive", "additively", "am", "3d", "sig", "metal", "laser", "powder", "based", "bed",
    "layer", "by", "multi", "build", "building", "printing", "print", "manufacturing",
    "fabrication", "deposition", "fusion", "specimen", "sample", "part",
    "process", "printer", "machine", "system",
}
# ... and the phrase must end in one of these heads
_GENERIC_HEADS = {"process", "printing", "print", "fabrication", "manufacturing", "deposition", "printer",
                  "machine", "build", "fusion"}


def is_generic_process(text: str) -> bool:
    """'printing process', 'fabrication', 'metal printer', 'layer-by-layer manufacturing process' -> True;
    'remelting', 'Marangoni flow', 'laser welding', 'powder compaction' -> False."""
    toks = [singularize(t) for t in re.findall(r"[a-z0-9]+", normalize_chars(text).lower())]
    return bool(toks) and toks[-1] in _GENERIC_HEADS and all(t in _GENERIC_WORDS for t in toks)


def main_process(analyses: dict[str, Any], lexicon: Any, min_share: float = 0.6) -> Optional[tuple[str, str]]:
    """(canonical_id, canonical_name) of the lexicon process that dominates the paper, or None."""
    counts: Counter[str] = Counter()
    names: dict[str, str] = {}
    for an in analyses.values():
        if an.sentence.section_type == "introduction":
            continue
        for m in an.mentions:
            if m.entity_type == "Process" and m.canonical_id in lexicon.entities:
                counts[m.canonical_id] += 1
                names[m.canonical_id] = m.canonical_name
    if not counts:
        return None
    cid, n = counts.most_common(1)[0]
    if n < 2 or n / sum(counts.values()) < min_share:
        return None
    return cid, names[cid]


def resolve_generic_processes(analyses: dict[str, Any], lexicon: Any) -> dict[str, int]:
    """Re-point generic Process mentions to the paper's main process (in place)."""
    target = main_process(analyses, lexicon)
    stats = {"remapped": 0}
    if target is None:
        return stats
    cid, name = target
    for an in analyses.values():
        for m in an.mentions:
            if m.entity_type == "Process" and m.canonical_id not in lexicon.entities and is_generic_process(m.text):
                m.canonical_id, m.canonical_name = cid, name
                m.normalization_method = "paper_main_process"
                m.normalization_confidence = 0.8
                stats["remapped"] += 1
    return stats
