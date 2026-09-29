"""Evaluation metrics against manually annotated papers.

Gold format: data/annotations/<source-file-stem>.json (see
docs/EVALUATION.md). Matching is on canonical names (case/spacing-insensitive).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

from ..evidence.verify import EvidenceIndex
from ..ingestion.loader import load_parsed
from ..knowledge_graph.kb import KnowledgeBase
from ..normalization.entity_normalizer import EntityNormalizer
from ..normalization.text import match_key, token_overlap
from ..ontology.models import Relation

log = logging.getLogger(__name__)
STRUCT_AND_EFFECT = {"PROCESSES", "HAS_PARAMETER", "HAS_CONDITION", "FABRICATES", "CHARACTERIZES",
                     "PRODUCES", "AFFECTS", "INFLUENCES", "CAUSES", "IMPROVES", "CHANGES"}


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    false_positives: list = field(default_factory=list)
    false_negatives: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, gold: set, pred: set) -> None:
        self.tp += len(gold & pred)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)
        self.false_positives += sorted(map(str, pred - gold))
        self.false_negatives += sorted(map(str, gold - pred))

    def summary(self) -> dict:
        return {"precision": round(self.precision, 3), "recall": round(self.recall, 3), "f1": round(self.f1, 3),
                "tp": self.tp, "fp": self.fp, "fn": self.fn}


def _k(s: str) -> str:
    return match_key(s)


def _value_match(gold: dict, q) -> bool:
    if q is None or (q.normalized_unit or "") != gold.get("normalized_unit"):
        return False

    def close(a: Optional[float], b: Optional[float]) -> bool:
        return a is not None and b is not None and abs(a - b) <= 0.01 * max(abs(a), abs(b), 1e-9)
    if "normalized_value" in gold:
        return close(gold["normalized_value"], q.normalized_value)
    return close(gold.get("normalized_min"), q.normalized_min) and close(gold.get("normalized_max"), q.normalized_max)


def evaluate(kb: KnowledgeBase, gold_dir: Path, processed_dir: Path, normalizer: EntityNormalizer,
             statuses: Iterable[str] = ("accepted", "review"), include_derived: bool = False) -> dict:
    statuses = set(statuses)
    ent, rel_strict, rel_lenient, val = PRF(), PRF(), PRF(), PRF()
    norm_total = norm_correct = 0
    norm_errors: list[dict] = []
    ev_total = ev_correct = 0
    cite_total = cite_ok = 0
    per_paper: dict[str, dict] = {}
    by_stem = {Path(p.source_file or "").stem: pid for pid, p in kb.papers.items()}
    gold_files = sorted(Path(gold_dir).glob("*.json"))
    if not gold_files:
        raise FileNotFoundError(f"No gold annotation files in {gold_dir}")

    for gf in gold_files:
        gold = json.loads(gf.read_text(encoding="utf-8"))
        stem = gold.get("source_file") or gf.stem
        pid = by_stem.get(stem)
        if pid is None:
            log.warning("Gold file %s has no matching extracted paper (source_file=%s)", gf.name, stem)
            continue
        claims = [c for c in kb.claims if c.evidence and c.evidence.paper_id == pid and c.status.value in statuses]
        if not include_derived:
            claims = [c for c in claims if c.knowledge_type.value != "DERIVED"]
        # --- entities (document level, canonical + type) --------------------
        g_ent = {(e["type"], _k(e["canonical"])) for e in gold["entities"]}
        p_ent = {(kb.entities[cid].entity_type, _k(kb.entities[cid].canonical_name))
                 for cid in kb.mention_counts.get(pid, {}) if cid in kb.entities}
        e_prf = PRF()
        e_prf.add(g_ent, p_ent)
        ent.add(g_ent, p_ent)
        # --- relations ------------------------------------------------------------
        g_rel = {(_k(r["subject"]), r["relation"], _k(r["object"])) for r in gold["relations"]}
        p_claims = [c for c in claims if c.relation in STRUCT_AND_EFFECT]
        p_rel = {(_k(c.subject.name), c.relation, _k(c.object.name)) for c in p_claims}
        r_prf = PRF()
        r_prf.add(g_rel, p_rel)
        rel_strict.add(g_rel, p_rel)
        rel_lenient.add({frozenset((s, o)) for s, _, o in g_rel}, {frozenset((s, o)) for s, _, o in p_rel})
        # --- values ------------------------------------------------------------------
        preds = [c for c in claims if c.relation == "HAS_VALUE"]
        used: set[int] = set()
        tp = 0
        missed = []
        for g in gold.get("values", []):
            hit = next((i for i, c in enumerate(preds) if i not in used and _k(c.subject.name) == _k(g["entity"])
                        and _value_match(g, c.object.quantity)), None)
            if hit is None:
                missed.append(f"{g['entity']}={g.get('normalized_value', (g.get('normalized_min'), g.get('normalized_max')))}")
            else:
                used.add(hit)
                tp += 1
        # predicted values not in gold count as FP only if their (entity,value) is absent (duplicates allowed)
        g_keys = [(_k(g["entity"]), g) for g in gold.get("values", [])]
        fp_list = [f"{c.subject.name}={c.object.name}" for i, c in enumerate(preds) if i not in used and not any(
            ek == _k(c.subject.name) and _value_match(g, c.object.quantity) for ek, g in g_keys)]
        val.tp += tp
        val.fn += len(missed)
        val.fp += len(fp_list)
        val.false_negatives += missed
        val.false_positives += fp_list
        # --- normalisation accuracy -------------------------------------------------
        for n in gold.get("normalization", []):
            res = normalizer.normalize(n["surface"], n["type"])
            ok = _k(res.canonical_name) == _k(n["canonical"]) and res.entity_type == n["type"]
            norm_total += 1
            norm_correct += ok
            if not ok:
                norm_errors.append({"surface": n["surface"], "expected": n["canonical"], "got": res.canonical_name,
                                    "method": res.method})
        # --- evidence correctness for matched relations ------------------------------------
        # a matched gold relation counts as correctly evidenced if at least one supporting
        # claim cites the annotated evidence sentence (token overlap >= 0.6)
        gold_ev = {(_k(r["subject"]), r["relation"], _k(r["object"])): r.get("evidence", "") for r in gold["relations"]}
        support: dict[tuple, list[Relation]] = defaultdict(list)
        for c in p_claims:
            support[(_k(c.subject.name), c.relation, _k(c.object.name))].append(c)
        for key, gev in gold_ev.items():
            if key in support and gev:
                ev_total += 1
                if any(c.evidence and token_overlap(gev, c.evidence.text) >= 0.6 for c in support[key]):
                    ev_correct += 1
        # citation support: does each cited evidence text mention both endpoints?
        for c in claims:
            if c.relation in STRUCT_AND_EFFECT | {"HAS_VALUE"} and c.evidence:
                cite_total += 1
                t = c.evidence.text.lower()
                subj_ok = (c.subject.surface or c.subject.name).lower() in t
                obj_ok = (c.object.surface or c.object.name).lower() in t
                cite_ok += int(subj_ok and obj_ok)
        per_paper[stem] = {"paper_id": pid, "entities": e_prf.summary(), "relations_strict": r_prf.summary()}

    return {
        "entities": {**ent.summary(), "false_positives": ent.false_positives, "false_negatives": ent.false_negatives},
        "relations_strict": {**rel_strict.summary(), "false_positives": rel_strict.false_positives,
                             "false_negatives": rel_strict.false_negatives},
        "relations_lenient_pair": rel_lenient.summary(),
        "values": {**val.summary(), "false_positives": val.false_positives, "false_negatives": val.false_negatives},
        "normalization": {"accuracy": round(norm_correct / norm_total, 3) if norm_total else None,
                          "correct": norm_correct, "total": norm_total, "errors": norm_errors},
        "evidence": {"accuracy": round(ev_correct / ev_total, 3) if ev_total else None,
                     "correct": ev_correct, "total": ev_total,
                     "definition": "matched gold relation has a supporting claim citing the annotated evidence sentence (>=60% token overlap)"},
        "citation_support": {"rate": round(cite_ok / cite_total, 3) if cite_total else None, "supported": cite_ok,
                             "total": cite_total,
                             "definition": "share of predicted claims whose evidence text contains both endpoint surface forms"},
        "per_paper": per_paper,
        "settings": {"statuses": sorted(statuses), "include_derived": include_derived},
    }


def graph_quality(kb: KnowledgeBase, processed_dir: Path, statuses: Iterable[str] = ("accepted", "review")) -> dict:
    """Graph-level quality indicators (no gold standard needed)."""
    st = set(statuses)
    active = [c for c in kb.claims if c.status.value in st]
    # unsupported = evidence missing or not verifiable in the parsed source document
    indexes: dict[str, EvidenceIndex] = {}
    for p in Path(processed_dir).glob("paper_*.json"):
        doc = load_parsed(p)
        indexes[doc.paper.paper_id] = EvidenceIndex(doc)
    unsupported = 0
    for c in active:
        idx = indexes.get(c.evidence.paper_id) if c.evidence else None
        if idx is None or not idx.verify(c.evidence)[0]:
            unsupported += 1
    # duplicate entity candidates: same type, near-identical keys, different ids
    by_type: dict[str, list] = defaultdict(list)
    for e in kb.entities.values():
        by_type[e.entity_type].append(e)
    import difflib
    dups = []
    for ents in by_type.values():
        for a, b in combinations(ents, 2):
            ka, kb_ = _k(a.canonical_name), _k(b.canonical_name)
            alias_hit = ka in {_k(x) for x in b.aliases} or kb_ in {_k(x) for x in a.aliases}
            if alias_hit or difflib.SequenceMatcher(None, ka, kb_).ratio() >= 0.9:
                dups.append((a.id, b.id))
    edges = {(c.subject.id, c.relation, c.object.id) for c in active}
    nodes = {c.subject.id for c in active} | {c.object.id for c in active}
    return {
        "nodes": len(nodes), "entity_nodes": sum(1 for n in nodes if n in kb.entities),
        "relationships_aggregated": len(edges), "claims": len(active),
        "duplicate_entity_pairs": dups, "duplicate_entity_rate": round(len(dups) / max(1, len(kb.entities)), 4),
        "unsupported_claims": unsupported, "unsupported_relation_rate": round(unsupported / max(1, len(active)), 4),
        "new_entities_pending_review": len(kb.new_entities()),
        "claims_by_status": {s: sum(c.status.value == s for c in kb.claims) for s in ("accepted", "review", "rejected")},
    }


def to_markdown(report: dict, quality: dict) -> str:
    lines = ["# AM-KG evaluation report", ""]
    lines.append("| Task | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("entities", "relations_strict", "relations_lenient_pair", "values"):
        r = report[name]
        lines.append(f"| {name} | {r['precision']} | {r['recall']} | {r['f1']} | {r['tp']} | {r['fp']} | {r['fn']} |")
    n = report["normalization"]
    e = report["evidence"]
    lines += ["", f"- Normalisation accuracy: **{n['accuracy']}** ({n['correct']}/{n['total']})",
              f"- Evidence accuracy (matched relations): **{e['accuracy']}** ({e['correct']}/{e['total']})",
              f"- Citation support rate: **{report['citation_support']['rate']}** "
              f"({report['citation_support']['supported']}/{report['citation_support']['total']})", "",
              "## Graph quality", ""]
    for k, v in quality.items():
        if k != "duplicate_entity_pairs":
            lines.append(f"- {k}: {v}")
    lines += ["", "## Errors (for error analysis)", "", "### Relation false negatives"]
    lines += [f"- {x}" for x in report["relations_strict"]["false_negatives"]] or ["- none"]
    lines += ["", "### Relation false positives"]
    lines += [f"- {x}" for x in report["relations_strict"]["false_positives"]] or ["- none"]
    lines += ["", "### Normalisation errors"]
    lines += [f"- {x['surface']!r}: expected {x['expected']!r}, got {x['got']!r} ({x['method']})" for x in n["errors"]] or ["- none"]
    lines += ["", f"_Settings: {report['settings']}_"]
    return "\n".join(lines)
