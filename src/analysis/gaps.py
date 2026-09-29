"""EXPERIMENTAL: potentially under-studied combinations in the current corpus.

Wording rule: this module never claims that something *is* a research gap. It
reports "Limited evidence was found in the current corpus", always together
with the corpus statistics and supporting papers, because a low count can
equally be caused by a small corpus, extraction errors or lexicon coverage.

Candidate types
  A  context gap      factor->response studied in >= N papers overall, but in
                      <= M papers for a given material x process context
                      (only factors reported as parameters of that process).
  B  joint-factor gap two factors each studied for a response, but rarely
                      in the same paper (e.g. preheating + scan strategy -> porosity).
  C  unlinked pair    a process parameter and a response both appear for a
                      process, but no evidence-backed effect claim links them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Iterable, Optional

from ..knowledge_graph.kb import KnowledgeBase
from ..ontology.models import Relation

FACTOR_TYPES = {"Parameter", "ExperimentalCondition", "PostProcessing"}
RESPONSE_TYPES = {"Property", "Defect", "Microstructure"}
EFFECTS = {"AFFECTS", "INFLUENCES", "IMPROVES", "CHANGES", "CAUSES"}
CAVEATS = [
    "Counts reflect only the papers currently in the corpus and what the extractor recognised.",
    "A low count may be caused by corpus size, extraction/normalisation errors or lexicon coverage.",
    "Synthetic DEMO papers are not scientific evidence; exclude them for real analyses.",
    "Always read the supporting papers before treating a candidate as a research gap.",
]


@dataclass
class GapCandidate:
    kind: str
    statement: str
    factor: Optional[str] = None
    response: Optional[str] = None
    material: Optional[str] = None
    process: Optional[str] = None
    second_factor: Optional[str] = None
    n_context_papers: int = 0
    n_reference_papers: int = 0
    context_papers: list[str] = field(default_factory=list)
    reference_papers: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class GapReport:
    corpus: dict
    candidates: list[GapCandidate]
    caveats: list[str]

    def to_dict(self) -> dict:
        return {"corpus": self.corpus, "candidates": [asdict(c) for c in self.candidates], "caveats": self.caveats}


def _effect_claims(kb: KnowledgeBase, statuses: Iterable[str], include_synthetic: bool) -> list[Relation]:
    out = []
    for c in kb.active_claims(statuses):
        if c.relation not in EFFECTS or not c.evidence:
            continue
        if c.subject.type not in FACTOR_TYPES or c.object.type not in RESPONSE_TYPES:
            continue
        if c.attributes.get("claim_origin") == "background":
            continue  # cited work is not evidence produced by this corpus
        p = kb.papers.get(c.evidence.paper_id)
        if p and p.is_synthetic and not include_synthetic:
            continue
        out.append(c)
    return out


def combination_table(kb: KnowledgeBase, statuses: Iterable[str] = ("accepted", "review"),
                      include_synthetic: bool = True) -> list[dict]:
    """Material x Process x Parameter x Response with the papers supporting each combination."""
    combos: dict[tuple, set[str]] = defaultdict(set)
    for c in _effect_claims(kb, statuses, include_synthetic):
        if c.subject.type != "Parameter":
            continue
        pid = c.evidence.paper_id
        ctx = kb.paper_context(pid, statuses)
        for m in ctx["materials"] or {"(unspecified)"}:
            for pr in ctx["processes"] or {"(unspecified)"}:
                combos[(kb.entity_name(m) if ":" in m else m, kb.entity_name(pr) if ":" in pr else pr,
                        c.subject.name, c.object.name)].add(pid)
    rows = [{"material": k[0], "process": k[1], "parameter": k[2], "response": k[3], "n_papers": len(v),
             "papers": sorted(v)} for k, v in combos.items()]
    return sorted(rows, key=lambda r: (r["n_papers"], r["material"], r["process"], r["parameter"]))


def find_gaps(kb: KnowledgeBase, min_pair_papers: int = 2, max_context_papers: int = 1,
              statuses: Iterable[str] = ("accepted", "review"), include_synthetic: bool = True,
              limit: int = 50) -> GapReport:
    statuses = list(statuses)
    claims = _effect_claims(kb, statuses, include_synthetic)
    papers = [p for p in kb.papers.values() if include_synthetic or not p.is_synthetic]
    paper_ids = {p.paper_id for p in papers}
    ctx = {pid: kb.paper_context(pid, statuses) for pid in paper_ids}

    pair_papers: dict[tuple[str, str], set[str]] = defaultdict(set)
    factor_resp_by_paper: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    names: dict[str, str] = {}
    for c in claims:
        pid = c.evidence.paper_id
        pair_papers[(c.subject.id, c.object.id)].add(pid)
        factor_resp_by_paper[c.object.id][pid].add(c.subject.id)
        names[c.subject.id], names[c.object.id] = c.subject.name, c.object.name

    # parameters reported per process (HAS_PARAMETER), used to keep contexts meaningful
    proc_params: dict[str, set[str]] = defaultdict(set)
    proc_param_papers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in kb.active_claims(statuses):
        if c.relation == "HAS_PARAMETER" and c.evidence and c.evidence.paper_id in paper_ids:
            proc_params[c.subject.id].add(c.object.id)
            proc_param_papers[(c.subject.id, c.object.id)].add(c.evidence.paper_id)
            names[c.object.id] = c.object.name

    contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pid, cx in ctx.items():
        for m in cx["materials"]:
            for pr in cx["processes"]:
                contexts[(m, pr)].add(pid)

    cands: list[GapCandidate] = []
    # --- A: context gaps -------------------------------------------------------
    for (f, r), fr_papers in pair_papers.items():
        if len(fr_papers) < min_pair_papers:
            continue
        for (m, pr), cx_papers in contexts.items():
            if f.startswith("Parameter:") and f not in proc_params.get(pr, set()):
                continue
            inside = fr_papers & cx_papers
            if len(inside) > max_context_papers:
                continue
            outside = fr_papers - cx_papers
            if not outside:
                continue
            fac, res, mat, proc = names[f], names[r], kb.entity_name(m), kb.entity_name(pr)
            cands.append(GapCandidate(
                kind="A_context", factor=fac, response=res, material=mat, process=proc,
                n_context_papers=len(inside), n_reference_papers=len(fr_papers),
                context_papers=sorted(inside), reference_papers=sorted(outside),
                score=round(len(fr_papers) / (1 + len(inside)) * min(1.0, len(cx_papers) / 2), 3),
                statement=(f"Limited evidence was found in the current corpus for '{fac}' -> '{res}' in "
                           f"{mat} / {proc}: {len(inside)} of {len(cx_papers)} {mat}/{proc} paper(s) report it, "
                           f"while {len(fr_papers)} paper(s) in the corpus report this relationship overall.")))
    # --- B: joint-factor gaps -----------------------------------------------------
    for r, by_paper in factor_resp_by_paper.items():
        factor_papers: dict[str, set[str]] = defaultdict(set)
        for pid, fs in by_paper.items():
            for f in fs:
                factor_papers[f].add(pid)
        strong = [f for f, ps in factor_papers.items() if len(ps) >= 1]

        def ctx_of(ps: set[str]) -> set[tuple[str, str]]:
            return {(m, pr) for pid in ps for m in ctx.get(pid, {}).get("materials", ())
                    for pr in ctx.get(pid, {}).get("processes", ())}
        for f1, f2 in combinations(sorted(strong), 2):
            joint = factor_papers[f1] & factor_papers[f2]
            total = factor_papers[f1] | factor_papers[f2]
            if len(total) < min_pair_papers or len(joint) > max_context_papers:
                continue
            if not (ctx_of(factor_papers[f1]) & ctx_of(factor_papers[f2])):
                continue  # never studied in a shared material/process context -> not comparable
            cands.append(GapCandidate(
                kind="B_joint_factors", factor=names[f1], second_factor=names[f2], response=names[r],
                n_context_papers=len(joint), n_reference_papers=len(total), context_papers=sorted(joint),
                reference_papers=sorted(total), score=round(len(total) / (1 + len(joint)) * 0.8, 3),
                statement=(f"Limited evidence was found in the current corpus of '{names[f1]}' and '{names[f2]}' "
                           f"being studied together for '{names[r]}': {len(joint)} paper(s) report both, "
                           f"{len(total)} paper(s) report at least one.")))
    # --- C: parameter/response pairs with no effect claim -----------------------------
    resp_by_proc: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for pid, cx in ctx.items():
        for c in claims:
            if c.evidence.paper_id == pid:
                for pr in cx["processes"]:
                    resp_by_proc[pr][c.object.id].add(pid)
    for pr, params in proc_params.items():
        for f in params:
            f_papers = proc_param_papers[(pr, f)]
            if len(f_papers) < min_pair_papers:
                continue
            for r, r_papers in resp_by_proc.get(pr, {}).items():
                if len(r_papers) < min_pair_papers or pair_papers.get((f, r)):
                    continue
                cands.append(GapCandidate(
                    kind="C_unlinked", factor=names.get(f, f), response=names.get(r, r), process=kb.entity_name(pr),
                    n_context_papers=0, n_reference_papers=len(f_papers | r_papers),
                    reference_papers=sorted(f_papers | r_papers), score=round(0.5 * min(len(f_papers), len(r_papers)), 3),
                    statement=(f"Limited evidence was found in the current corpus linking '{names.get(f, f)}' to "
                               f"'{names.get(r, r)}' for {kb.entity_name(pr)}: the parameter is reported in "
                               f"{len(f_papers)} paper(s) and the response is studied in {len(r_papers)} paper(s), "
                               f"but no evidence-backed effect claim connects them.")))
    cands.sort(key=lambda c: (c.kind, -c.score))
    corpus = {"papers": len(paper_ids), "synthetic_papers": sum(p.is_synthetic for p in papers),
              "effect_claims_used": len(claims), "factor_response_pairs": len(pair_papers),
              "material_process_contexts": {f"{kb.entity_name(m)} / {kb.entity_name(pr)}": len(v)
                                            for (m, pr), v in contexts.items()},
              "statuses": statuses, "thresholds": {"min_pair_papers": min_pair_papers,
                                                   "max_context_papers": max_context_papers}}
    return GapReport(corpus=corpus, candidates=cands[:limit], caveats=CAVEATS)
