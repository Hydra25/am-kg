"""Rule-based relation extraction with evidence, direction and hedging.

Every relation produced here is backed by one evidence sentence (or a table
row, or explicitly listed context sentences). Nothing is inferred from general
scientific knowledge.

Relation families
-----------------
EXPLICIT   causal/effect predicates ("increasing laser power reduced porosity"),
           passive voice ("porosity may be promoted by excessive laser power"),
           nominal ("the effect of X on Y", "X had no effect on Y"),
           fabrication predicates ("Ti-6Al-4V was fabricated by LPBF").
EXTRACTED  structural relations built from co-occurrence in one sentence:
           HAS_VALUE, HAS_PARAMETER, HAS_CONDITION, PRODUCES, CHARACTERIZES.
DERIVED    trends computed from table rows (see derive_table_trends).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..entity_extraction.rule_based import SentenceAnalysis, evidence_for
from ..normalization.text import normalize_chars
from ..ontology.models import EntityMention, EntityRef, Evidence, Relation, Sentence, stable_id
from ..ontology.schema import EntityType, KnowledgeType, RelationType, infer_effect_relation

E = EntityType
R = RelationType

# ---------------------------------------------------------------------------
# lexical cues
# ---------------------------------------------------------------------------
TRIGGERS: dict[str, list[str]] = {
    "increase": ["increased", "increases", "raised", "raises", "enhanced", "enhances", "promoted",
                 "promotes", "elevated", "intensified", "intensifies", "accelerated", "facilitated",
                 "increase", "promote", "enhance"],
    "decrease": ["decreased", "decreases", "reduced", "reduces", "lowered", "lowers", "suppressed",
                 "suppresses", "minimized", "minimised", "minimizes", "eliminated", "eliminates",
                 "closed", "closes", "mitigated", "mitigates", "diminished", "inhibited", "inhibits",
                 "hindered", "prevented", "prevents", "dissolved", "dissolves", "refined", "refines",
                 "decrease", "reduce", "suppress", "eliminate", "mitigate"],
    "improve": ["improved", "improves", "improve"],
    "deteriorate": ["deteriorated", "deteriorates", "degraded", "degrades", "worsened", "impaired",
                    "compromised", "weakened", "deteriorate", "degrade"],
    "affect": ["affected", "affects", "influenced", "influences", "governed", "governs", "controlled",
               "controls", "determined", "determines", "altered", "alters", "changed", "changes",
               "modified", "modifies", "dominated", "transformed", "transforms", "dictated",
               "affect", "influence", "alter", "change", "govern", "control", "transform"],
    "cause": ["caused", "causes", "led to", "leads to", "lead to", "resulted in", "results in",
              "induced", "induces", "generated", "gave rise to", "gives rise to", "triggered",
              "initiated", "cause", "induce"],
}
_BASE_FORMS = {"increase", "promote", "enhance", "decrease", "reduce", "suppress", "eliminate", "mitigate",
               "improve", "deteriorate", "degrade", "affect", "influence", "alter", "change", "govern",
               "control", "transform", "cause", "induce", "lead to"}
_TRIGGER_CLASS = {w: cls for cls, words in TRIGGERS.items() for w in words}
_TRIGGER_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(w) for w in _TRIGGER_CLASS), key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
_AUX_BEFORE_BASE = re.compile(r"\b(did|does|do|can|could|may|might|would|will|to|should|must|not)\s+(?:\w+ly\s+)?$", re.I)
_PASSIVE_AUX = re.compile(r"\b(was|were|is|are|been|be|being)\s+(?:\w+ly\s+)?$", re.I)
_BY = re.compile(r"^\s*(?:by|with|through|after|following|upon|via|due to)\b(?:\s+(?:the|an?|excessive|high|higher|low|lower|insufficient|increased|decreased|increasing|decreasing))*\s*", re.I)
_NEGATION = re.compile(r"\b(not|no|never|neither|nor|hardly|barely|without)\b|n't", re.I)
_HEDGE = re.compile(r"\b(may|might|could|possibly|likely|suggests?|suggested|appears?|seems?|presumably|"
                    r"probably|potentially|hypothesi[sz]ed|speculated|expected to)\b", re.I)
_BACKGROUND = re.compile(r"\[\d+(?:[,–\-]\s*\d+)*\]|\bet al\b|\b(previous|earlier|prior|other)\s+(studies|work|research|reports?)\b|"
                         r"\breported (that|by)\b|\bliterature\b", re.I)
_INC_MOD = re.compile(r"\b(increasing|increased|higher|high|elevated|larger|greater|excessive|more|increase in|rise in|raising)\s+(?:the\s+)?$", re.I)
_DEC_MOD = re.compile(r"\b(decreasing|decreased|lower|low|reduced|smaller|insufficient|less|reduction in|decrease in|lowering|reducing)\s+(?:the\s+)?$", re.I)
_FAB_TRIGGER = re.compile(r"\b(fabricated|manufactured|produced|printed|built|processed|deposited|made|"
                          r"fabricating|manufacturing|printing|building|processing|depositing|process|"
                          r"fabricate|manufacture|print|deposit)\b", re.I)
_CHAR_TRIGGER = re.compile(r"\b(analy[sz]ed|characteri[sz]ed|examined|observed|measured|investigated|revealed|"
                           r"detected|quantified|determined|identified|confirmed|evaluated|inspected|imaged|"
                           r"mapped|assessed|analy[sz]is|characteri[sz]ation|measurements?|using|via|by)\b", re.I)
_COREF_CUE = re.compile(r"\b(resulting|resultant|obtained|as-built|as-deposited|as-printed|these|the specimens?|"
                        r"the samples?|the parts?|the walls?|the cubes?)\b", re.I)
_STATE_ASBUILT = re.compile(r"\b(as-built|as built|as-deposited|as-printed|as-fabricated|as-received)\b", re.I)
_CLAUSE_SPLIT = re.compile(r";|,\s*(?:while|whereas|but|although|however)\b|\s(?:whereas|although)\s")

CHAIN = r"«E\d+»(?:\s*(?:,|and|or|,\s*and|as well as)\s*(?:the\s+)?«E\d+»)*"
_GAP = r"(?:[^«;.]|«Q\d+»)"
_NOMINAL_RE = re.compile(
    rf"\b(?:effects?|influences?|impacts?|roles?|dependence)\s+of\s+(?:the\s+)?(?:\w+\s+){{0,2}}?(?P<s>{CHAIN})"
    rf"(?:\s+{_GAP}{{0,30}}?)?\s+(?:on|upon)\s+(?:the\s+)?(?:\w+\s+){{0,2}}?(?P<o>{CHAIN})", re.I)
_HAD_EFFECT_RE = re.compile(
    rf"(?P<s>{CHAIN})\s+(?:has|had|have|showed|exhibited|exerted|shows)\s+(?P<q>no|a|an|little|negligible|minor|"
    rf"strong|significant|considerable|marginal)?\s*(?:\w+\s+){{0,2}}?(?:effect|influence|impact)\s+on\s+"
    rf"(?:the\s+)?(?:\w+\s+){{0,2}}?(?P<o>{CHAIN})", re.I)
_ATTRIB_RE = re.compile(
    rf"(?P<o>{CHAIN}){_GAP}{{0,50}}?\b(?:attributed|ascribed|due|owing|related)\s+to\s+(?:the\s+)?"
    rf"(?:\w+\s+){{0,3}}?(?P<s>{CHAIN})", re.I)


@dataclass
class Tagged:
    text: str
    ent_at: dict[int, int]        # tag number -> mention index


def tag_sentence(an: SentenceAnalysis) -> Tagged:
    """Replace mentions with «E#» and quantities with «Q#» (on normalised text)."""
    text = normalize_chars(an.sentence.text)
    spans: list[tuple[int, int, str]] = []
    for i, m in enumerate(an.mentions):
        if m.char_start is not None and m.char_end is not None:
            spans.append((m.char_start, m.char_end, f"«E{i}»"))
    for j, q in enumerate(an.quantities):
        if not any(q.start < b and a < q.end for a, b, _ in spans):
            spans.append((q.start, q.end, f"«Q{j}»"))
    spans.sort()
    out, pos = [], 0
    for a, b, tag in spans:
        if a < pos:
            continue
        out.append(text[pos:a])
        out.append(tag)
        pos = b
    out.append(text[pos:])
    return Tagged("".join(out), {i: i for i in range(len(an.mentions))})


def chain_ids(chain: str) -> list[int]:
    return [int(x) for x in re.findall(r"«E(\d+)»", chain)]


def _ref(m: EntityMention) -> EntityRef:
    return EntityRef(id=m.canonical_id or "", name=m.canonical_name or m.text, type=m.entity_type,
                     surface=m.text, normalization_confidence=m.normalization_confidence)


class RuleRelationExtractor:
    """Produces Relation objects from SentenceAnalysis objects of one paper."""

    def __init__(self, paper_id: str) -> None:
        self.paper_id = paper_id

    # ------------------------------------------------------------------
    def _make(self, s: EntityRef, rel: RelationType, o: EntityRef, ev: Evidence, conf: float,
              ktype: KnowledgeType, method: str = "rule", **attrs) -> Relation:
        cid = stable_id("c", self.paper_id, s.id, rel.value, o.id, ev.sentence_id, *ev.context_sentence_ids)
        return Relation(claim_id=cid, subject=s, relation=rel.value, object=o, evidence=ev,
                        confidence=round(max(0.0, min(conf, 0.99)), 3), method=method,
                        knowledge_type=ktype, attributes={k: v for k, v in attrs.items() if v is not None})

    @staticmethod
    def _base_adjust(sentence: Sentence) -> tuple[float, str]:
        """Confidence adjustment + claim origin from section / citation cues."""
        if sentence.section_type == "introduction" or _BACKGROUND.search(sentence.text):
            return -0.15, "background"
        if sentence.section_type == "abstract":
            return 0.0, "abstract"
        return 0.0, "this_study"

    # ------------------------------------------------------------------
    def effect_relations(self, an: SentenceAnalysis) -> list[Relation]:
        """EXPLICIT causal / effect statements."""
        rels: list[Relation] = []
        tg = tag_sentence(an)
        ev = evidence_for(an.sentence, self.paper_id)
        adj, origin = self._base_adjust(an.sentence)
        hedged = bool(_HEDGE.search(an.sentence.text))
        seen: set[tuple[int, int]] = set()

        def emit(sids: list[int], oids: list[int], cls: str, base: float, trigger: str,
                 negated: bool, claim_type: str, subj_change: Optional[str]) -> None:
            obj_change = {"increase": "increase", "decrease": "decrease", "improve": "improve",
                          "deteriorate": "deteriorate", "cause": "induced"}.get(cls, "unspecified")
            trend = "unspecified"
            if subj_change in ("increase", "decrease") and obj_change in ("increase", "decrease"):
                trend = "positive" if subj_change == obj_change else "negative"
            penalty = 0.03 * (len(sids) - 1 + len(oids) - 1)
            for si in sids:
                for oi in oids:
                    if (si, oi) in seen or si == oi:
                        continue
                    sm, om = an.mentions[si], an.mentions[oi]
                    rel = infer_effect_relation(EntityType(sm.entity_type), EntityType(om.entity_type), cls)
                    if rel is None:
                        continue
                    seen.add((si, oi))
                    conf = base + adj - penalty - (0.12 if hedged else 0.0)
                    rels.append(self._make(
                        _ref(sm), rel, _ref(om), ev, conf, KnowledgeType.EXPLICIT,
                        trigger=trigger, trigger_class=cls, claim_type=claim_type, claim_origin=origin,
                        subject_change=subj_change, object_change=obj_change if not negated else "none",
                        trend=trend if not negated else "none", negated=negated, hedged=hedged,
                    ))

        for _clause_start, clause in _clauses(tg.text):
            # (a) nominal: "effect of X on Y", "X had no effect on Y"
            for m in _NOMINAL_RE.finditer(clause):
                before = clause[:m.start()]
                investigated = bool(re.search(r"(investigat|stud|examin|explor|analy[sz]|evaluat|assess|aim)\w*\W+(?:\w+\W+){0,4}$", before, re.I))
                negated = bool(re.search(r"\b(no|negligible|insignificant|little|minor)\s+(?:\w+\s+)?$", before, re.I))
                emit(chain_ids(m.group("s")), chain_ids(m.group("o")), "affect",
                     0.8 if investigated else 0.85, "effect of ... on", negated,
                     "investigated" if investigated else "finding", None)
            for m in _HAD_EFFECT_RE.finditer(clause):
                q = (m.group("q") or "").lower()
                negated = q in {"no", "negligible", "little", "marginal"}
                emit(chain_ids(m.group("s")), chain_ids(m.group("o")), "affect", 0.88,
                     "had effect on", negated, "finding", None)
            for m in _ATTRIB_RE.finditer(clause):
                emit(chain_ids(m.group("s")), chain_ids(m.group("o")), "cause", 0.82,
                     "attributed to", False, "finding", _subject_change(clause, m.start("s")))
            # (b) verbal triggers
            last: Optional[tuple[list[int], list[int], Optional[str]]] = None  # previous (subjects, objects, change)
            for tm in _TRIGGER_RE.finditer(clause):
                trig = tm.group(1).lower()
                cls = _TRIGGER_CLASS.get(trig)
                if cls is None:
                    continue
                left, right = clause[:tm.start()], clause[tm.end():]
                if trig in _BASE_FORMS and not _AUX_BEFORE_BASE.search(left):
                    continue  # "an increase in ..." is a noun, not a predicate
                passive = bool(_PASSIVE_AUX.search(left)) and bool(_BY.match(right))
                if passive:
                    subj_m = re.match(_BY.pattern + rf"(?P<c>{CHAIN})", right, re.I)
                    obj_m = re.search(rf"(?P<c>{CHAIN}){_GAP}{{0,60}}$", left)
                    if not subj_m or not obj_m:
                        continue
                    by_mods = right[:subj_m.start("c")]
                    subj_change = _subject_change(by_mods + " ", len(by_mods) + 1)
                    sids, oids = chain_ids(subj_m.group("c")), chain_ids(obj_m.group("c"))
                    base = 0.86
                else:
                    subj_m = re.search(rf"(?P<c>{CHAIN}){_GAP}{{0,80}}$", left)
                    obj_m = re.match(rf"{_GAP}{{0,50}}?(?P<c>{CHAIN})", right)
                    if not subj_m or not obj_m:
                        continue
                    subj_change = _subject_change(left, subj_m.start("c"))
                    sids, oids = chain_ids(subj_m.group("c")), chain_ids(obj_m.group("c"))
                    gap = left[subj_m.end("c"):]
                    gap_len = len(gap)
                    base = 0.9 - (0.05 if gap_len > 50 else 0.0)
                    # coordinated predicate: "HIP closed the pores and improved the elongation"
                    if last and set(sids) <= set(last[1]) and re.fullmatch(
                            r"(?:\s|«Q\d+»|to|the|of|by|from|\w+ly)*,?\s*(?:and|but)\s+(?:also\s+|then\s+|thereby\s+)?", gap):
                        sids, subj_change = last[0], last[2]
                        base -= 0.03
                mid = left[obj_m.end("c"):] if passive else left[subj_m.end("c"):]
                negated = bool(_NEGATION.search(mid))
                emit(sids, oids, cls, base, trig, negated, "finding", subj_change)
                last = (sids, oids, subj_change)
        return rels

    # ------------------------------------------------------------------
    def structural_relations(self, an: SentenceAnalysis, prev: list[SentenceAnalysis],
                             paper_processes: list[EntityMention]) -> list[Relation]:
        """EXTRACTED relations: values, parameters, conditions, processing, characterization."""
        rels: list[Relation] = []
        s = an.sentence
        ev = evidence_for(s, self.paper_id)
        adj, origin = self._base_adjust(s)
        text = s.text
        procs = an.of_type(E.PROCESS.value)
        mats = an.of_type(E.MATERIAL.value)
        state = _material_state(an)

        # --- process resolution for sentences without an explicit process -----
        def resolve_process() -> tuple[list[EntityMention], Evidence, float, str]:
            if procs:
                return [m for _, m in procs], ev, 0.0, "same_sentence"
            for p in reversed(prev[-4:]):
                pp = [m for _, m in p.of_type(E.PROCESS.value)]
                if pp and p.sentence.section_id == s.section_id:
                    ctx_ev = Evidence(**{**ev.model_dump(), "text": f"{p.sentence.text} {s.text}",
                                         "context_sentence_ids": [p.sentence.sentence_id]})
                    return pp, ctx_ev, -0.1, "previous_sentence"
            uniq = {m.canonical_id: m for m in paper_processes}
            if len(uniq) == 1:
                return list(uniq.values()), ev, -0.18, "paper_single_process"
            return [], ev, 0.0, "unresolved"

        # --- HAS_VALUE ---------------------------------------------------------
        for a in an.assignments:
            m = an.mentions[a.mention_index]
            vtype = E.PARAMETER_VALUE if m.entity_type in (E.PARAMETER.value, E.CONDITION.value) else E.PROPERTY_VALUE
            q = a.qspan.quantity
            vid = stable_id("val", self.paper_id, s.sentence_id, m.canonical_id, a.qspan.start, q.raw)
            vref = EntityRef(id=vid, name=q.raw, type=vtype.value, surface=q.raw, quantity=q)
            conf = 0.93 - (0.05 if a.distance > 40 else 0.0) + (adj / 3)
            ctx = {"materials": sorted({x.canonical_name for _, x in mats}),
                   "processes": sorted({x.canonical_name for _, x in procs}), "state": state}
            rels.append(self._make(_ref(m), R.HAS_VALUE, vref, ev, conf, KnowledgeType.EXTRACTED,
                                   claim_origin=origin, value_context=ctx))

        # --- HAS_PARAMETER / HAS_CONDITION / PRODUCES ----------------------------
        params = an.of_type(E.PARAMETER.value)
        conds = an.of_type(E.CONDITION.value)
        valued_props = {a.mention_index for a in an.assignments
                        if an.mentions[a.mention_index].entity_type in (E.PROPERTY.value, E.MICROSTRUCTURE.value)}
        if params or conds or valued_props:
            if origin == "background" and not procs:
                p_list, p_ev, p_adj, how = [], ev, 0.0, "unresolved"
            else:
                p_list, p_ev, p_adj, how = resolve_process()
            for pm in p_list:
                for _, par in params:
                    has_val = any(an.mentions[a.mention_index] is par for a in an.assignments)
                    conf = (0.9 if has_val or re.search(r"\b(with|using|at|of)\b", text) else 0.85) + p_adj + adj
                    rels.append(self._make(_ref(pm), R.HAS_PARAMETER, _ref(par), p_ev, conf,
                                           KnowledgeType.EXTRACTED, claim_origin=origin, context_resolution=how))
                for _, cm in conds:
                    rels.append(self._make(_ref(pm), R.HAS_CONDITION, _ref(cm), p_ev, 0.85 + p_adj + adj,
                                           KnowledgeType.EXTRACTED, claim_origin=origin, context_resolution=how))
                for idx in valued_props:
                    prop = an.mentions[idx]
                    if how == "same_sentence":
                        conf = 0.86
                    elif how == "previous_sentence" and _COREF_CUE.search(text):
                        conf = 0.86          # "... using LPBF. The resulting tensile strength was ..."
                    else:
                        continue             # values without process context stay value-only
                    rels.append(self._make(_ref(pm), R.PRODUCES, _ref(prop), p_ev, conf + adj,
                                           KnowledgeType.EXTRACTED, claim_origin=origin,
                                           context_resolution=how, material_state=state))

        # --- PROCESSES (Process -> Material) / FABRICATES ------------------------
        if procs and mats:
            fab = _FAB_TRIGGER.search(text)
            for _, pm in procs:
                for _, mm in mats:
                    between = _between(text, pm, mm)
                    if fab:
                        conf, kt = 0.92, KnowledgeType.EXPLICIT
                    elif len(between.strip(" -")) <= 12 and re.fullmatch(r"[\s\-]*(?:of|processed|printed|fabricated|-processed)?[\s\-]*", between, re.I):
                        conf, kt = 0.86, KnowledgeType.EXTRACTED   # "LPBF Ti-6Al-4V", "LPBF of Ti64"
                    else:
                        conf, kt = 0.62, KnowledgeType.EXTRACTED   # plain co-occurrence -> review
                    rels.append(self._make(_ref(pm), R.PROCESSES, _ref(mm), ev, conf + adj, kt,
                                           claim_origin=origin, trigger=fab.group(1) if fab else None))
        geoms = an.of_type(E.GEOMETRY.value)
        if geoms and _FAB_TRIGGER.search(text):
            p_list, p_ev, p_adj, how = (([m for _, m in procs], ev, 0.0, "same_sentence") if procs
                                        else resolve_process())
            for pm in p_list:
                for _, gm in geoms:
                    rels.append(self._make(_ref(pm), R.FABRICATES, _ref(gm), p_ev, 0.88 + p_adj + adj,
                                           KnowledgeType.EXPLICIT if procs else KnowledgeType.EXTRACTED,
                                           claim_origin=origin, context_resolution=how))

        # --- CHARACTERIZES --------------------------------------------------------
        rels += self._characterization(an, ev, adj, origin)
        return rels

    def _characterization(self, an: SentenceAnalysis, ev: Evidence, adj: float, origin: str) -> list[Relation]:
        methods = an.of_type(E.CHARACTERIZATION.value)
        targets = an.of_type(E.MATERIAL.value, E.MICROSTRUCTURE.value, E.DEFECT.value, E.PROPERTY.value)
        if not methods or not targets or not _CHAR_TRIGGER.search(an.sentence.text):
            return []
        # segment the sentence at ", and"/" and "/";" and merge segments without a trigger verb
        text = an.sentence.text
        cuts = [0] + [m.end() for m in re.finditer(r",\s*and\s+|;\s*|\s+and\s+|,\s+", text)] + [len(text)]
        segs: list[list[int]] = []
        for a, b in zip(cuts, cuts[1:]):
            has_trig = bool(re.search(r"\b(was|were|is|are)\b", text[a:b])) or not segs
            if has_trig:
                segs.append([a, b])
            else:
                segs[-1][1] = b
        out: list[Relation] = []
        for a, b in segs:
            ms = [m for _, m in methods if a <= (m.char_start or 0) < b]
            ts = [m for _, m in targets if a <= (m.char_start or 0) < b]
            if not ts and ms:
                continue
            for mm in ms:
                for tm in ts:
                    out.append(self._make(_ref(mm), R.CHARACTERIZES, _ref(tm), ev, 0.87 + adj,
                                          KnowledgeType.EXTRACTED, claim_origin=origin))
        return out


# ---------------------------------------------------------------------------
def _clauses(text: str) -> Iterable[tuple[int, str]]:
    pos = 0
    for m in _CLAUSE_SPLIT.finditer(text):
        yield pos, text[pos:m.start()]
        pos = m.end()
    yield pos, text[pos:]


def _subject_change(text: str, subj_start: int) -> Optional[str]:
    before = text[max(0, subj_start - 30):subj_start]
    if _INC_MOD.search(before):
        return "increase"
    if _DEC_MOD.search(before):
        return "decrease"
    return None


def _between(text: str, a: EntityMention, b: EntityMention) -> str:
    x0, x1 = sorted([(a.char_start or 0, a.char_end or 0), (b.char_start or 0, b.char_end or 0)])
    return text[x0[1]:x1[0]]


def _material_state(an: SentenceAnalysis) -> Optional[str]:
    posts = sorted({m.canonical_name for _, m in an.of_type(E.POST_PROCESSING.value) if m.canonical_name})
    if posts:
        return "post-processed: " + ", ".join(posts)
    if _STATE_ASBUILT.search(an.sentence.text):
        return "as-built"
    return None
