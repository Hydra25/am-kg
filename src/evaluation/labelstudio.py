"""Label Studio bridge for building the gold benchmark.

Workflow (see docs/BENCHMARK.md):

1. `python main.py annotate-export data/papers/lpbf` writes
   data/labelstudio/labeling_config.xml and data/labelstudio/tasks.json.
   Each task is one chunk of one section (about 12 sentences). By default it
   carries the system's extraction as *predictions* (pre-annotations): the
   annotator only corrects, deletes and adds instead of starting from nothing.
2. In Label Studio: create a project, paste the labeling config, import
   tasks.json, annotate, then export the project as JSON.
3. `python main.py annotate-import export.json` converts the human annotations
   into the gold format used by `python main.py evaluate`
   (data/annotations/<pdf stem>.json).

Honesty note for the paper: pre-annotation can bias annotators towards the
system (anchoring). Record `annotation_mode` and annotate a few papers from
scratch (`--no-predictions`) to measure the effect.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from ..normalization.entity_normalizer import EntityNormalizer
from ..normalization.text import match_key
from ..normalization.units import parse_quantity_text
from ..ontology.models import ExtractionResult, ParsedDocument, Sentence

log = logging.getLogger(__name__)

ENTITY_LABELS: dict[str, str] = {
    "Material": "#1f77b4", "Process": "#2ca02c", "Parameter": "#ff7f0e", "Property": "#d62728",
    "Defect": "#8c564b", "Microstructure": "#9467bd", "PostProcessing": "#e377c2",
    "CharacterizationMethod": "#17becf", "Geometry": "#7f7f7f", "ExperimentalCondition": "#bcbd22",
    "Value": "#000000",
}
RELATION_LABELS = ["PROCESSES", "HAS_PARAMETER", "HAS_VALUE", "HAS_CONDITION", "FABRICATES", "CHARACTERIZES",
                   "PRODUCES", "AFFECTS", "INFLUENCES", "CAUSES", "IMPROVES", "CHANGES"]
_SKIP_SECTIONS = {"references", "acknowledgements", "front_matter", "title"}


def labeling_config() -> str:
    """Label Studio labeling interface (paste into Settings -> Labeling Interface -> Code)."""
    labels = "\n".join(f'    <Label value="{name}" background="{color}"/>' for name, color in ENTITY_LABELS.items())
    rels = "\n".join(f'    <Relation value="{r}"/>' for r in RELATION_LABELS)
    return f"""<View>
  <Header value="$source_file | $section | page $pages"/>
  <Labels name="label" toName="text">
{labels}
  </Labels>
  <Relations>
{rels}
  </Relations>
  <Text name="text" value="$text" granularity="word"/>
  <Header value="Canonical name for the selected entity (only if it differs from the text, e.g. SLM -> LPBF)" size="6"/>
  <TextArea name="canonical" toName="text" perRegion="true" editable="true" maxSubmissions="1"
            placeholder="canonical name (optional)"/>
</View>
"""


# ----------------------------------------------------------------------------------- export
def _chunks(doc: ParsedDocument, max_sentences: int) -> list[tuple[str, list[Sentence]]]:
    out: list[tuple[str, list[Sentence]]] = []
    cur: list[Sentence] = []
    for s in doc.sentences:
        if s.section_type in _SKIP_SECTIONS:
            continue            # table rows are kept: many property/parameter values live there
        if cur and (s.section_id != cur[0].section_id or len(cur) >= max_sentences):
            out.append((cur[0].section_title, cur))
            cur = []
        cur.append(s)
    if cur:
        out.append((cur[0].section_title, cur))
    return out


def build_tasks(doc: ParsedDocument, result: Optional[ExtractionResult] = None, max_sentences: int = 12,
                statuses: Iterable[str] = ("accepted", "review"), model_version: str = "am-kg") -> list[dict]:
    """Label Studio tasks for one paper, optionally with the system output as predictions."""
    statuses = set(statuses)
    stem = Path(doc.paper.source_file or doc.paper.paper_id).stem
    mentions_by_sent: dict[str, list] = defaultdict(list)
    rels_by_sent: dict[str, list] = defaultdict(list)
    if result is not None:
        for m in result.entities:
            if m.char_start is not None and m.char_end is not None:
                mentions_by_sent[m.sentence_id].append(m)
        for r in result.relations:
            if (r.status.value in statuses and r.evidence and r.relation in RELATION_LABELS
                    and r.knowledge_type.value != "DERIVED"):
                rels_by_sent[r.evidence.sentence_id].append(r)

    tasks = []
    for n, (section, sents) in enumerate(_chunks(doc, max_sentences)):
        parts, offsets, pos = [], [], 0
        for s in sents:
            parts.append(s.text)
            offsets.append([s.sentence_id, pos, pos + len(s.text), s.page])
            pos += len(s.text) + 1
        text = "\n".join(parts)
        pages = sorted({o[3] for o in offsets if o[3] is not None})
        data = {"text": text, "paper_id": doc.paper.paper_id, "source_file": stem, "section": section,
                "pages": ", ".join(map(str, pages)), "chunk": n, "sentence_offsets": offsets,
                "is_synthetic": doc.paper.is_synthetic}
        task: dict[str, Any] = {"data": data}
        if result is not None:
            task["predictions"] = [{"model_version": model_version,
                                    "result": _predictions(sents, offsets, mentions_by_sent, rels_by_sent)}]
        tasks.append(task)
    return tasks


def _predictions(sents: list[Sentence], offsets: list, mentions_by_sent: dict, rels_by_sent: dict) -> list[dict]:
    result: list[dict] = []
    region_of: dict[tuple, str] = {}        # (sentence_id, canonical_id) -> region id
    text_start = {o[0]: o[1] for o in offsets}

    def region(rid: str, start: int, end: int, text: str, label: str) -> dict:
        return {"id": rid, "from_name": "label", "to_name": "text", "type": "labels",
                "value": {"start": start, "end": end, "text": text, "labels": [label]}}

    for s in sents:
        base = text_start[s.sentence_id]
        taken: list[tuple[int, int]] = []
        for i, m in enumerate(sorted(mentions_by_sent.get(s.sentence_id, []), key=lambda m: m.char_start)):
            if m.entity_type not in ENTITY_LABELS or any(m.char_start < e and b < m.char_end for b, e in taken):
                continue
            rid = f"{s.sentence_id}_m{i}"
            taken.append((m.char_start, m.char_end))
            result.append(region(rid, base + m.char_start, base + m.char_end, m.text, m.entity_type))
            region_of.setdefault((s.sentence_id, m.canonical_id), rid)
        for j, r in enumerate(rels_by_sent.get(s.sentence_id, [])):
            src = region_of.get((s.sentence_id, r.subject.id))
            if r.relation == "HAS_VALUE":
                raw = r.object.quantity.raw if r.object.quantity else r.object.name
                k = s.text.find(raw)
                if src is None or k < 0 or any(k < e and b < k + len(raw) for b, e in taken):
                    continue
                dst = f"{s.sentence_id}_v{j}"
                taken.append((k, k + len(raw)))
                result.append(region(dst, base + k, base + k + len(raw), raw, "Value"))
            else:
                dst = region_of.get((s.sentence_id, r.object.id))
            if src and dst and src != dst:
                result.append({"from_id": src, "to_id": dst, "type": "relation", "direction": "right",
                               "labels": [r.relation]})
    return result


# ----------------------------------------------------------------------------------- import
def _latest_annotation(task: dict) -> Optional[list[dict]]:
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    anns.sort(key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    return anns[-1].get("result", [])


def import_annotations(export: list[dict], normalizer: EntityNormalizer, annotator: str = "",
                       annotation_mode: str = "model_assisted") -> dict[str, dict]:
    """Label Studio JSON export -> {pdf stem: gold dict} in the format of data/annotations/*.json."""
    papers: dict[str, dict] = {}
    for task in export:
        data = task.get("data", {})
        result = _latest_annotation(task)
        stem = data.get("source_file")
        if result is None or not stem:
            continue
        g = papers.setdefault(stem, {"source_file": stem, "annotator": annotator, "annotation_mode": annotation_mode,
                                     "notes": "Imported from Label Studio.", "entities": [], "relations": [],
                                     "values": [], "normalization": [], "_tasks": 0})
        g["_tasks"] += 1
        text = data.get("text", "")
        offsets = data.get("sentence_offsets", [])

        regions: dict[str, dict] = {}
        canon_override: dict[str, str] = {}
        for item in result:
            if item.get("type") == "labels":
                v = item["value"]
                regions[item["id"]] = {"label": v["labels"][0], "text": v.get("text") or text[v["start"]:v["end"]],
                                       "start": v["start"]}
            elif item.get("type") == "textarea" and item.get("from_name") == "canonical":
                vals = item.get("value", {}).get("text") or []
                if vals and vals[0].strip():
                    canon_override[item["id"]] = vals[0].strip()

        canonical: dict[str, str] = {}
        for rid, r in regions.items():
            if r["label"] == "Value":
                continue
            if rid in canon_override:
                canonical[rid] = canon_override[rid]
                g["normalization"].append({"surface": r["text"], "type": r["label"], "canonical": canon_override[rid]})
            else:
                canonical[rid] = normalizer.normalize(r["text"], r["label"]).canonical_name
            g["entities"].append({"type": r["label"], "canonical": canonical[rid]})

        for item in result:
            if item.get("type") != "relation" or not item.get("labels"):
                continue
            a, b = item["from_id"], item["to_id"]
            if item.get("direction") == "left":
                a, b = b, a
            if a not in regions or b not in regions:
                continue
            rel = item["labels"][0]
            evidence = _sentence_at(regions[a]["start"], text, offsets)
            if rel == "HAS_VALUE" and regions[b]["label"] == "Value" and a in canonical:
                q = parse_quantity_text(regions[b]["text"])
                if q is None or not q.normalized_unit:
                    log.warning("Could not parse value %r in %s", regions[b]["text"], stem)
                    continue
                val = {"entity": canonical[a], "normalized_unit": q.normalized_unit, "raw": regions[b]["text"]}
                if q.normalized_value is not None:
                    val["normalized_value"] = q.normalized_value
                else:
                    val["normalized_min"], val["normalized_max"] = q.normalized_min, q.normalized_max
                g["values"].append(val)
            elif a in canonical and b in canonical:
                g["relations"].append({"subject": canonical[a], "relation": rel, "object": canonical[b],
                                       "evidence": evidence})

    for g in papers.values():   # document-level sets: remove duplicates, keep order
        g["entities"] = _dedup(g["entities"], lambda e: (e["type"], match_key(e["canonical"])))
        g["relations"] = _dedup(g["relations"], lambda r: (match_key(r["subject"]), r["relation"], match_key(r["object"])))
        g["values"] = _dedup(g["values"], lambda v: (match_key(v["entity"]), v.get("normalized_value"),
                                                     v.get("normalized_min"), v.get("normalized_max"), v["normalized_unit"]))
        g["normalization"] = _dedup(g["normalization"], lambda n: (n["type"], match_key(n["surface"])))
        g["notes"] += f" {g.pop('_tasks')} annotated chunks."
    return papers


def _sentence_at(pos: int, text: str, offsets: list) -> str:
    for _sid, a, b, _page in offsets:
        if a <= pos <= b:
            return text[a:b]
    return ""


def _dedup(items: list[dict], key) -> list[dict]:
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def write_gold(papers: dict[str, dict], out_dir: Path, overwrite: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, g in papers.items():
        p = out_dir / f"{stem}.json"
        if p.exists() and not overwrite:
            log.warning("%s exists; use --overwrite to replace it", p)
            continue
        p.write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(p)
    return written
