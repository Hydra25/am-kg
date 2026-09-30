"""Silver-standard annotation helper (used to build data/annotations_silver/, see docs/SILVER_GUIDELINE.md).

Run from the project root. Reads data/labelstudio/tasks.json, writes data/labelstudio/silver_out/<stem>/<chunk>.json
and data/labelstudio/silver_export.json (Label Studio export format) for `python main.py annotate-import`.

  python scripts/silver_annotate.py list                       # papers and chunk counts
  python scripts/silver_annotate.py show <stem> <chunk>        # numbered sentences of one task (raw text, no system predictions)
  python scripts/silver_annotate.py check <stem> <chunk>       # validate out/<stem>/<chunk>.json, print normalised names
  python scripts/silver_annotate.py checkall [<stem>]          # validate every file
  python scripts/silver_annotate.py build                      # write export.json in Label Studio export format

Annotation file out/<stem>/<chunk>.json:
{
  "entities": [
    {"id": "e1", "s": 3, "text": "laser power", "label": "Parameter"},
    {"id": "e2", "s": 3, "text": "200 W", "label": "Value"},
    {"id": "e3", "s": 5, "text": "SLM", "label": "Process", "occ": 2, "canonical": "LPBF"}
  ],
  "relations": [ {"from": "e1", "to": "e2", "label": "HAS_VALUE"} ]
}
  s   = sentence number shown by `show` (0-based)
  occ = which occurrence of `text` inside that sentence (1-based, default 1); matching is exact and case-sensitive
  canonical = optional, only if the automatic normalised name (printed by `check`) is wrong
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data" / "labelstudio"
sys.path.insert(0, str(ROOT))

from src.normalization.units import parse_quantity_text  # noqa: E402

TASKS = json.loads((HERE / "tasks.json").read_text())
LABELS = {"Material", "Process", "Parameter", "Property", "Defect", "Microstructure", "PostProcessing",
          "CharacterizationMethod", "Geometry", "ExperimentalCondition", "Value"}
RELS = {  # relation -> (allowed subject labels, allowed object labels)
    "PROCESSES": ({"Process"}, {"Material"}),
    "HAS_PARAMETER": ({"Process"}, {"Parameter"}),
    "HAS_VALUE": ({"Parameter", "Property", "Defect", "ExperimentalCondition", "Microstructure", "Geometry"}, {"Value"}),
    "HAS_CONDITION": ({"Process"}, {"ExperimentalCondition"}),
    "AFFECTS": ({"Parameter", "PostProcessing", "Microstructure", "ExperimentalCondition", "Defect", "Process"},
                {"Property", "Microstructure"}),
    "INFLUENCES": ({"Parameter", "PostProcessing", "ExperimentalCondition"}, {"Defect"}),
    "CAUSES": ({"Process", "Defect", "Parameter"}, {"Defect"}),
    "IMPROVES": ({"PostProcessing"}, {"Property"}),
    "CHANGES": ({"PostProcessing"}, {"Microstructure"}),
    "CHARACTERIZES": ({"CharacterizationMethod"}, {"Material", "Microstructure", "Defect", "Property"}),
    "PRODUCES": ({"Process"}, {"Property", "Microstructure"}),
    "FABRICATES": ({"Process"}, {"Geometry"}),
}


def task(stem, chunk):
    for t in TASKS:
        if t["data"]["source_file"] == stem and t["data"]["chunk"] == int(chunk):
            return t
    raise SystemExit(f"no task {stem} {chunk}")


def sentences(t):
    d = t["data"]
    return [(a, b, d["text"][a:b]) for _sid, a, b, _p in d["sentence_offsets"]]


def cmd_list():
    from collections import Counter
    c = Counter(t["data"]["source_file"] for t in TASKS)
    for k, v in sorted(c.items()):
        done = len(list((HERE / "silver_out" / k).glob("*.json"))) if (HERE / "silver_out" / k).exists() else 0
        print(f"{k}: {v} chunks (0..{v - 1}), annotated {done}")


def cmd_show(stem, chunk):
    t = task(stem, chunk)
    d = t["data"]
    print(f"# {stem} chunk {chunk} | section: {d['section']} | pages {d['pages']}")
    for i, (_a, _b, s) in enumerate(sentences(t)):
        print(f"[{i}] {s}")


_norm = None


def normalizer():
    global _norm
    if _norm is None:
        import os

        from src.config import get_settings
        from src.normalization.entity_normalizer import EntityNormalizer
        from src.normalization.lexicon import Lexicon
        os.chdir(ROOT)
        _norm = EntityNormalizer(Lexicon.load(get_settings().path("lexicon_file")))
    return _norm


def resolve(stem, chunk, ann, verbose=False):
    """-> (regions, relations, errors, warnings)"""
    t = task(stem, chunk)
    sents = sentences(t)
    errs, warns, regions = [], [], {}
    spans = []
    for e in ann.get("entities", []):
        eid = e.get("id")
        if not eid or eid in regions:
            errs.append(f"entity without unique id: {e}")
            continue
        if e.get("label") not in LABELS:
            errs.append(f"{eid}: bad label {e.get('label')}")
            continue
        s = e.get("s")
        if not isinstance(s, int) or not 0 <= s < len(sents):
            errs.append(f"{eid}: bad sentence index {s}")
            continue
        a, _b, stext = sents[s]
        occ = int(e.get("occ", 1))
        k = -1
        for _ in range(occ):
            k = stext.find(e["text"], k + 1)
            if k < 0:
                break
        if k < 0 or not e["text"].strip():
            errs.append(f"{eid}: text {e['text']!r} (occ {occ}) not found in sentence {s}")
            continue
        start, end = a + k, a + k + len(e["text"])
        if any(start < y and x < end for x, y in spans):
            errs.append(f"{eid}: overlaps another entity ({e['text']!r})")
            continue
        spans.append((start, end))
        if e["text"] != e["text"].strip():
            errs.append(f"{eid}: leading/trailing space in {e['text']!r}")
        canon = None
        if e["label"] == "Value":
            q = parse_quantity_text(e["text"])
            if q is None or not q.normalized_unit:
                warns.append(f"{eid}: value {e['text']!r} does not parse to a number with a unit (will be dropped)")
        else:
            canon = e.get("canonical") or normalizer().normalize(e["text"], e["label"]).canonical_name
        regions[eid] = {"start": start, "end": end, "text": e["text"], "label": e["label"], "s": s,
                        "canonical_override": e.get("canonical"), "canon": canon}
    rels = []
    for r in ann.get("relations", []):
        a, b, lab = r.get("from"), r.get("to"), r.get("label")
        if lab not in RELS:
            errs.append(f"bad relation label {lab}")
            continue
        if a not in regions or b not in regions:
            errs.append(f"relation {r} refers to unknown entity")
            continue
        subj_ok, obj_ok = RELS[lab]
        if regions[a]["label"] not in subj_ok or regions[b]["label"] not in obj_ok:
            errs.append(f"{lab}: {regions[a]['label']} -> {regions[b]['label']} not allowed "
                        f"({regions[a]['text']!r} -> {regions[b]['text']!r})")
            continue
        if regions[a]["s"] != regions[b]["s"]:
            warns.append(f"{lab} spans sentences {regions[a]['s']} and {regions[b]['s']}: "
                         f"{regions[a]['text']!r} -> {regions[b]['text']!r} (evidence = subject's sentence)")
        rels.append((a, b, lab))
    if verbose:
        for eid, g in regions.items():
            extra = f" -> canonical {g['canon']!r}" if g["canon"] and g["canon"] != g["text"] else ""
            print(f"  {eid} [{g['s']}] {g['label']}: {g['text']!r}{extra}")
        for a, b, lab in rels:
            print(f"  {regions[a]['canon'] or regions[a]['text']} --{lab}--> {regions[b]['canon'] or regions[b]['text']}")
    return regions, rels, errs, warns


def load(stem, chunk):
    p = HERE / "silver_out" / stem / f"{int(chunk)}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cmd_check(stem, chunk, verbose=True):
    ann = load(stem, chunk)
    if ann is None:
        print(f"{stem} {chunk}: MISSING")
        return 1
    _r, rels, errs, warns = resolve(stem, chunk, ann, verbose)
    for w in warns:
        print(f"  WARN {w}")
    for e in errs:
        print(f"  ERROR {e}")
    print(f"{stem} {chunk}: {len(_r)} entities, {len(rels)} relations, {len(errs)} errors, {len(warns)} warnings")
    return 1 if errs else 0


def cmd_checkall(stem=None):
    bad = 0
    for t in TASKS:
        d = t["data"]
        if stem and d["source_file"] != stem:
            continue
        bad += cmd_check(d["source_file"], d["chunk"], verbose=False)
    print(f"files with errors or missing: {bad}")


def cmd_build():
    out, n_missing = [], 0
    for i, t in enumerate(TASKS):
        d = t["data"]
        ann = load(d["source_file"], d["chunk"])
        if ann is None:
            n_missing += 1
            continue
        regions, rels, errs, _w = resolve(d["source_file"], d["chunk"], ann)
        if errs:
            raise SystemExit(f"fix errors first: {d['source_file']} {d['chunk']}: {errs[:3]}")
        result = []
        for eid, g in regions.items():
            rid = f"t{i}_{eid}"
            val = {"start": g["start"], "end": g["end"], "text": g["text"], "labels": [g["label"]]}
            result.append({"id": rid, "from_name": "label", "to_name": "text", "type": "labels", "value": val})
            if g["canonical_override"]:
                result.append({"id": rid, "from_name": "canonical", "to_name": "text", "type": "textarea",
                               "value": {"start": g["start"], "end": g["end"], "text": [g["canonical_override"]]}})
        for a, b, lab in rels:
            result.append({"from_id": f"t{i}_{a}", "to_id": f"t{i}_{b}", "type": "relation",
                           "direction": "right", "labels": [lab]})
        out.append({"id": i + 1, "data": d, "annotations": [{"id": i + 1, "completed_by": "claude-silver",
                                                              "result": result, "was_cancelled": False}]})
    (HERE / "silver_export.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"silver_export.json: {len(out)} annotated tasks, {n_missing} missing")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["list"]
    {"list": cmd_list, "show": cmd_show, "check": cmd_check, "checkall": cmd_checkall, "build": cmd_build}[cmd](*rest)
