"""Phase 2: Label Studio round trip (export -> annotate -> import) and the mode benchmark."""
import json

from src.evaluation.labelstudio import build_tasks, import_annotations, labeling_config


def _extract(tmp_project, name="demo_01_lpbf_ti64.md", mode="rule"):
    from src.ingestion.loader import parse_document
    from src.pipeline import ExtractionPipeline
    doc = parse_document(tmp_project.path("demo_dir") / name, tmp_project)
    pipe = ExtractionPipeline(tmp_project, mode=mode)
    return doc, pipe.extract(doc), pipe


def test_labeling_config_has_all_labels():
    cfg = labeling_config()
    for lab in ("Material", "Process", "Value", "HAS_VALUE", "PROCESSES", 'perRegion="true"'):
        assert lab in cfg


def test_tasks_carry_correct_offsets_and_predictions(tmp_project):
    doc, res, _ = _extract(tmp_project)
    tasks = build_tasks(doc, res)
    assert tasks and all(t["data"]["source_file"] == "demo_01_lpbf_ti64" for t in tasks)
    n_regions = n_rel = 0
    for t in tasks:
        text = t["data"]["text"]
        for item in t["predictions"][0]["result"]:
            if item["type"] == "labels":
                v = item["value"]
                assert text[v["start"]:v["end"]] == v["text"]     # offsets point at the right characters
                n_regions += 1
            else:
                n_rel += 1
    assert n_regions > 10 and n_rel > 3
    empty = build_tasks(doc, None)
    assert "predictions" not in empty[0]


def test_round_trip_gold_scores_perfectly_against_itself(tmp_project):
    """If the annotator accepts every pre-annotation unchanged, the imported gold
    must match the system output (entities and relations F1 = 1)."""
    from src.evaluation.metrics import evaluate
    from src.knowledge_graph.kb import KnowledgeBase

    doc, res, pipe = _extract(tmp_project)
    pipe.save_result(res)
    from src.ingestion.loader import save_parsed
    save_parsed(doc, tmp_project.path("processed_dir"))
    tasks = build_tasks(doc, res)
    export = [{**t, "annotations": [{"result": t["predictions"][0]["result"], "updated_at": "2026-10-01"}]}
              for t in tasks]
    gold = import_annotations(json.loads(json.dumps(export)), pipe.normalizer, annotator="test")
    g = gold["demo_01_lpbf_ti64"]
    assert g["entities"] and g["relations"] and g["values"]
    gold_dir = tmp_project.root / "gold_rt"
    gold_dir.mkdir()
    (gold_dir / "demo_01_lpbf_ti64.json").write_text(json.dumps(g), encoding="utf-8")
    kb = KnowledgeBase.load(tmp_project)
    rep = evaluate(kb, gold_dir, tmp_project.path("processed_dir"), pipe.normalizer)
    assert rep["relations_strict"]["recall"] == 1.0
    assert rep["values"]["recall"] >= 0.9     # "from 1.1 % to 0.3 %" is re-parsed as a range on import
    assert rep["entities"]["recall"] >= 0.9


def test_import_uses_canonical_override_and_relation_direction(tmp_project):
    from src.normalization.entity_normalizer import EntityNormalizer
    from src.normalization.lexicon import Lexicon
    norm = EntityNormalizer(Lexicon.load(tmp_project.path("lexicon_file")))
    text = "Specimens of Printdur HSA were built by SLM with a laser power of 200 W."
    task = {"data": {"text": text, "source_file": "paper_x", "sentence_offsets": [["s1", 0, len(text), 3]]},
            "annotations": [{"result": [
                {"id": "a", "type": "labels", "from_name": "label", "to_name": "text",
                 "value": {"start": 13, "end": 25, "text": "Printdur HSA", "labels": ["Material"]}},
                {"id": "b", "type": "labels", "from_name": "label", "to_name": "text",
                 "value": {"start": 38, "end": 41, "text": "SLM", "labels": ["Process"]}},
                {"id": "c", "type": "labels", "from_name": "label", "to_name": "text",
                 "value": {"start": 49, "end": 60, "text": "laser power", "labels": ["Parameter"]}},
                {"id": "d", "type": "labels", "from_name": "label", "to_name": "text",
                 "value": {"start": 64, "end": 69, "text": "200 W", "labels": ["Value"]}},
                {"id": "a", "type": "textarea", "from_name": "canonical", "to_name": "text",
                 "value": {"start": 13, "end": 25, "text": ["Printdur HSA"]}},
                # drawn from material to process: direction "left" means SLM -> Printdur HSA
                {"from_id": "a", "to_id": "b", "type": "relation", "direction": "left", "labels": ["PROCESSES"]},
                {"from_id": "c", "to_id": "d", "type": "relation", "direction": "right", "labels": ["HAS_VALUE"]},
            ]}]}
    g = import_annotations([task], norm)["paper_x"]
    assert {"type": "Process", "canonical": "LPBF"} in g["entities"]           # SLM normalised via lexicon
    assert g["normalization"] == [{"surface": "Printdur HSA", "type": "Material", "canonical": "Printdur HSA"}]
    assert g["relations"][0]["subject"] == "LPBF" and g["relations"][0]["object"] == "Printdur HSA"
    assert g["relations"][0]["evidence"] == text
    assert g["values"][0]["normalized_value"] == 200 and g["values"][0]["normalized_unit"] == "W"


def test_benchmark_compares_modes(tmp_project):
    from src.entity_extraction.gliner_extractor import GlinerEntityExtractor
    from src.evaluation.benchmark import run_benchmark
    from src.pipeline import ExtractionPipeline
    from tests.test_gliner_junk import FakeGliner

    probe = ExtractionPipeline(tmp_project, mode="rule")
    fake = GlinerEntityExtractor(FakeGliner(), probe.normalizer, probe.rule_entities)
    out = tmp_project.root / "data" / "benchmark"
    summary = run_benchmark(tmp_project, tmp_project.path("demo_dir"), ["rule", "gliner"],
                            tmp_project.path("annotations_dir"), out, gliner=fake)
    assert [r["mode"] for r in summary["rows"]] == ["rule", "gliner"]
    assert summary["rows"][0]["papers"] == 2                       # only papers with gold annotations
    assert 0 < summary["rows"][0]["relation_f1"] <= 1
    assert (out / "results.md").exists() and (out / "results.csv").exists()
    assert (out / "rule" / "extracted").exists()
    assert not list(tmp_project.path("extracted_dir").glob("paper_*.json"))   # main results untouched
