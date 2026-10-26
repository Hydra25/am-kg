"""Option B: open-access harvester filters and the auto-lexicon rule v1 (no network, fake LLM)."""
import json
import shutil
from pathlib import Path

from scripts.harvest_oa import abstract_text, judge
from scripts.lexicon_discovery import Candidate, TermChecker, aggregate, group_key, pre_classify, write_auto_lexicon
from src.llm.client import FakeLLM
from src.normalization.lexicon import AUTO_CONFIDENCE, Lexicon

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ harvester
def _work(doi="10.3390/x1", url="https://www.mdpi.com/x.pdf", lic="cc-by", title="Laser powder bed fusion of 316L"):
    return {"doi": f"https://doi.org/{doi}", "title": title, "publication_year": 2024,
            "best_oa_location": {"pdf_url": url, "license": lic, "source": {"display_name": "Materials"}},
            "abstract_inverted_index": {"steel": [0], "porosity": [1]}}


def test_harvester_rules():
    assert judge(_work(), set())[1] == ""
    assert judge(_work(), {"10.3390/x1"})[1] == "benchmark_paper"                 # DEV/TEST never harvested
    assert judge(_work(url="https://www.sciencedirect.com/a.pdf"), set())[1] == "blocked_host"
    assert judge(_work(lic="publisher-specific-oa"), set())[1] == "licence_not_open"
    assert judge(_work(url=""), set())[1] == "no_pdf_link"
    assert judge(_work(title="Fused deposition of PLA"), set())[1] == "not_lpbf"
    assert abstract_text({"world": [1], "hello": [0]}) == "hello world"


# ------------------------------------------------------------------ grouping + rule v1
def test_group_key_merges_variants():
    assert group_key("Weak fusion defects") == group_key("weak fusion")
    assert group_key("surface tensions") == group_key("Surface tension")
    assert group_key("porosities") == group_key("porosity")


def _papers(term, typ, materials, extra_type=None):
    out = []
    for i, mat in enumerate(materials):
        t = extra_type if (extra_type and i == 0) else typ
        out.append({"paper": f"p{i}.pdf", "material": mat,
                    "mentions": [{"text": term, "type": t, "canonical": term, "sentence": f"{term} in paper {i}."}]})
    return out


def test_rule_v1_tiers(lexicon):
    ok = aggregate(_papers("surface tension", "Property", ["Ti-6Al-4V", "316L", "AlSi10Mg"]))
    c = next(iter(ok.values()))
    pre_classify(c, lexicon, {})
    assert c.tier == "llm"                                              # passes steps 1-4

    few = next(iter(aggregate(_papers("surface tension", "Property", ["316L", "IN718"])).values()))
    pre_classify(few, lexicon, {})
    assert few.tier == "watch" and few.reason.startswith("papers_")

    one_mat = next(iter(aggregate(_papers("surface tension", "Property", ["316L"] * 3)).values()))
    pre_classify(one_mat, lexicon, {})
    assert one_mat.tier == "watch" and one_mat.reason.startswith("materials_")

    mixed = next(iter(aggregate(_papers("melt flow", "Property", ["316L", "IN718", "Ti-6Al-4V"],
                                        extra_type="Microstructure")).values()))
    pre_classify(mixed, lexicon, {})
    assert mixed.tier == "watch" and mixed.reason.startswith("type_share")

    known = next(iter(aggregate(_papers("porosity", "Defect", ["316L", "IN718", "Ti-6Al-4V"])).values()))
    pre_classify(known, lexicon, {})
    assert known.tier == "known"

    abbr = next(iter(aggregate(_papers("BSE", "CharacterizationMethod", ["316L", "IN718", "Ti-6Al-4V"])).values()))
    pre_classify(abbr, lexicon, {})
    assert abbr.tier == "watch" and abbr.reason == "abbreviation_never_auto"


def test_checker_and_auto_file(tmp_path, lexicon):
    cands = [Candidate("surfacetension"), Candidate("unfusedpowderparticle"), Candidate("certainthing")]
    for c, (name, typ) in zip(cands, [("surface tension", "Property"), ("unfused powder particle", "Defect"),
                                      ("certain thing", "Property")]):
        c.names[name] = 3
        c.types[typ] = 3
        c.papers = {"a.pdf", "b.pdf", "c.pdf"}
        c.materials = {"316L", "IN718"}
        c.examples = [("a.pdf", f"The {name} was measured.")]
    answer = json.dumps({"decisions": [{"id": 1, "concept": True, "type": "Property", "synonym_of": None},
                                       {"id": 2, "concept": True, "type": "Defect", "synonym_of": "unmelted powder"},
                                       {"id": 3, "concept": False, "type": "Property", "synonym_of": None}]})
    checker = TermChecker(FakeLLM([answer]), tmp_path / "verdicts.json")
    checker.check(cands, lexicon)
    assert [c.tier for c in cands] == ["auto", "alias", "block"]
    assert cands[1].target == "unmelted powder"
    assert (tmp_path / "verdicts.json").exists()                         # cached: no second call next time

    cfg = tmp_path / "config"
    cfg.mkdir()
    shutil.copy(ROOT / "config" / "lexicon.yaml", cfg / "lexicon.yaml")
    added = write_auto_lexicon(cfg / "lexicon_auto.yaml", cands, lexicon, "fake", {"a.pdf": "10.1/a"})
    assert added == (1, 1)
    lex = Lexicon.load(cfg / "lexicon.yaml", auto=True)
    hit = lex.lookup("surface tension", "Property")
    assert hit and hit[1] == AUTO_CONFIDENCE                            # new concept, lower confidence
    assert lex.lookup("unfused powder particle", "Defect")[0] == "Defect:unmelted powder"   # new alias
    assert lex.entities["Property:surface tension"].attributes.get("auto") is True
    assert Lexicon.load(cfg / "lexicon.yaml", auto=False).lookup("surface tension", "Property") is None


def test_harvester_uses_repository_copies():
    w = _work(url="https://www.sciencedirect.com/a.pdf")
    w["locations"] = [{"is_oa": True, "pdf_url": "https://europepmc.org/articles/PMC1/pdf"},
                      {"is_oa": False, "pdf_url": "https://paywalled.example/x.pdf"}]
    row, reason = judge(w, set())
    assert reason == "" and row["pdf_url"] == "https://europepmc.org/articles/PMC1/pdf" and row["alt_urls"] == []
