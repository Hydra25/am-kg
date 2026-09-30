"""Generic process phrases -> the paper's main process (silver-standard error analysis)."""
import pytest

from src.normalization.process_context import is_generic_process


@pytest.mark.parametrize("text", ["printing process", "printing", "fabrication", "the deposition process",
                                  "metal printer", "3D printer", "build process", "fusion process",
                                  "layer-by-layer manufacturing process", "powder-based process",
                                  "multi-laser printer", "additive process"])
def test_generic(text):
    assert is_generic_process(text)


@pytest.mark.parametrize("text", ["remelting", "Marangoni flow", "laser welding", "powder compaction",
                                  "recoating mechanism", "gas atomization process", "hatching strategy",
                                  "laser powder bed fused component"])
def test_not_generic(text):
    assert not is_generic_process(text)


def _analyses(tmp_project, text):
    from src.entity_extraction.rule_based import RuleEntityExtractor
    from src.ingestion.loader import parse_document
    from src.normalization.entity_normalizer import EntityNormalizer
    from src.normalization.lexicon import Lexicon

    md = tmp_project.path("papers_dir") / "generic_process.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")
    doc = parse_document(md, tmp_project)
    lex = Lexicon.load(tmp_project.path("lexicon_file"))
    rx = RuleEntityExtractor(lex, EntityNormalizer(lex))
    return lex, {s.sentence_id: rx.analyze(s, doc.paper.paper_id) for s in doc.sentences}


def _add_process(an, surface):
    from src.ontology.models import EntityMention
    k = an.sentence.text.index(surface)
    an.mentions.append(EntityMention(mention_id=f"m_{surface}", text=surface, entity_type="Process",
                                     canonical_id=f"Process:{surface.lower()}", canonical_name=surface.lower(),
                                     normalization_method="new_entity", normalization_confidence=0.5,
                                     sentence_id=an.sentence.sentence_id, char_start=k, char_end=k + len(surface),
                                     method="gliner"))


TEXT = ("# Generic process test\n\n## Methods\n\nTi-6Al-4V samples were produced by LPBF.\n"
        "LPBF was used for every specimen.\n"
        "The printing process used a laser power of 200 W.\n"
        "Remelting used a laser power of 150 W.\n")


def test_generic_mention_is_remapped_to_main_process(tmp_project):
    from src.normalization.process_context import resolve_generic_processes
    lex, analyses = _analyses(tmp_project, TEXT)
    ans = list(analyses.values())
    _add_process(next(a for a in ans if "printing process" in a.sentence.text), "printing process")
    _add_process(next(a for a in ans if "Remelting" in a.sentence.text), "Remelting")
    assert resolve_generic_processes(analyses, lex)["remapped"] == 1
    procs = {m.text: m.canonical_name for a in ans for m in a.mentions if m.entity_type == "Process"}
    assert procs["printing process"] == "LPBF" and procs["Remelting"] == "remelting"


def test_non_lexicon_process_gets_no_structural_relations(tmp_project):
    from src.relation_extraction.rule_based import RuleRelationExtractor
    lex, analyses = _analyses(tmp_project, TEXT)
    an = next(a for a in analyses.values() if "Remelting" in a.sentence.text)
    _add_process(an, "Remelting")
    rrx = RuleRelationExtractor("p", process_ok=lambda m: m.canonical_id in lex.entities)
    rels = rrx.structural_relations(an, [], [])
    assert not [r for r in rels if r.subject.name == "remelting"]
    rels_all = RuleRelationExtractor("p").structural_relations(an, [], [])
    assert [r for r in rels_all if r.subject.name == "remelting"]      # without the filter it would
