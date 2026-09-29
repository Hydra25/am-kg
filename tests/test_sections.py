from src.section_parser.sections import Line, build_sections, classify_heading
from src.section_parser.sentences import split_sentences


def test_classify_headings():
    assert classify_heading("2. Materials and methods").value == "materials_methods"
    assert classify_heading("3 Results and discussion").value == "results_discussion"
    assert classify_heading("Experimental results").value == "results"
    assert classify_heading("5. Conclusions").value == "conclusion"
    assert classify_heading("References").value == "references"
    assert classify_heading("2.1 Powder") is None


def test_sentence_splitter_keeps_abbreviations():
    t = "Specimens (Fig. 2) were built by Smith et al. in 2020. Laser power of 200 W was used, i.e. the max. value. Porosity was 0.5 %."
    parts = [t[a:b] for a, b in split_sentences(t)]
    assert len(parts) == 3 and parts[1].startswith("Laser power")


def test_sections_pages_and_subsection_inheritance():
    lines = [Line("Abstract", 1, 12, True), Line("We printed parts.", 1),
             Line("2. Materials and methods", 1, 12, True), Line("2.1 Powder", 2, 11, True),
             Line("The powder was fabri-", 2), Line("cated by gas atomisation.", 2),
             Line("References", 3, 12, True), Line("[1] A. Author.", 3)]
    body = [ln for ln in lines]
    for ln in body:
        if ln.size == 10.0:
            ln.size = 10.0
    secs, sents = build_sections(body, "p")
    types = [s.section_type for s in secs]
    assert types == ["abstract", "materials_methods", "references"]
    powder = [s for s in sents if "powder" in s.text][0]
    assert powder.page == 2 and "fabricated" in powder.text   # de-hyphenation across lines


def test_publisher_layout_headings_and_titles():
    """Seen on real Elsevier / MDPI / Nature PDFs of the LPBF benchmark."""
    from src.ingestion.loader import _plausible_title
    from src.ontology.schema import SectionType as S
    from src.section_parser.sections import classify_heading
    assert classify_heading("A B S T R A C T") == S.ABSTRACT          # Elsevier letter-spaced heading
    assert classify_heading("A R T I C L E  I N F O") == S.FRONT_MATTER
    # a keyword deep inside a long line is not a section type
    assert classify_heading("Additive manufacturing (AM), contrary to the conventional machining methods") is None
    assert not _plausible_title("Journal of Materials Research and Technology")
    assert not _plausible_title("Materials & Design")
    assert _plausible_title("Effect of process parameters on the mechanical behavior of Ti6Al4V alloys")
