"""Fixes found while building the silver standard on 10 real LPBF papers."""
import pytest

from src.normalization.entity_normalizer import EntityNormalizer, find_symbol_definitions, fuzzy_compatible
from src.normalization.lexicon import Lexicon
from src.normalization.units import parse_quantity_text


@pytest.mark.parametrize("text,value,unit", [
    ("920 ◦C", 920.0, "°C"),          # PDF degree look-alike
    ("67◦", 67.0, "°"),
    ("80 lm", 80.0, "μm"),            # Elsevier: μm extracted as "lm"
    ("104 VHN", 104.0, "HV"),
    ("1‰", 0.1, "%"),
    ("300 KV", 300000.0, "V"),
    ("155 J/ mm3", 155.0, "J/mm³"),
    ("500 g", 500.0, "g"),
])
def test_pdf_unit_variants(text, value, unit):
    q = parse_quantity_text(text)
    assert q is not None and q.normalized_unit == unit and q.normalized_value == pytest.approx(value)


def test_cooling_rate_and_small_values():
    q = parse_quantity_text("20 ◦C/min")
    assert q.normalized_unit == "°C/s" and q.normalized_value == pytest.approx(1 / 3, rel=1e-3)
    assert parse_quantity_text("1.4 nA").normalized_value == pytest.approx(1.4e-9)


@pytest.mark.parametrize("text,lo,hi", [("21e37 J/mm3", 21, 37), ("(200e716) MPa", 200, 716), ("(4e42)%", 4, 42)])
def test_en_dash_extracted_as_e_is_a_range(text, lo, hi):
    q = parse_quantity_text(text)
    assert (q.normalized_min, q.normalized_max) == (lo, hi)


def test_scientific_notation_is_not_a_range():
    assert parse_quantity_text("1e5 cycles") is None


@pytest.mark.parametrize("term,cand,ok", [
    ("toughness", "surface roughness", False),
    ("toughness", "roughness", False),
    ("α′ phase", "α phase", False),
    ("macrostructure", "microstructure", False),
    ("α+β microstructure", "microstructure", False),
    ("X-ray Computedd Tomography", "X-ray computed tomography", True),
    ("scaning speed", "scanning speed", True),
    ("key hole", "keyhole", True),
])
def test_fuzzy_guard(term, cand, ok):
    assert fuzzy_compatible(term, cand) is ok


def test_paper_symbols_resolve_only_when_unambiguous(tmp_project):
    norm = EntityNormalizer(Lexicon.load(tmp_project.path("lexicon_file")))
    text = ("The energy density was E = P/(v h t), where P is the laser power, v is the scanning speed and "
            "t is the layer thickness. Hatch distance (h) was 100 μm; h is the layer height in some tables. "
            "It is a common process.")
    norm.set_paper_context({}, find_symbol_definitions(text))
    assert norm.normalize("P", "Parameter").canonical_name == "laser power"
    assert norm.normalize("v", "Parameter").canonical_name == "scanning speed"
    assert norm.normalize("t", "Parameter").canonical_name == "layer thickness"
    assert norm.normalize("It", "Parameter").method == "new_entity"
    norm.set_paper_context()                       # next paper: no context
    assert norm.normalize("P", "Parameter").method == "new_entity"


def test_fuzzy_guard_allows_location_qualifiers():
    assert fuzzy_compatible("top surface roughness", "surface roughness")
    assert fuzzy_compatible("side-surface roughness", "surface roughness")
    assert not fuzzy_compatible("surface roughness", "top surface roughness")


def test_descending_range_is_sorted():
    q = parse_quantity_text("22 to 2 s")
    assert (q.normalized_min, q.normalized_max) == (2, 22)
