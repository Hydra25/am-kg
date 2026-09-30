"""Relaxed relation matching: granularity/wording differences count, wrong facts do not."""
from src.evaluation.metrics import names_compatible, relaxed_relation_prf


def test_names_compatible():
    assert names_compatible("solution treatment", "single solution heat treatment")
    assert names_compatible("strength", "tensile strength")
    assert names_compatible("Ti-6Al-4V", "Ti6Al4V")
    assert not names_compatible("heat treatment", "solution treatment")
    assert not names_compatible("porosity", "hardness")
    assert not names_compatible("α' martensite", "α phase")


def test_relaxed_is_one_to_one_and_never_below_strict():
    gold = [("single solution heat treatment", "IMPROVES", "tensile strength"),
            ("laser power", "INFLUENCES", "porosity"),
            ("LPBF", "PROCESSES", "Ti-6Al-4V")]
    pred = [("solution treatment", "AFFECTS", "tensile strength"),   # same fact, other wording -> match
            ("laser power", "INFLUENCES", "porosity"),                 # exact
            ("laser power", "AFFECTS", "porosity"),                    # duplicate fact: cannot match twice
            ("LPBF", "HAS_PARAMETER", "Ti-6Al-4V")]                    # structural type differs -> no match
    prf = relaxed_relation_prf(gold, pred)
    assert (prf.tp, prf.fp, prf.fn) == (2, 2, 1)
