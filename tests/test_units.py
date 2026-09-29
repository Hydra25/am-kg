import pytest

from src.normalization.units import parse_quantities, parse_quantity_text, same_quantity, unit_from_header


@pytest.mark.parametrize("text,value,unit", [
    ("200 W", 200, "W"), ("0.2 kW", 200, "W"), ("30 μm", 30, "μm"), ("30 µm", 30, "μm"), ("0.03 mm", 30, "μm"),
    ("800 mm/s", 800, "mm/s"), ("800 mm s−1", 800, "mm/s"), ("1.2 m/s", 1200, "mm/s"), ("6 m/min", 100, "mm/s"),
    ("60 J/mm³", 60, "J/mm³"), ("60 J mm-3", 60, "J/mm³"), ("950 MPa", 950, "MPa"), ("1.2 GPa", 1200, "MPa"),
    ("1073 K", 799.85, "°C"), ("350 HV0.5", 350, "HV"), ("99.5 %", 99.5, "%"), ("2 h", 7200, "s"),
])
def test_normalized_values(text, value, unit):
    q = parse_quantity_text(text)
    assert q is not None
    assert q.normalized_unit == unit
    assert q.normalized_value == pytest.approx(value, rel=1e-4)
    assert q.raw.replace(" ", "") == text.replace(" ", "")      # original representation kept


def test_equivalence_kw_w_and_um_mm():
    assert same_quantity(parse_quantity_text("200 W"), parse_quantity_text("0.2 kW"))
    assert same_quantity(parse_quantity_text("30 μm"), parse_quantity_text("0.03 mm"))
    assert not same_quantity(parse_quantity_text("30 μm"), parse_quantity_text("30 mm"))


def test_ranges_lists_uncertainty():
    q = parse_quantity_text("150–250 W")
    assert (q.value_min, q.value_max, q.normalized_unit) == (150, 250, "W")
    q = parse_quantities("from 150 W to 250 W")[0].quantity
    assert (q.value_min, q.value_max) == (150, 250)
    q = parse_quantities("speeds between 1000 and 1600 mm/s")[0].quantity
    assert (q.value_min, q.value_max) == (1000, 1600)
    vals = [s.quantity.value for s in parse_quantities("150, 200 and 250 W")]
    assert vals == [150, 200, 250]
    q = parse_quantity_text("870 ± 15 MPa")
    assert q.value == 870 and q.uncertainty == 15


def test_no_false_units_in_words():
    assert parse_quantities("the current study used 3 samples") == []
    assert parse_quantities("Table 1 shows") == []


def test_header_unit():
    assert unit_from_header("Laser power (W)") == "W"
    assert unit_from_header("Scan speed [mm/s]") == "mm/s"
    assert unit_from_header("Specimen") is None
