"""Unit recognition and normalisation (no external dependency such as pint).

Every quantity keeps its original text/unit and additionally gets a value in
the canonical unit of its physical dimension, e.g.

    0.2 kW  -> 200 W          30 um / 0.03 mm -> 30 um
    1 m/min -> 16.67 mm/s     1073 K          -> 799.85 degC

Canonical units were chosen to match AM reporting conventions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..ontology.models import Quantity
from .text import normalize_chars


@dataclass(frozen=True)
class UnitDef:
    symbol: str          # canonical spelling of this unit
    dimension: str
    factor: float        # multiply to get canonical unit of the dimension
    offset: float = 0.0  # added after multiplication (temperature)


CANONICAL_UNIT: dict[str, str] = {
    "power": "W", "velocity": "mm/s", "length": "μm", "temperature": "°C",
    "stress": "MPa", "energy_density_vol": "J/mm³", "energy_density_lin": "J/mm",
    "energy_density_area": "J/mm²", "energy": "J", "percent": "%", "ppm": "ppm",
    "hardness": "HV", "hardness_hrc": "HRC", "mass_density": "g/cm³", "time": "s",
    "current": "A", "voltage": "V", "volume_flow": "L/min", "mass_flow": "g/min",
    "frequency": "Hz", "angle": "°", "cycles": "cycles",
    "thermal_conductivity": "W/(m·K)", "fracture_toughness": "MPa·m^0.5",
    "rotation": "rpm", "cooling_rate": "°C/s", "mass": "g",
}


def _ratio_variants(num: str, den: str, power: str = "") -> list[str]:
    """All common spellings of num/den^power, e.g. J/mm3, J mm-3, J.mm-3, J/mm^3."""
    p = power
    out = [f"{num}/{den}{p}", f"{num} / {den}{p}", f"{num}/ {den}{p}", f"{num} /{den}{p}", f"{num}/{den}^{p}" if p else f"{num}/{den}",
           f"{num} {den}-{p or 1}", f"{num}.{den}-{p or 1}", f"{num} {den}^-{p or 1}",
           f"{num}{den}-{p or 1}"]
    if not p:
        out += [f"{num} {den}-1", f"{num}.{den}-1"]
    return list(dict.fromkeys(out))


def _build_table() -> dict[str, UnitDef]:
    t: dict[str, UnitDef] = {}

    def add(variants: list[str], dim: str, factor: float, offset: float = 0.0, symbol: str | None = None) -> None:
        sym = symbol or variants[0]
        for v in variants:
            t[v] = UnitDef(sym, dim, factor, offset)

    add(["W"], "power", 1.0)
    add(["kW"], "power", 1000.0)
    add(["mW"], "power", 0.001)
    add(_ratio_variants("mm", "s"), "velocity", 1.0, symbol="mm/s")
    add(_ratio_variants("m", "s"), "velocity", 1000.0, symbol="m/s")
    add(_ratio_variants("cm", "s"), "velocity", 10.0, symbol="cm/s")
    add(_ratio_variants("m", "min"), "velocity", 1000.0 / 60.0, symbol="m/min")
    add(_ratio_variants("mm", "min"), "velocity", 1.0 / 60.0, symbol="mm/min")
    add(_ratio_variants("cm", "min"), "velocity", 10.0 / 60.0, symbol="cm/min")
    add(["in/s", "ips"], "velocity", 25.4, symbol="in/s")
    add(["in/min", "ipm"], "velocity", 25.4 / 60.0, symbol="in/min")
    # "lm": Elsevier PDFs often extract the μ of "μm" as "l" (e.g. "30 lm")
    add(["μm", "um", "micron", "microns", "μ m", "lm"], "length", 1.0, symbol="μm")
    add(["nm"], "length", 0.001)
    add(["mm"], "length", 1000.0)
    add(["cm"], "length", 10000.0)
    add(["°C", "° C", "℃", "oC", "degC", "deg C"], "temperature", 1.0, symbol="°C")
    add(["K"], "temperature", 1.0, offset=-273.15)
    add(["MPa", "N/mm2", "N/mm^2", "N mm-2"], "stress", 1.0, symbol="MPa")
    add(["GPa"], "stress", 1000.0)
    add(["kPa"], "stress", 0.001)
    add(["ksi"], "stress", 6.894757)
    add(["psi"], "stress", 0.006894757)
    add(_ratio_variants("J", "mm", "3"), "energy_density_vol", 1.0, symbol="J/mm³")
    add(_ratio_variants("J", "cm", "3"), "energy_density_vol", 0.001, symbol="J/cm³")
    add(_ratio_variants("J", "mm", "2"), "energy_density_area", 1.0, symbol="J/mm²")
    add(_ratio_variants("J", "mm"), "energy_density_lin", 1.0, symbol="J/mm")
    add(_ratio_variants("kJ", "mm"), "energy_density_lin", 1000.0, symbol="kJ/mm")
    add(_ratio_variants("J", "m"), "energy_density_lin", 0.001, symbol="J/m")
    add(_ratio_variants("kJ", "m"), "energy_density_lin", 1.0, symbol="kJ/m")
    add(["J"], "energy", 1.0)
    add(["kJ"], "energy", 1000.0)
    add(["‰"], "percent", 0.1, symbol="‰")
    add(["%", "pct", "vol%", "vol.%", "vol. %", "wt%", "wt.%", "wt. %", "at.%"], "percent", 1.0, symbol="%")
    add(["ppm"], "ppm", 1.0)
    add(["HV", "Hv", "VHN", "HVN", "HV0.05", "HV0.1", "HV0.2", "HV0.3", "HV0.5", "HV1", "HV5", "HV10", "HV 0.1", "HV 0.5"], "hardness", 1.0, symbol="HV")
    add(["HRC"], "hardness_hrc", 1.0)
    add(_ratio_variants("g", "cm", "3") + ["g/cc"], "mass_density", 1.0, symbol="g/cm³")
    add(_ratio_variants("kg", "m", "3"), "mass_density", 0.001, symbol="kg/m³")
    add(["s", "sec", "seconds"], "time", 1.0, symbol="s")
    add(["ms"], "time", 0.001)
    add(["min", "mins", "minutes"], "time", 60.0, symbol="min")
    add(["h", "hr", "hrs", "hours", "hour"], "time", 3600.0, symbol="h")
    add(["A"], "current", 1.0)
    add(["mA"], "current", 0.001)
    add(["μA", "uA"], "current", 1e-6, symbol="μA")
    add(["nA"], "current", 1e-9)
    add(["V"], "voltage", 1.0)
    add(["kV", "KV"], "voltage", 1000.0, symbol="kV")
    add(_ratio_variants("°C", "s") + _ratio_variants("K", "s"), "cooling_rate", 1.0, symbol="°C/s")
    add(_ratio_variants("°C", "min") + _ratio_variants("K", "min"), "cooling_rate", 1.0 / 60.0, symbol="°C/min")
    add(_ratio_variants("°C", "h") + _ratio_variants("K", "h"), "cooling_rate", 1.0 / 3600.0, symbol="°C/h")
    add(["g"], "mass", 1.0)
    add(["kg"], "mass", 1000.0)
    add(["mg"], "mass", 0.001)
    add(_ratio_variants("L", "min") + _ratio_variants("l", "min") + ["lpm"], "volume_flow", 1.0, symbol="L/min")
    add(_ratio_variants("g", "min"), "mass_flow", 1.0, symbol="g/min")
    add(_ratio_variants("g", "s"), "mass_flow", 60.0, symbol="g/s")
    add(["Hz"], "frequency", 1.0)
    add(["kHz"], "frequency", 1000.0)
    add(["°", "deg", "degrees", "degree"], "angle", 1.0, symbol="°")
    add(["cycles", "cycle"], "cycles", 1.0, symbol="cycles")
    add(["W/mK", "W/(m K)", "W/(m.K)", "W/m.K", "W/(mK)", "W m-1 K-1", "W.m-1.K-1", "W.m-1 K-1"],
        "thermal_conductivity", 1.0, symbol="W/(m·K)")
    add(["MPa√m", "MPa m1/2", "MPa.m1/2", "MPa m^0.5", "MPa.m^0.5", "MPa m^1/2", "MPa.m^1/2", "MPa m0.5"],
        "fracture_toughness", 1.0, symbol="MPa·m^0.5")
    add(["rpm"], "rotation", 1.0)
    return t


UNIT_TABLE: dict[str, UnitDef] = _build_table()
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNIT_TABLE, key=len, reverse=True))

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
_SCI = r"(?:\s*x\s*10\s*\^?\s*(?P<{name}>-?\d+))"
_QTY_RE = re.compile(
    r"(?P<op><=|>=|≤|≥|<|>|~|≈)?\s*"
    r"(?P<sign>(?<![\w.)\]])-)?"
    r"(?P<lp>\()?"
    rf"(?<![\w.])(?P<n1>{_NUM})" + _SCI.format(name="e1") + "?"
    rf"(?:\s*(?:±|\+/-|\+-)\s*(?P<unc>{_NUM}))?"
    rf"(?:(?:\s*(?:-|to|~)\s*|(?P<esep>e))(?P<n2>{_NUM})" + _SCI.format(name="e2") + "?)?"
    r"(?(lp)\))"
    rf"\s?(?P<unit>{_UNIT_ALT})(?![A-Za-z0-9μ])"
)
# "150, 200 and 250 W"  -> three quantities
_LIST_RE = re.compile(
    rf"(?<![\w.])(?P<nums>(?:{_NUM})(?:\s*,\s*(?:{_NUM}))*,?\s*(?:and|or)\s*(?:{_NUM}))\s?(?P<unit>{_UNIT_ALT})(?![A-Za-z0-9μ])"
)
_HEADER_UNIT_RE = re.compile(rf"[\(\[]\s*(?P<unit>{_UNIT_ALT})\s*[\)\]]")


def _num(s: Optional[str], exp: Optional[str] = None) -> Optional[float]:
    if s is None:
        return None
    v = float(s.replace(",", ""))
    if exp is not None:
        v *= 10 ** int(exp)
    return v


def lookup_unit(unit: str | None) -> Optional[UnitDef]:
    if not unit:
        return None
    u = normalize_chars(unit).strip()
    if u in UNIT_TABLE:
        return UNIT_TABLE[u]
    compact = re.sub(r"\s+", " ", u)
    return UNIT_TABLE.get(compact) or UNIT_TABLE.get(compact.replace("^", ""))


def _round(v: float) -> float:
    """6 decimals, but keep small values (1.4 nA = 1.4e-9 A) instead of rounding them to 0."""
    return round(v, 6) if abs(v) >= 1e-3 or v == 0 else float(f"{v:.6g}")


def normalize_quantity(q: Quantity) -> Quantity:
    """Fill normalized_* fields in-place (and return q). Original fields untouched."""
    ud = lookup_unit(q.unit)
    if ud is None:
        return q
    q.dimension = ud.dimension
    q.normalized_unit = CANONICAL_UNIT.get(ud.dimension, ud.symbol)

    def conv(v: Optional[float]) -> Optional[float]:
        return None if v is None else _round(v * ud.factor + ud.offset)

    q.normalized_value = conv(q.value)
    q.normalized_min = conv(q.value_min)
    q.normalized_max = conv(q.value_max)
    q.normalized_uncertainty = None if q.uncertainty is None else _round(q.uncertainty * ud.factor)
    return q


@dataclass
class QuantitySpan:
    start: int
    end: int
    quantity: Quantity


def parse_quantities(text: str) -> list[QuantitySpan]:
    """Find all value+unit expressions in `text` (offsets refer to `text`)."""
    norm = normalize_chars(text)
    spans: list[QuantitySpan] = []
    taken: list[tuple[int, int]] = []

    for m in _LIST_RE.finditer(norm):
        nums = re.findall(_NUM, m.group("nums"))
        before = norm[max(0, m.start() - 9):m.start()].lower()
        if len(nums) == 2 and re.search(r"between\s*$", before):
            # "between 1000 and 1600 mm/s" is a range, not a list
            raw = text[m.start():m.end()]
            q = Quantity(raw=raw, value_min=_num(nums[0]), value_max=_num(nums[1]),
                         unit=text[m.start("unit"):m.end("unit")], operator="between")
            spans.append(QuantitySpan(m.start(), m.end(), normalize_quantity(q)))
            taken.append((m.start(), m.end()))
            continue
        for nm in re.finditer(_NUM, m.group("nums")):
            s = m.start("nums") + nm.start()
            e = m.start("nums") + nm.end()
            raw = f"{text[s:e]} {text[m.start('unit'):m.end('unit')]}"
            q = Quantity(raw=raw, value=_num(nm.group()), unit=text[m.start("unit"):m.end("unit")])
            spans.append(QuantitySpan(s, e, normalize_quantity(q)))
        taken.append((m.start(), m.end()))

    for m in _QTY_RE.finditer(norm):
        if any(a <= m.start("n1") < b for a, b in taken):
            continue
        if m.group("esep") and (m.group("sign") or len(m.group("n2")) < 2
                                or float(m.group("n2").replace(",", "")) <= float(m.group("n1").replace(",", ""))):
            continue    # "1e5" is scientific notation, not a PDF-mangled "1–5" range
        start = m.start("op") if m.group("op") else (m.start("sign") if m.group("sign") else
                                                     (m.start("lp") if m.group("lp") else m.start("n1")))
        raw = text[start:m.end()].strip()
        sign = -1.0 if m.group("sign") else 1.0
        n1 = _num(m.group("n1"), m.group("e1"))
        n2 = _num(m.group("n2"), m.group("e2"))
        q = Quantity(raw=raw, unit=text[m.start("unit"):m.end("unit")], operator=m.group("op"))
        if n2 is not None:
            q.value_min, q.value_max = sorted((sign * n1, n2))  # type: ignore[operator]  # "22 to 2 s"
            q.value = None
        else:
            q.value = sign * n1 if n1 is not None else None
        q.uncertainty = _num(m.group("unc"))
        spans.append(QuantitySpan(start, m.end(), normalize_quantity(q)))

    spans.sort(key=lambda s: s.start)
    return _merge_from_to_ranges(text, spans)


def _merge_from_to_ranges(text: str, spans: list[QuantitySpan]) -> list[QuantitySpan]:
    """'from 150 W to 250 W' / 'between 150 W and 250 W' -> one range quantity."""
    out: list[QuantitySpan] = []
    i = 0
    while i < len(spans):
        a = spans[i]
        if i + 1 < len(spans):
            b = spans[i + 1]
            gap = text[a.end:b.start]
            before = text[max(0, a.start - 9):a.start].lower()
            same_unit = a.quantity.unit == b.quantity.unit
            if (same_unit and re.fullmatch(r"\s*(to|and|-)\s*", gap)
                    and re.search(r"(from|between)\s*$", before)
                    and a.quantity.value is not None and b.quantity.value is not None):
                lo, hi = sorted([a.quantity.value, b.quantity.value])
                q = Quantity(raw=text[a.start:b.end], unit=a.quantity.unit, value_min=lo, value_max=hi,
                             operator="from_to" if before.strip().endswith("from") else "between")
                out.append(QuantitySpan(a.start, b.end, normalize_quantity(q)))
                i += 2
                continue
        out.append(a)
        i += 1
    return out


def parse_quantity_text(text: str) -> Optional[Quantity]:
    """Parse a standalone value string such as '200 W' or '0.03 mm'."""
    spans = parse_quantities(text)
    return spans[0].quantity if spans else None


def unit_from_header(header: str) -> Optional[str]:
    """Return the unit declared in a table header like 'Laser power (W)'."""
    m = _HEADER_UNIT_RE.search(normalize_chars(header))
    return m.group("unit") if m else None


def same_quantity(a: Quantity, b: Quantity, rel_tol: float = 1e-6) -> bool:
    """True if two quantities denote the same physical value (e.g. 0.2 kW == 200 W)."""
    if a.dimension is None or a.dimension != b.dimension:
        return False
    va, vb = a.normalized_value, b.normalized_value
    if va is None or vb is None:
        return False
    return abs(va - vb) <= rel_tol * max(abs(va), abs(vb), 1e-12)
