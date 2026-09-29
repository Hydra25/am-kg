"""Filter obviously wrong entity proposals from the LLM or GLiNER.

Seen on real papers: a trade-fair venue tagged as a characterization method
("fiera milano"), colours ("blue"), sub-figure labels ("Fig. 4 (e"), specimen
IDs ("LT1") and bare numbers. Terms that are in the lexicon are always trusted;
this filter only judges NEW terms, and it records why a term was dropped so the
audit trail shows it.
"""
from __future__ import annotations

import re
from typing import Optional

from ..normalization.text import normalize_chars
from ..normalization.units import parse_quantity_text

_GENERIC_WORDS = {
    "blue", "red", "green", "black", "white", "yellow", "orange", "grey", "gray",
    "left", "right", "top", "bottom", "middle", "center", "centre",
    "fig", "figure", "figures", "table", "tables", "section", "equation",
    "sample", "samples", "specimen", "specimens", "part", "parts", "component", "components",
    "result", "results", "value", "values", "data", "study", "paper", "work", "case", "cases",
    "method", "methods", "material", "materials", "process", "processes", "parameter",
    "parameters", "property", "properties", "test", "tests", "condition", "conditions",
    "area", "region", "image", "images", "line", "lines", "point", "points", "side", "type",
    "types", "level", "levels", "group", "series", "set", "range", "et al", "i.e", "e.g",
    "manufacturing", "production", "behaviour", "behavior", "response", "structure", "structures",
    # generic AM / materials phrases that name no specific thing (seen from GLiNER on a real paper)
    "am", "additive manufacturing", "additively manufactured", "am process", "am processes",
    "manufacturing process", "manufacturing processes", "am material", "am materials", "am component",
    "am components", "am application", "am applications", "base material", "base materials",
    "defect", "defects", "strength", "mechanical properties", "material properties", "process parameters", "applied process parameters", "geometry", "specimen geometry", "feature",
    "features", "design", "designs", "steel", "steels", "metal", "metals", "alloy", "alloys",
    "welded specimens", "specimen width", "preliminary trials", "limited reproducibility",
}
# Words that make a phrase plausible as a characterization method.
_METHOD_CUES = re.compile(
    r"(test|testing|microscop|spectroscop|spectromet|tomograph|diffract|analys|measur|imag|"
    r"correlation|scann|metallograph|dilatomet|calorimet|fractograph|profilomet|indent|inspect|"
    r"simulat|model|camera|gauge|sensor|thermograph|ultrason|radiograph|hardness|tensile|fatigue|"
    r"charpy|impact|x-ray|laser|optical|probe|\b(SEM|TEM|EBSD|XRD|XCT|CT|DIC|EDS|EDX|EDXS|DSC|OM|"
    r"FEM|FEA|EPMA|XPS|AFM|DMA|TGA|FTIR)\b)", re.I)
_FIG_REF = re.compile(r"\b(fig|figs|figure|table|tab|eq|eqs|ref|refs)\b\.?\s*\(?\d", re.I)
_SUBFIG = re.compile(r"(^\(?[a-z]\)\s*$)|(\(\s*[a-z]\s*$)|(^\s*\(\s*[a-z]\s*\)\s*\()", re.I)
_NUMBER_ONLY = re.compile(r"[\d\s.,±%()\-–+/×x:]+")
_SPECIMEN_ID = re.compile(r"[A-Z]{1,3}[-_]?\d{1,2}[a-z]?")
# Specimen labels that the LLM likes to call materials ("LT1", "MT3"): 2-3 letters + ONE digit.
# Real grades such as H13, S460 or 316L do not match.
_SPECIMEN_ID_STRICT = re.compile(r"[A-Z]{2,3}[-_]?\d")


def junk_reason(text: str, entity_type: str, in_lexicon: bool) -> Optional[str]:
    """Return why `text` is not a usable entity, or None if it looks fine."""
    if in_lexicon:
        return None
    t = normalize_chars(text).strip(" .,;:")
    t = re.sub(r"^(the|a|an|this|that|these|those|such|their|its|our)\s+", "", t, flags=re.I)
    low = t.lower()
    if len(t) < 2:
        return "too_short"
    if _NUMBER_ONLY.fullmatch(t):
        return "number_only"
    if t[0].isdigit() and parse_quantity_text(t) is not None:
        return "value_not_entity"        # "1000 °C", "419.30 MPa": a value, handled by HAS_VALUE
    if _FIG_REF.search(t) or _SUBFIG.search(t):
        return "figure_or_table_reference"
    if low in _GENERIC_WORDS:
        return "generic_word"
    if entity_type not in ("Material", "PostProcessing") and _SPECIMEN_ID.fullmatch(t):
        return "specimen_id"
    if _SPECIMEN_ID_STRICT.fullmatch(t):
        return "specimen_id"
    if len(t) > 80 or t.count(" ") > 8:
        return "too_long"
    if entity_type == "CharacterizationMethod" and not _METHOD_CUES.search(t):
        return "not_a_method"
    return None
