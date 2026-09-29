from src.normalization.entity_normalizer import EntityNormalizer, NormalizationOverrides, find_abbreviations
from src.normalization.text import match_key, normalize_chars


def test_char_normalisation_is_length_preserving():
    s = "Ti–6Al–4V at 30 µm and 800 mm s−1"
    assert len(normalize_chars(s)) == len(s)
    assert match_key("Ti–6Al–4V") == match_key("Ti6Al4V") == match_key("Ti-6Al-4V")


def test_process_aliases(normalizer):
    for s in ["SLM", "Selective laser melting", "Selective Laser Melting", "LPBF", "L-PBF", "Laser Powder Bed Fusion", "PBF-LB/M"]:
        r = normalizer.normalize(s, "Process")
        assert r.canonical_name == "LPBF", s
    assert normalizer.normalize("DMLS", "Process").confidence < normalizer.normalize("SLM", "Process").confidence


def test_do_not_blindly_merge(normalizer):
    assert normalizer.normalize("SLS", "Process").canonical_name == "SLS"          # not LPBF
    assert normalizer.normalize("WAAM", "Process").canonical_name == "WAAM"        # not DED
    assert normalizer.normalize("keyhole porosity", "Defect").canonical_name == "keyhole porosity"
    r = normalizer.normalize("gyroid infill pattern", "Parameter")
    assert r.method == "new_entity" and r.confidence <= 0.5 and not r.in_lexicon


def test_material_variants(normalizer):
    for s in ["Ti6Al4V", "Ti-6Al-4V", "Ti–6Al–4V", "Ti 6Al 4V", "Ti64"]:
        assert normalizer.normalize(s, "Material").canonical_name == "Ti-6Al-4V"
    assert normalizer.normalize("Inconnel 718", "Material").method == "fuzzy"


def test_retyping_and_plurals(normalizer):
    r = normalizer.normalize("porosity", "Property")
    assert r.entity_type == "Defect" and r.confidence < 0.9
    assert normalizer.normalize("scan speeds", "Parameter").canonical_name == "scanning speed"
    assert normalizer.normalize("lack of fusion defects", "Defect").canonical_name == "lack of fusion"


def test_manual_override_and_merge(lexicon, tmp_path):
    ov = NormalizationOverrides(tmp_path / "ov.json")
    ov.add_alias("Parameter", "laser scanning velocity", "Parameter:scanning speed")
    ov.save()
    n = EntityNormalizer(lexicon, NormalizationOverrides(tmp_path / "ov.json"))
    r = n.normalize("laser scanning velocity", "Parameter")
    assert r.canonical_name == "scanning speed" and r.method == "manual_override" and r.confidence == 1.0
    ov.add_merge("Parameter:preheat temp", "Parameter:preheating temperature")
    assert ov.resolve("Parameter:preheat temp") == "Parameter:preheating temperature"


def test_abbreviations():
    ab = find_abbreviations("using selective laser melting (SLM) and a volumetric energy density (VED)")
    assert ab["SLM"] == "selective laser melting"
    assert ab["VED"] == "volumetric energy density"
