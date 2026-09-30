"""Character-level text normalisation helpers.

`normalize_chars` performs ONLY one-to-one character substitutions so that
character offsets computed on the normalised string are valid on the original
string. This lets us match on clean text while keeping verbatim evidence.
"""
from __future__ import annotations

import re
import unicodedata

_ONE_TO_ONE = {
    "µ": "μ",   # micro sign -> Greek mu
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-", "﹣": "-", "－": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "′": "'", "’": "'", "‘": "'", "´": "'",
    "″": '"', "“": '"', "”": '"',
    "×": "x",        # multiplication sign
    "·": ".", "⋅": ".",  # middle dot (units like W·m-1)
    "³": "3", "²": "2", "⁻": "-", "¹": "1",
    # degree sign look-alikes produced by PDF fonts: "920 ◦C", "67∘", "20 ˚C"
    "◦": "°", "∘": "°", "˚": "°", "º": "°",
}
_TRANS = str.maketrans(_ONE_TO_ONE)


def normalize_chars(text: str) -> str:
    """Length-preserving normalisation (dashes, micro sign, spaces, superscripts)."""
    out = text.translate(_TRANS)
    assert len(out) == len(text)
    return out


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def match_key(text: str) -> str:
    """Aggressive key used for alias lookup: case-folded, no spaces/dashes/dots.

    Ti-6Al-4V, Ti–6Al–4V, Ti6Al4V, 'Ti 6Al 4V' -> 'ti6al4v'
    """
    t = unicodedata.normalize("NFKC", normalize_chars(text)).casefold()
    t = t.replace("′", "'").replace("″", "''")
    t = re.sub(r"[\s\-_/().,]+", "", t)
    return t


def singularize(word: str) -> str:
    w = word.strip()
    low = w.lower()
    if len(w) <= 3 or w.isupper():
        return w
    if low.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if low.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    if low.endswith("s") and not low.endswith(("ss", "is", "us", "sis")):
        return w[:-1]
    return w


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9α-ω]+", normalize_chars(text).lower()))


def token_overlap(a: str, b: str) -> float:
    """Fraction of tokens in `a` that also occur in `b`."""
    ta, tb = token_set(a), token_set(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)
