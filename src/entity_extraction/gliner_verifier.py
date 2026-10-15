"""LLM checker for NEW GLiNER entities (way 2).

GLiNER finds terms that are not in the lexicon, but many of them are noise ("utility degree",
"the cube edge", "melt"). A small local LLM looks at each new candidate *in its sentence* and says
keep / drop (and may correct the type). Lexicon entities are never sent: they are trusted.

Candidates are sent in small batches; answers are cached by InstrumentedLLM, so re-running a sweep
with another threshold only asks about candidates it has not seen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..llm.instrumented import llm_stage
from ..llm.json_utils import parse_json_lenient

log = logging.getLogger(__name__)

TYPES = {
    "Material": "alloy, metal powder or feedstock (Ti-6Al-4V, 316L, AlSi10Mg, IN718)",
    "Process": "additive manufacturing or joining process (LPBF, SLM, DED, EBM)",
    "Parameter": "machine/process setting chosen before building (laser power, scan speed, hatch spacing, "
                 "layer thickness, energy density, scan strategy, spot size)",
    "Property": "measured property of the part or powder (tensile strength, hardness, density, roughness, "
                "fatigue life, residual stress)",
    "Defect": "manufacturing defect (pores, porosity, cracks, lack of fusion, keyhole, spatter, balling)",
    "Microstructure": "microstructural or fracture feature (grains, phases, precipitates, melt pool, dendrites, "
                      "dimples, texture)",
    "PostProcessing": "treatment after building (heat treatment, HIP, aging, machining, polishing)",
    "CharacterizationMethod": "measurement or test method/instrument (SEM, XRD, CT, EBSD, tensile test)",
    "Geometry": "specimen or part shape (tensile specimen, cube, lattice, thin wall)",
    "ExperimentalCondition": "build or test condition (build orientation, atmosphere, argon, preheating "
                             "temperature, room temperature, strain rate)",
}

SYSTEM = (
    "You check candidate entities for a knowledge graph about laser powder bed fusion (metal additive "
    "manufacturing). For each candidate decide if it is a SPECIFIC domain concept of one of these types:\n"
    + "\n".join(f"- {k}: {v}" for k, v in TYPES.items())
    + "\nDROP: generic words or phrases (\"the process\", \"properties\", \"results\", \"samples\", \"different "
    "parameters\"), numbers or values with units, statistics (p-value, ANOVA, regression), specimen ids (S1, "
    "LT2), figure/table references, people, places, companies, software, whole clauses, and fragments that are "
    "not a complete term. KEEP specific technical terms even if they are rare.\n"
    "If the term is a domain concept but the given type is wrong, keep it and give the correct type.\n"
    'Answer JSON only: {"decisions": [{"id": 1, "keep": true, "type": "Defect"}, ...]} with one entry per id.'
)


@dataclass
class Candidate:
    sentence: str
    text: str
    entity_type: str


@dataclass
class Verdict:
    keep: bool
    entity_type: str
    reason: str = ""


class GlinerVerifier:
    def __init__(self, llm: Any, batch_size: int = 8) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.calls = self.failures = 0

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "model", "llm"))

    # answers per (model, sentence, term, type), shared by all verifiers in this process: a sweep over
    # thresholds asks only about candidates it has not seen yet (the disk cache keys whole batches)
    _memo: dict[tuple[str, str, str, str], Verdict] = {}

    def verify(self, candidates: list[Candidate]) -> list[Verdict]:
        keys = [(self.model_name, c.sentence, c.text, c.entity_type) for c in candidates]
        todo = [c for c, k in zip(candidates, keys) if k not in self._memo]
        todo = list({(c.sentence, c.text, c.entity_type): c for c in todo}.values())
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            for c, v in zip(batch, self._batch(batch)):
                if v.reason != "checker_failed":      # failures are not remembered: retried next time
                    self._memo[(self.model_name, c.sentence, c.text, c.entity_type)] = v
        return [self._memo.get(k, Verdict(True, c.entity_type, "checker_failed")) for c, k in zip(candidates, keys)]

    def _batch(self, batch: list[Candidate]) -> list[Verdict]:
        lines = []
        for n, c in enumerate(batch, 1):
            lines.append(f'{n}. sentence: "{c.sentence}"\n   candidate: "{c.text}"   proposed type: {c.entity_type}')
        user = "Candidates:\n" + "\n".join(lines)
        self.calls += 1
        try:
            with llm_stage(self.llm, "gliner_check"):
                raw = self.llm.chat(SYSTEM, user, json_mode=True)
            data = parse_json_lenient(raw)
        except Exception as exc:          # LLM down / bad JSON: keep everything (fail open, like no checker)
            self.failures += 1
            log.warning("GLiNER checker failed (%s); keeping %d candidate(s)", exc, len(batch))
            return [Verdict(True, c.entity_type, "checker_failed") for c in batch]
        decisions = data.get("decisions", data) if isinstance(data, dict) else data
        by_id: dict[int, dict] = {}
        for d in decisions if isinstance(decisions, list) else []:
            try:
                by_id[int(d.get("id"))] = d
            except (TypeError, ValueError, AttributeError):
                continue
        verdicts = []
        for n, c in enumerate(batch, 1):
            d = by_id.get(n)
            if d is None:                  # no answer for this id: keep (fail open)
                verdicts.append(Verdict(True, c.entity_type, "no_answer"))
                continue
            keep = d.get("keep")
            keep = keep if isinstance(keep, bool) else str(keep).strip().lower() in {"true", "yes", "1"}
            etype = d.get("type") if d.get("type") in TYPES else c.entity_type
            verdicts.append(Verdict(keep, etype, "" if keep else "llm_checker_rejected"))
        return verdicts


def make_verifier(spec: str, settings: Any) -> Optional[GlinerVerifier]:
    """'ollama:qwen2.5:3b' (local, free) or 'groq:openai/gpt-oss-120b' -> GlinerVerifier; '' -> None."""
    if not spec:
        return None
    from ..llm.client import OllamaClient
    from ..llm.instrumented import InstrumentedLLM
    from ..llm.openai_compat import parse_llm_spec
    provider, _, model = spec.partition(":")
    if provider.strip().lower() == "ollama":
        c = settings.llm
        client: Any = OllamaClient(c.base_url, model.strip(), 0.0, c.num_ctx, c.timeout_s, False,
                                   num_predict=600, keep_alive=getattr(c, "keep_alive", "30m"))
    else:
        client = parse_llm_spec(spec, settings, num_predict=1500)
    cache = settings.path("cache_dir") if settings.llm.cache else None
    return GlinerVerifier(InstrumentedLLM(client, cache))
