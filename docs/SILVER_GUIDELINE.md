# Silver-standard annotation guideline (AM-KG LPBF benchmark)

> **Provenance.** `data/annotations_silver/` (10 LPBF papers, 313 chunks) was annotated on 2026-09-30 by Claude
> (one LLM sub-agent per paper) with exactly this guideline, from the raw chunk text only, without seeing the
> system's predictions. It was **not verified by a human**. Report it as a *silver standard*; a human-verified
> subset is needed before any claim about "gold" accuracy. Build: `python scripts/silver_annotate.py build`,
> then `python main.py annotate-import data/labelstudio/silver_export.json --out data/annotations_silver`
> and set `annotation_mode: llm_silver`.
>
> **Consistency check (2026-09-30).** Two papers (ti64_cacace2026, in718_tan2023) were annotated a second time by
> an independent sub-agent (`data/annotations_silver_b/`). Agreement with the first pass
> (`python scripts/silver_agreement.py data/annotations_silver data/annotations_silver_b`): entities F1 0.92,
> relations F1 0.85 strict / 0.88 relaxed, values F1 0.91. This shows the silver standard is consistent; both
> passes share one model, so it does not show that it is correct.

You are annotating one scientific paper (laser powder bed fusion of metals) to build a reference
("silver standard") for scoring an information-extraction system. Be careful, consistent and literal:
annotate what the text says, never what is generally true.

## Independence (important)
- Work ONLY from `python scripts/silver_annotate.py show <stem> <chunk>` output.
- Do NOT open `data/labelstudio/tasks.json` directly, and do not look at any system predictions or extracted files.

## Workflow
From the project root. For every chunk of your paper (0..N-1, see `python scripts/silver_annotate.py list`):
1. `python scripts/silver_annotate.py show <stem> <chunk>`
2. Write `data/labelstudio/silver_out/<stem>/<chunk>.json` (format below). A chunk with nothing to annotate gets `{"entities": [], "relations": []}`.
3. `python scripts/silver_annotate.py check <stem> <chunk>`: fix every ERROR; read the canonical names it prints.
Finish with `python scripts/silver_annotate.py checkall <stem>`: it must report 0 files with errors or missing.
Several chunks per shell call is fine (e.g. show 3 chunks, then write 3 files, then check them).

## File format
```json
{"entities": [
  {"id": "e1", "s": 3, "text": "laser power", "label": "Parameter"},
  {"id": "e2", "s": 3, "text": "200 W", "label": "Value"},
  {"id": "e3", "s": 5, "text": "SLM", "label": "Process", "occ": 2}
 ],
 "relations": [{"from": "e1", "to": "e2", "label": "HAS_VALUE"}]}
```
- `s` = sentence number in the `show` output. `text` must be copied EXACTLY (case, symbols such as ±, µ, ³, –).
- `occ` = which occurrence of that text in that sentence (1-based, default 1).
- Entities must not overlap. Ids unique within the file.
- `canonical` (optional): only when the name printed by `check` is clearly wrong for this mention
  (e.g. it maps a different alloy or a different property). Otherwise leave it out.

## Entities
| Label | Mark | Do not mark |
|---|---|---|
| Material | alloy / powder / feedstock: `Ti-6Al-4V`, `Ti6Al4V`, `316L`, `AlSi10Mg`, `IN718`, `Inconel 718` (alloy name only, without "powder"/"alloy" unless part of the name) | generic: "steel", "the material", "samples", "metal" |
| Process | `LPBF`, `SLM`, `L-PBF`, `PBF`, `laser powder bed fusion`, `selective laser melting`, `DMLS`, `EBM`, `DED` | "additive manufacturing"/"AM" in general, "the process" (HIP is PostProcessing) |
| Parameter | `laser power`, `scanning speed`, `scan speed`, `hatch spacing`, `layer thickness`, `energy density`, `volumetric energy density`, `VED`, `scan strategy`, `overlap ratio`, `spot size`, `point distance`, `exposure time` | "process parameters" in general |
| Property | `tensile strength`, `ultimate tensile strength`, `yield strength`, `elongation`, `hardness`, `relative density`, `density`, `surface roughness`, `Ra`, `fatigue life`, `Young's modulus`, `residual stress` | "mechanical properties"/"properties" in general |
| Defect | `porosity`, `pores`, `lack of fusion`, `keyhole pores`, `cracks`, `balling`, `spatter`, `gas porosity` | "defects" in general |
| Microstructure | `columnar grains`, `α' martensite`, `melt pool`, `grain size`, `texture`, `cellular structure`, `Si network`, `δ phase`, `Laves phase`, `dislocation cells` | "microstructure" alone, unless it is the specific object of a claim |
| PostProcessing | `heat treatment`, `HIP`, `stress relief`, `annealing`, `solution treatment`, `aging`, `machining`, `T6` | — |
| CharacterizationMethod | `SEM`, `XRD`, `EBSD`, `CT`, `XCT`, `X-ray computed tomography`, `tensile testing`, `tensile test`, `DIC`, `optical microscopy`, `EDS`, `TEM`, `Archimedes method`, `profilometry` | figure names, software, statistics (ANOVA) |
| Geometry | `cube`, `cubes`, `tensile specimen(s)`, `lattice structure`, `thin walls`, `cylinders` | specimen IDs such as "S1" or "LT2" |
| ExperimentalCondition | `build orientation`, `preheating temperature`, `build plate temperature`, `atmosphere`, `argon`, `test temperature`, `strain rate` | — |
| Value | a number WITH its unit: `200 W`, `30 μm`, `950 ± 12 MPa`, `99.6 %`, `79 J/mm3`, `150–250 W` | numbers without units, "Fig. 3", years, citations, sample counts |

Rules:
- Shortest complete name: `laser power`, not "the applied laser power"; no articles ("the", "this").
- Annotate what the paper states about ITS OWN work. In introduction / literature review sentences that only
  report other studies ("Smith et al. found ..."), do NOT annotate relations or values; mark entities there only
  if the paper itself uses them.
- Mark every mention you need for a relation. Otherwise marking each distinct entity once per chunk is enough
  (entity scores are per paper). Prefer marking it in a sentence where it takes part in a relation.
- Plain-English words (e.g. "power" meaning laser power) → mark only if clearly the parameter; the check output
  shows the canonical name.

## Relations (arrow from subject → object)
| Relation | From → To | Example |
|---|---|---|
| PROCESSES | Process → Material | "Ti-6Al-4V was fabricated by LPBF" |
| HAS_PARAMETER | Process → Parameter | "LPBF with a laser power of …" / "LPBF parameters: laser power, scan speed" |
| HAS_VALUE | Parameter / Property / Defect / ExperimentalCondition / Microstructure / Geometry → Value | laser power → 200 W; relative density → 99.5 % |
| HAS_CONDITION | Process → ExperimentalCondition | LPBF → argon atmosphere |
| AFFECTS | Parameter / PostProcessing / Microstructure / Condition / Defect / Process → Property / Microstructure | "higher power increased hardness"; "pores reduced elongation" |
| INFLUENCES | Parameter / PostProcessing / Condition → Defect | "energy density reduced porosity" |
| CAUSES | Process / Defect / Parameter → Defect | "keyholing caused pores"; "low energy density caused lack of fusion" |
| IMPROVES | PostProcessing → Property | "HIP improved elongation" |
| CHANGES | PostProcessing → Microstructure | "heat treatment transformed α' into α+β" |
| CHARACTERIZES | CharacterizationMethod → Material / Microstructure / Defect / Property | "porosity was measured by CT" |
| PRODUCES | Process → Property / Microstructure | "LPBF produced columnar grains" |
| FABRICATES | Process → Geometry | "cubes were built by LPBF" |

Rules:
- Only annotate a relation that ONE sentence states (both ends in that sentence). Never add a relation because it
  is generally true. (Exception allowed but rare: a value in a table row whose parameter is named in the same row.)
- Negated statements ("had no effect on") are still relations; the sentence states the link.
- HAS_VALUE: link the value to the quantity it measures in that sentence ("relative density, surface roughness and
  hardness were 99.5%, 1.2 µm and 240 HV, respectively" → three HAS_VALUE relations, in order).
- Use the most specific relation allowed by the table; `silver_annotate.py check` rejects disallowed type pairs.
- Be thorough: the reference is used to measure RECALL, so missing relations/values hurt as much as wrong ones.

## Report when done
Return a short summary: chunks annotated, total entities / relations / values, and any systematic ambiguity you
resolved (so the lead annotator can keep decisions consistent across papers).
