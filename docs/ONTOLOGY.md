# AM ontology

The source of truth in code is `src/ontology/schema.py` (types, relations, allowed signatures). Canonical entities and their aliases live in `config/lexicon.yaml`.

## Entity types

| Type | Examples | Key attributes |
|---|---|---|
| Paper | – | paper_id, title, authors, year, journal, DOI, source_file, abstract, is_synthetic |
| Material | Ti-6Al-4V, AlSi10Mg, Inconel 718, 316L, PLA, ABS | canonical_name, aliases, material_family, composition |
| Process (AMProcess) | LPBF, EBM, DED, WAAM, FDM, SLA | canonical_name, aliases, process_family (ISO/ASTM 52900 category) |
| Parameter | laser power, scanning speed, layer thickness, hatch spacing, volumetric energy density, wire feed rate | canonical_name, aliases, allowed dimensions |
| ParameterValue / PropertyValue | 200 W, 800 mm/s, 30 μm, 950 MPa | raw, value / min / max / ±, unit, normalized_value, normalized_unit, dimension, context |
| Geometry | tensile specimen, cube, lattice structure, thin wall | |
| Microstructure | grain size, columnar grains, α phase, α' martensite, dendritic structure | |
| Defect | porosity, lack of fusion, keyhole porosity, cracking, balling | |
| Property | tensile strength, yield strength, elongation, hardness, relative density, surface roughness | allowed dimensions |
| PostProcessing | heat treatment, HIP, annealing, machining | family |
| CharacterizationMethod | SEM, XRD, CT, EBSD, tensile testing | |
| ExperimentalCondition | build orientation, atmosphere, substrate temperature | |
| Finding | a verbatim result sentence | finding_type (result, trend, comparison, null_result, conclusion) |
| Claim | one piece of evidence for one relation (reified) | evidence, page, section, confidence, status, knowledge_type |

Value nodes are **per observation**. The same "200 W" in two papers gives two nodes, so a value can never be taken out of its paper context.

## Relations and allowed signatures

| Relation | Subject → Object |
|---|---|
| STUDIES | Paper → Material (DERIVED from mentions) |
| USES_PROCESS | Paper → Process (DERIVED) |
| PROCESSES | Process → Material. `PROCESSED_BY` in the brief is the same relation stored in this direction. |
| HAS_PARAMETER | Process → Parameter |
| HAS_VALUE | Parameter/ExperimentalCondition → ParameterValue; Property/Defect/Microstructure → PropertyValue |
| HAS_CONDITION | Process → ExperimentalCondition |
| FABRICATES | Process → Geometry |
| PRODUCES | Process → Property or Microstructure |
| AFFECTS | Parameter/Microstructure/PostProcessing/Defect/ExperimentalCondition/Geometry → Property; Parameter/ExperimentalCondition → Microstructure |
| INFLUENCES | Parameter/ExperimentalCondition/PostProcessing/Microstructure → Defect |
| CAUSES | Process → Defect, Defect → Defect |
| IMPROVES | PostProcessing → Property |
| CHANGES | PostProcessing → Microstructure |
| CHARACTERIZES | CharacterizationMethod → Material/Microstructure/Defect/Property |
| REPORTS | Paper → Finding |
| SUPPORTS | Paper → Claim (the paper supports that relationship) |
| MENTIONS | Paper → any entity (with count) |

A relation whose signature is not listed is **rejected** by quality control.

## Claim attributes

- `trend`: `positive`, `negative`, `unspecified` or `none`. This is computed from the subject modifier ("increasing laser power") and the predicate ("reduced porosity").
- `negated`: an explicit null result, for example "did not significantly affect". These are kept, because null results are knowledge too.
- `hedged`: the claim uses words like may, might, could or suggests. Confidence is lowered.
- `claim_type`: `finding` (a result of this study) or `investigated` (a topic of study, not a result).
- `claim_origin`: `this_study`, `abstract` or `background`. Background claims come from the introduction or from citations; they describe cited work and are capped at review.
- `context_resolution`: where the process came from — `same_sentence`, `previous_sentence` or `paper_single_process`.
- `value_context`: the materials and processes named in the sentence, and the material state (as-built or post-processed).

## Epistemic types (rule 22)

| knowledge_type | Meaning | Example |
|---|---|---|
| EXPLICIT | The sentence states the relation with a predicate | "Increasing laser power reduced porosity." |
| EXTRACTED | Built from entities in one sentence or table row, with no explicit predicate | "LPBF with a laser power of 200 W" → HAS_PARAMETER |
| DERIVED | Computed by the system | STUDIES from mentions; a Spearman trend across table rows |
| INTERPRETED | LLM explanation | Only in assistant answers, labelled "Interpretation". Never stored as a fact. |

## Normalisation policy

- Aliases carry confidences. Trade names get lower confidence than names for the same process category (DMLS → LPBF: 0.85; SLM → LPBF: 0.95).
- Related but distinct concepts are separate entities and are never merged: SLS vs LPBF, WAAM vs DED, keyhole porosity vs porosity, hot cracking vs cracking.
- Unknown terms become *new* entities with confidence 0.5. Any claim that touches one goes to review, where a human can merge it or accept it.
- Manual corrections go to `data/review/normalization_overrides.json` and take precedence over the lexicon.
- Units: the original value and unit are always kept. A normalised value is added in the canonical unit of the dimension: W, mm/s, μm, °C, MPa, J/mm³, J/mm, %, HV, g/cm³ and so on.
