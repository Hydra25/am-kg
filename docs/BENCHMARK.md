# Building the LPBF benchmark (Phase 2)

Goal: a small, honest gold standard of real **LPBF metal** papers. Every extraction mode is scored on it, and the resulting table goes in the paper.

Plan: **10 pilot papers** first (fix the guideline and the tools), then grow to **30**.

## 1. Choosing papers

Pick papers that satisfy all of these:

| Criterion | Why |
|---|---|
| Process is laser powder bed fusion (LPBF / SLM / L-PBF / PBF-LB/M / DMLS) | the benchmark scope |
| Metal alloy: Ti-6Al-4V, 316L, AlSi10Mg, IN718, IN625, 17-4PH, CoCr, Cu, maraging steel | covered by the lexicon; comparable across papers |
| Experimental study with numbers (parameters and measured properties) | we evaluate values, not reviews |
| Text PDF (you can select the text), not a scan | the parser needs a text layer |
| You may use it: open access (CC-BY / CC-BY-NC) or accessed through the institute | publishing and sharing rules |

Good places for open-access LPBF papers:

- *Materials* and *Metals* (MDPI)
- *Additive Manufacturing Letters*
- *Materials & Design*
- *Journal of Materials Research and Technology*
- *Procedia CIRP / Procedia Structural Integrity*
- Scientific Reports

Useful search strings:

- `"laser powder bed fusion" Ti-6Al-4V porosity tensile`
- `"selective laser melting" 316L process parameters density`

Mix of papers:

- parameter studies: power, speed, hatch and layer versus density or defects
- heat treatment or HIP versus microstructure or mechanical properties
- a few with tables of values

Record every paper in `data/benchmark_papers.csv` with the columns `file, doi, alloy, topic, licence`.

Keep the real papers apart from the demo papers, for example in `data/papers/lpbf/`.

> Only published papers go into cloud tools. Never put the benchmark PDFs in git: `data/papers/*` is git-ignored except for the demo files.

## 2. Setting up Label Studio (free, local)

Install it once, in its own virtual environment, so it does not disturb the project:

```bash
python3 -m venv ~/ls-venv
~/ls-venv/bin/pip install label-studio
~/ls-venv/bin/label-studio start          # opens http://localhost:8080
```

Then:

1. Sign up locally (the account stays on your Mac) and choose **Create project**, named "AM-KG LPBF benchmark".
2. Under **Labeling Setup → Custom template → Code**, paste the contents of `data/labelstudio/labeling_config.xml`.
3. Import `data/labelstudio/tasks.json`.

To create the tasks:

```bash
python main.py extract data/papers/lpbf --mode hybrid --entities gliner   # pre-annotations come from this
python main.py annotate-export data/papers/lpbf
```

Each task is about 12 sentences of one section and shows the system's proposals as pre-annotations. You correct them rather than starting from zero.

## 3. Annotation guideline

Annotate **what the paper states about its own work**. Skip what is only cited from other studies (typically in the introduction), unless the paper uses it itself.

### Entities

| Label | Mark | Do not mark |
|---|---|---|
| Material | alloy / powder / feedstock: `Ti-6Al-4V`, `316L powder` | generic words: "steel", "the material", "samples" |
| Process | `LPBF`, `SLM`, `EBM`, `DED` | "additive manufacturing" in general, "the process" (HIP is PostProcessing) |
| Parameter | `laser power`, `scanning speed`, `hatch spacing`, `layer thickness`, `energy density`, `scan strategy` | "process parameters" in general |
| Property | `tensile strength`, `yield strength`, `elongation`, `hardness`, `relative density`, `surface roughness` | "mechanical properties" in general |
| Defect | `porosity`, `lack of fusion`, `keyhole pores`, `cracks`, `balling` | "defects" in general |
| Microstructure | `columnar grains`, `α' martensite`, `melt pool`, `grain size`, `texture` | "microstructure" alone, only if it is the specific object of a claim |
| PostProcessing | `heat treatment`, `HIP`, `stress relief`, `machining` | — |
| CharacterizationMethod | `SEM`, `XRD`, `EBSD`, `CT`, `tensile testing`, `DIC` | figure names, software |
| Geometry | `cube`, `tensile specimen`, `lattice structure` | specimen IDs such as "S1" or "LT2" |
| ExperimentalCondition | `build orientation`, `preheating temperature`, `atmosphere` | — |
| **Value** | a number with its unit: `200 W`, `30 μm`, `950 ± 12 MPa`, `99.6 %` | numbers without units, or references such as "Fig. 3" |

Rules:

- Mark the **shortest complete name**: `laser power`, not "the applied laser power". Leave out articles ("the", "this").
- If the text uses a synonym or an abbreviation (SLM, L-PBF), you may type the canonical name (LPBF) in the *canonical* box. The importer also normalises known synonyms automatically.
- Mark every occurrence you need for a relation. Entity sets are compared per paper, so marking a term once is enough for the entity score.

### Relations

Draw the arrow **from subject to object**, as in the table:

| Relation | From → To | Example |
|---|---|---|
| PROCESSES | Process → Material | "Ti-6Al-4V was fabricated by LPBF" |
| HAS_PARAMETER | Process → Parameter | "LPBF with a laser power of …" |
| HAS_VALUE | Parameter / Property / Defect / Condition → Value | laser power → 200 W |
| HAS_CONDITION | Process → ExperimentalCondition | LPBF → argon atmosphere |
| AFFECTS | Parameter / PostProcessing / Microstructure → Property / Microstructure | "higher power increased hardness" |
| INFLUENCES | Parameter / PostProcessing / Condition → Defect | "power reduced porosity" |
| CAUSES | Process / Defect → Defect | "keyholing caused pores" |
| IMPROVES | PostProcessing → Property | "HIP improved elongation" |
| CHANGES | PostProcessing → Microstructure | "heat treatment transformed α' into α+β" |
| CHARACTERIZES | Method → Material / Microstructure / Defect / Property | "porosity was measured by CT" |
| PRODUCES | Process → Property / Microstructure | "LPBF produced columnar grains" |
| FABRICATES | Process → Geometry | "cubes were built by LPBF" |

Rules:

- Only annotate a relation that **one sentence states**. Never add a relation because it is generally true.
- Negated statements ("had no effect on") are still relations; the sentence says so.
- Leave a system proposal in place if it is correct, delete it if it is wrong, and add what is missing. Take particular care over what is missing: recall depends on it.

## 4. Importing, then scoring

In Label Studio: **Export → JSON**. Then:

```bash
python main.py annotate-import ~/Downloads/project-1-at-XXXX.json --annotator "your name"
python main.py benchmark data/papers/lpbf --modes rule,gliner,hybrid_gliner,hybrid_llm
```

The result is in `data/benchmark/results.md` (also `.csv` and `.json`). It holds entity, relation and value P/R/F1, seconds, LLM calls and tokens per paper.

Add `--accepted-only` to score only automatically accepted claims (the precision view).

## 5. Rigour: what reviewers will ask

- **Pre-annotation bias.** The annotator saw the system output. Mitigate it:
  - annotate at least 3 papers from scratch: `annotate-export --no-predictions`, then `annotate-import --from-scratch`;
  - report both numbers;
  - the gold files record `annotation_mode`.
- **Inter-annotator agreement.** A second person annotates 3–5 papers. Report agreement: F1 between the two annotators, or Cohen's κ on relations.
- **No tuning on the test set.** Improve rules and prompts on the 10 pilot papers only. Freeze the system, then score the other 20.
- **Lexicon leakage.** Report how many gold entities were already in the lexicon before annotation.
