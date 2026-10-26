# AM-KG — Additive Manufacturing Knowledge Graph: Complete Project Explanation

*Status as of 27 October 2026 · Author: M.Tech student (Additive Manufacturing) · Guide: Prof. Abhishek Tewari*
*Code: GitHub `Hydra25/am-kg` (private), branch `phase2-benchmark`*

This document explains the whole project in simple words: what it does, how every part works, what was done step by step, every important number, what we learned, and the answers to questions people are likely to ask. Read sections 1–3 for the big picture, section 9 for the numbers, and section 13 before any meeting.

---

## 1. The project in one paragraph

Research papers on **Laser Powder Bed Fusion (LPBF)** – metal 3D printing – contain thousands of facts such as *"increasing laser power reduced porosity in Ti-6Al-4V"* or *"tensile strength was 1050 MPa"*. These facts are locked inside PDFs. **AM-KG** is a software pipeline that reads LPBF papers automatically, extracts the **entities** (materials, process parameters, properties, defects, microstructures…), their **values with units**, and the **relations** between them, normalises everything to standard names and units, checks the quality, and stores it in a **knowledge graph** (Neo4j). **Every fact keeps its evidence** (paper, page, section and the exact sentence), so nothing in the graph is a guess. The graph can then be searched, visualised, and used by a research assistant to answer questions like *"What affects porosity in LPBF Ti-6Al-4V, and in which papers?"*. Everything is built with **free tools only** and runs on a MacBook Air.

---

## 2. Why this project? (motivation)

- **Too many papers.** LPBF research grows by thousands of papers per year. Nobody can read all of them to find, for example, all reported values of laser power vs. porosity.
- **Knowledge is unstructured.** The same thing is written in many ways: *SLM / L-PBF / laser powder bed fusion*, *Ti64 / Ti-6Al-4V / Ti6Al4V*, *◦C / °C*. A database needs one standard name.
- **Process–structure–property (PSP) links** are the heart of materials science. A graph is the natural way to store "parameter → affects → property" links and to find gaps (combinations nobody has studied).
- **LLMs alone are not trustworthy.** They can invent facts. Our design: *the LLM may propose, but the code verifies against the text*, and every fact carries its source sentence.
- **Connection to Prof. Tewari's work.** His paper on LLM-based AI agents for automated extraction of material properties and structural features shows the same goal. AM-KG covers this objective for LPBF with a free, local, evidence-backed pipeline, and its planned next stage (agents / research assistant, PSP dataset) extends it.

---

## 3. Design principles (our rules)

1. **Evidence first** – every claim stores paper, page, section, sentence id and the verbatim sentence; the quality checker re-verifies the text.
2. **LLM proposes, code verifies** – an LLM output is used only if the entity text really occurs in the sentence, values parse with a valid unit, and the relation fits the ontology.
3. **Free only** – local models (Ollama qwen2.5:3b, GLiNER), free API tiers (Groq), free databases (Neo4j Community), free papers (open access).
4. **Honest evaluation** – fixed DEV/TEST split; we tune only on DEV and run TEST once per stage; the annotation set is called "silver" (LLM-made), never "gold".
5. **Legal and ethical** – no automated downloads from ScienceDirect/Elsevier (their terms forbid it), only Creative-Commons papers for harvesting, API keys only in a private `.env` file (never in code or git).
6. **Works without extras** – without Ollama it runs rule-only; without Neo4j it uses a local graph file.

---

## 4. How the system works (the pipeline)

```
PDF ──► 1. Parse (text, sections, sentences, tables)
     ──► 2. Paper context (abbreviations "LPBF (laser powder bed fusion)", symbols "P = laser power")
     ──► 3. Rules + Lexicon (dictionary of 174 AM concepts) ──► entities + values with units
     ──► 4. GLiNER (AI entity finder) ──► extra/new entities (optional, discovery)
     ──► 5. LLM (qwen2.5:3b local, or Groq gpt-oss-120b) ──► extra relations + findings
     ──► 6. Normalisation (one standard name + SI units)
     ──► 7. Relations (rules + LLM) + Quality control (accepted / review / rejected)
     ──► 8. Knowledge graph (Neo4j) ──► Streamlit explorer, queries, research assistant
```

### Step by step (simple explanation)

**Step 1 – Parsing (`src/ingestion/`, `src/pdf_parser/`).** PyMuPDF (or GROBID) reads the PDF. We fixed many publisher problems: Elsevier's spaced "A B S T R A C T", Nature layouts, wrong titles, broken symbols (`◦C`, `lm` for µm, `–` read as `e`). Output: sections (abstract, methods, results…) and numbered sentences, saved in `data/processed/`.

**Step 2 – Paper context.** The pipeline learns each paper's own abbreviations ("as-built (AB)") and symbols ("P is the laser power"), so later "P = 200 W" is understood.

**Step 3 – Rules + lexicon (`src/entity_extraction/rule_based.py`, `config/lexicon.yaml`).** The lexicon is our dictionary: **174 concepts, 736 aliases** in 10 types (Material, Process, Parameter, Property, Defect, Microstructure, PostProcessing, CharacterizationMethod, Geometry, ExperimentalCondition). It follows **ISO/ASTM 52900** for process names and contains standard terms from ISO 21920 / ISO 25178 (Ra, Rz, Sq…), ASTM E8, E23 (Charpy), E384, B212/B527 (powder density). The matcher finds the longest alias in each sentence. The **unit parser** (`src/normalization/units.py`) reads "200 W", "21–37 J/mm³", "950 ± 12 MPa", and only attaches a value to an entity of the right physical dimension (W → power, not porosity).

**Step 4 – GLiNER (`src/entity_extraction/gliner_extractor.py`).** GLiNER is a small free AI model for "zero-shot" named-entity recognition: we give it labels in plain English (e.g. *"manufacturing defect such as pores, cracks or lack of fusion"*) and it marks matching spans with a confidence score (0–1). We keep a span only if its score ≥ the **threshold** (0.5), it passes the **junk filter** (`src/quality/junk.py`: no specimen labels like "AB sample", no citation numbers like "strength35", no generic words like "powder"), and the rule/lexicon match wins if both overlap. We now use GLiNER mainly to **discover new terms** (see Option B).

**Step 5 – LLM (`src/entity_extraction/llm_extractor.py`, `src/llm/`).** A local LLM (Ollama qwen2.5:3b) reads text chunks and proposes relations and findings as JSON. A "gate" sends only chunks that contain AM entities or numbers. Every answer is cached. Optionally a bigger free cloud model (Groq gpt-oss-120b) does the relations. A fallback chain (Groq → Cerebras → Gemini → OpenRouter → Ollama) exists for general use, but benchmark runs always use one fixed model.

**Step 6 – Normalisation (`src/normalization/`).** Every mention gets one canonical id, e.g. *SLM, L-PBF → LPBF*; *Ti64 → Ti-6Al-4V*. Seven steps: manual override → exact alias → paper abbreviation/symbol → plural/singular → generic head noun → fuzzy match (with guards: "toughness" must not become "surface roughness") → otherwise a new entity. Generic phrases like "the printing process" are mapped to the paper's main process.

**Step 7 – Relations and quality control (`src/relation_extraction/`, `src/quality/validator.py`).** Rule relations use trigger words ("increased", "reduced", "affects") and structure (HAS_VALUE, HAS_PARAMETER, STUDIES…). Each claim gets a confidence: ≥ 0.85 **accepted**, 0.60–0.85 **review**, < 0.60 **rejected** (kept for audit, not loaded).

**Step 8 – Knowledge graph and apps (`src/knowledge_graph/`, `app/streamlit_app.py`).** Neo4j stores Paper, Entity (per type), Value, Claim (with evidence sentence) and Finding nodes plus aggregated edges with the list of supporting papers. The Streamlit app has a **Neo4j Explorer** page with 9 ready demo questions ("What affects …?", "Evidence for a relation", "Reported values", "Alloy neighbourhood", "Research gaps"), read-only Cypher, and an interactive graph view.

---

## 5. The data

### 5.1 Pilot papers (10 open-access LPBF papers, 4 alloys)

| Set | Paper | Alloy | Topic |
|---|---|---|---|
| DEV | ti64_baskin2024 | Ti-6Al-4V | parameters vs tensile, Charpy, roughness, porosity |
| DEV | ti64_cacace2026 | Ti-6Al-4V | parameter sets at constant VED vs pores and tensile |
| DEV | 316l_lee2022 | 316L | energy density vs pores, hardness, roughness, tensile |
| DEV | alsi10mg_kumar2023 | AlSi10Mg | Box-Behnken: parameters vs roughness, hardness, porosity |
| DEV | in718_tan2023 | IN718 | energy density + heat treatment vs porosity, tensile, phases |
| TEST | ti64_lee2025 | Ti-6Al-4V ELI | energy density, hatch, power vs density, roughness, UTS |
| TEST | ti64_squillaci2025 | Ti-6Al-4V | HIP temperatures vs tensile, Charpy, hardness |
| TEST | 316l_obeidi2021 | 316L | same parameters on 4 machines vs porosity, UTS |
| TEST | alsi10mg_minkowitz2023 | AlSi10Mg | 108 samples: parameters vs density, UTS, hardness |
| TEST | in718_baldi2023 | IN718 | inter-layer cooling time vs porosity, melt pool |

**DEV** = used for development and tuning. **TEST** = held out; used only once per stage for the final number (prevents "overfitting", i.e. fitting our rules to the exact papers we test on).

### 5.2 Silver standard (the answer key)

To measure accuracy we need correct answers. Human annotation was not possible, so an LLM (Claude) annotated all 313 text chunks of the 10 papers from the raw text **without seeing our system's output**, following a written guideline (`docs/SILVER_GUIDELINE.md`). Result: 897 entities, 593 relations, 482 values. A second independent pass on 2 papers agreed well (entities F1 0.92, relations 0.85 strict / 0.88 relaxed, values 0.91), so the silver set is consistent. Because it is LLM-made and not human-verified, we always call it **"silver standard"**.

### 5.3 Harvested papers (for automatic lexicon growth)

A free harvester downloads open-access LPBF papers from **OpenAlex** (with a free API key). First round: **30 papers** (Ti-6Al-4V, 316L, IN718, IN625, AlSi10Mg, CuCrZr…). Skipped: 44 Elsevier/ScienceDirect (rule), 47 without PDF link, 27 not Creative-Commons, 31 refused by the publisher (HTTP 403), 8 not LPBF/metal. The 10 pilot papers (and 3 spare papers) are always excluded, so DEV/TEST stay clean.

---

## 6. How we measure (metrics in simple words)

- **Precision (P)** – of everything the system extracted, how much is correct? (few false alarms = high P)
- **Recall (R)** – of everything that should be found, how much did we find? (few misses = high R)
- **F1** – one number that balances P and R (harmonic mean). 1.0 = perfect.
- **Entity metric** – per paper, the set of (type, standard name) found vs. the silver set.
- **Relation strict** – subject, relation type and object must all match. **Relaxed** – similar relation family (e.g. AFFECTS ≈ INFLUENCES) and compatible names count.
- **Value metric** – the number must match within 1 % after unit normalisation.
- **New-entity recall (new-R)** – of the silver entities that are *not* in our lexicon, how many does GLiNER find? (shows GLiNER's value for discovery)

---

## 7. Timeline – what was done, in order

| Phase | What we did | Result |
|---|---|---|
| Base system | Ontology (entity/relation types), lexicon, parser, rule + LLM hybrid extraction, normalisation, QC, Neo4j/local graph, queries, Graph-RAG assistant, review UI, evaluation, Excel export | Working end-to-end pipeline |
| Phase 0 | Git, GitHub repo, CI (GitHub Actions: lint + tests), pre-commit, Makefile | Professional code base |
| Phase 1 – speed (merged) | LLM cache + per-stage measurement, rule-first gating, GLiNER entities, compact relation JSON, junk filter, quality fixes | 1033 s → ~185 s per paper (≈ 5.6× faster) |
| Phase 2 – benchmark | Label Studio export/import, benchmark runner, annotation guideline | Evaluation framework |
| | Parser fixes for publisher PDFs | All 10 papers parse correctly |
| | Free LLM fallback chain | Cloud LLMs without cost |
| | 10-paper pilot extraction (73 min on M4) | 6,440 mentions, 2,036 relations (45 % accepted), 895 findings |
| | Silver standard (313 chunks) + agreement check | Answer key |
| | Unit/normaliser/symbol fixes, generic-process fix, relaxed metric, Groq relation model, daily-limit detection | Fewer errors, better relations |
| | Neo4j + Streamlit Neo4j Explorer | Demo for professor |
| | GLiNER precision fix | Noise −92 % |
| | Lexicon expansion with standards (142 → 174 concepts) | TEST entity F1 0.480 → 0.507 |
| | GLiNER "4 ways" study (models, context, LLM checker, error analysis, junk rules) | GLiNER becomes a discovery tool |
| | Option B: free harvester + automatic lexicon growth (rule v1.1) | 12 new terms; TEST F1 0.507 → 0.510 |

---

## 8. The GLiNER study (why GLiNER is used for discovery, not as the main extractor)

1. **Labels matter most.** Domain labels ("laser process parameter such as laser power, scan speed…") instead of generic labels cut GLiNER's noise by 92 % (TEST precision 0.275 → 0.713, old lexicon).
2. **Model comparison (DEV).** small/medium models add almost nothing new at high thresholds; **gliner_large-v2.1** finds the most new terms (new-R up to 0.17–0.20) with acceptable precision; NuNER_Zero and gliner-multitask were worse.
3. **More context helps.** Giving GLiNER the previous + next sentence ("neighbours") raised both precision and discovery (at threshold 0.5: P 0.477 → 0.520, new-R 0.141 → 0.178). Several labels per type did not help.
4. **A small LLM checker (qwen2.5:3b) did not help** – it was no better than simply raising the threshold.
5. **Error analysis.** Of GLiNER's "wrong" terms, ~30 % were actually real AM terms the silver set missed (surface tension, Marangoni flow, grain boundaries…), so measured precision underestimates the truth; ~50 % were junk, mostly fixable by rules (specimen labels, citation numbers, generic phrases). Junk rules raised DEV precision by 0.07 without losing any discovery.
6. **Honest TEST result.** With the best settings GLiNER still lowers entity F1 vs. rules-only on unseen papers (0.454 vs 0.507) because the junk rules were learned from DEV papers (overfitting). **Decision:** the graph's entities come from rules + lexicon; GLiNER is the **discovery layer** that suggests new terms for the lexicon.

**Threshold:** GLiNER gives each span a confidence score; the threshold (0.5) is the cut-off. Higher threshold = fewer, surer entities (precision ↑, discovery ↓). The mechanism is built into GLiNER; we chose the value by experiments on DEV and added per-type thresholds and junk filters on top.

---

## 9. All important numbers

### 9.1 Entities on TEST (silver, rules-only unless stated)

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Old lexicon (142 concepts) | 0.807 | 0.341 | 0.480 |
| Lexicon with standards (174 concepts) | 0.770 | 0.378 | 0.507 |
| **+ automatic lexicon from 30 open-access papers** | 0.758 | 0.384 | **0.510** |
| Old GLiNER (medium, generic labels) + rules | 0.303 | 0.466 | 0.367 |
| Best GLiNER (large, neighbours, LPBF labels) + rules | 0.426 | 0.484 | 0.454 |

### 9.2 Other results

| Item | Result |
|---|---|
| Speed (Phase 1) | 1033 s → ~185 s per paper |
| Relations (10 papers, 3B model) | strict F1 0.301 (P 0.327, R 0.279); relaxed 0.331; accepted-only precision 0.64 |
| Relations pilot, 2 papers | qwen2.5:3b F1 0.273 vs Groq gpt-oss-120b 0.329 |
| Values | F1 ≈ 0.39–0.49 (TEST rules-only 0.391; 10-paper hybrid 0.492) |
| Silver consistency | entities 0.92, relations 0.85/0.88, values 0.91 |
| Neo4j load (10 papers) | 1,055 entities, 1,779 claims, 1,175 edges, 893 findings |
| Automatic lexicon (30 papers) | 1,404 candidate terms → 12 accepted + 1 alias; DEV F1 0.600 → 0.597, TEST 0.507 → 0.510 |

### 9.3 Automatic lexicon – rule v1.1 and its first terms

A new term is added to `config/lexicon_auto.yaml` only if it appears in **≥ 3 papers**, from **≥ 2 different alloys**, with the **same entity type ≥ 70 %** of the time, passes the junk filter, is **not a short abbreviation**, **one-word terms need ≥ 5 papers**, and a large LLM (Groq gpt-oss-120b, free) confirms it is a specific AM concept. Synonyms of existing terms become aliases; rejected terms go to a block list; the rest wait on a watch list for more papers.

First terms: *grain boundary* (11 papers, 6 alloys), *high-angle grain boundary*, *subgrain boundary*, *penetration depth*, *heat capacity*, *surface quality*, *pre-alloyed powder*, *copper powder*, *build plate*, *base plate*, *bare plate*, *single tracks*; alias *crystallographic orientation → crystallographic texture*.
Safety: separate file (curated lexicon untouched), lower confidence (0.8), provenance (papers, DOIs, date, checker), regression check on DEV, undo by renaming one file.

---

## 10. Key findings / lessons

1. A curated, standards-based lexicon + rules is the strongest and most trustworthy entity extractor on unseen papers.
2. Zero-shot AI models (GLiNER) are useful for **discovering new vocabulary**, not as the main extractor.
3. Wording of labels and context matter more than model size for zero-shot NER.
4. Small LLMs (3B) are weak judges; large free models (120B) are much better for checking and for relations.
5. Rules tuned on development papers do not fully transfer to new papers → a fixed TEST set is essential.
6. An LLM-made silver standard is consistent but incomplete; part of the "errors" are real terms it missed.
7. Relations are the weakest part (F1 ≈ 0.3) and the main target for future work.
8. With multi-paper evidence and an LLM check, the lexicon can grow automatically and safely from free papers.

---

## 11. Limitations (say these honestly)

- Evaluation uses a **silver** standard (LLM-annotated), not human gold; only 10 papers (5 TEST).
- Relation extraction is still weak (F1 ≈ 0.30).
- Many publishers (Elsevier, MDPI) cannot be downloaded automatically, which limits the free corpus.
- Tables without lines and scanned PDFs are hard to parse.
- The local 3B LLM is small; better models need free cloud tiers with daily limits.
- Only LPBF metals are covered so far.

---

## 12. Current state and next steps

**Current state (27 Oct 2026):** all code is on GitHub (`phase2-benchmark`, last commit `546178f` + one small pending change for the TEST check). The automatic lexicon (12 terms) is committed. 30 harvested papers are on the Mac (not in git).

**Next steps (all free):**
1. Grow the harvest to 100+ papers and re-run the automatic lexicon.
2. Full re-extraction of the 10 papers with the new lexicon → final numbers.
3. Improve relations (error analysis, larger model for relations, better rules).
4. Fine-tune GLiNER on automatically labelled LPBF sentences (knowledge distillation from a large LLM).
5. Build a PSP (process–structure–property) dataset and a LangGraph research assistant (agents) on top of the graph.
6. Deploy a demo (FastAPI/Docker/Hugging Face Spaces) and write the paper.

---

## 13. Questions people may ask – with answers

**Q1. What is your project in one line?**
An evidence-backed knowledge graph that automatically extracts process–structure–property knowledge from LPBF research papers using rules, a domain lexicon, and free AI models.

**Q2. Why not just ask ChatGPT to read the papers?**
LLMs can invent facts and do not give exact, checkable sources. In our pipeline the LLM only proposes; every fact must be found in the actual sentence, units are verified, and each claim keeps its evidence. It is also free, local and reproducible.

**Q3. What is a knowledge graph and why Neo4j?**
A network of nodes (materials, parameters, properties, papers) and edges (affects, has value, studies). It shows connections such as "laser power → affects → porosity (5 papers)". Neo4j is a free graph database with a query language (Cypher) and a visual browser.

**Q4. What is the lexicon?**
Our domain dictionary: 174 AM concepts with 736 synonyms, organised in 10 entity types, built on ISO/ASTM 52900 and test/measurement standards. It maps all spellings to one standard name.

**Q5. How do the rules work?**
The matcher finds lexicon terms in each sentence; the unit parser reads numbers with units and attaches them to an entity of the right physical dimension; trigger words ("increased", "reduced", "affects") and sentence structure create relations.

**Q6. What is GLiNER and why use it if rules are better?**
A free zero-shot NER model that finds entities from plain-English labels. It finds terms the lexicon does not know. Alone it is noisier than rules, so we use it to discover new terms, which enter the lexicon only after multi-paper evidence and an LLM check.

**Q7. What is the threshold in GLiNER?**
The minimum confidence score (0–1) to keep a prediction. Higher = fewer but surer entities. GLiNER has it built in; we chose 0.5 by experiments on DEV papers.

**Q8. What is precision, recall, F1?**
Precision = how much of what we extracted is correct; recall = how much of the correct answers we found; F1 = the balance of both.

**Q9. Your F1 is about 0.5. Is that good?**
For automatic extraction of a specialised domain with strict matching and an incomplete silver answer key, it is a reasonable baseline, and precision is high (0.76). Part of the "errors" are real terms the silver set missed (~30 % of GLiNER's "false" terms). The main point is the improvement trend under an honest DEV/TEST protocol.

**Q10. What is the silver standard? Why not gold?**
An answer key made by an LLM following a written guideline, without seeing our system's output; a second pass showed high agreement (0.85–0.92). It is not human-verified, so we call it silver. A small human-verified sample would make it gold.

**Q11. What is DEV/TEST and why is it important?**
We tune only on 5 DEV papers and check the final result once on 5 TEST papers. This shows whether the improvements generalise; e.g. GLiNER junk rules worked on DEV but less on TEST – we report that honestly.

**Q12. How do you make sure facts are correct?**
Evidence check (text must be in the sentence), unit and dimension check, ontology check, confidence thresholds (accepted ≥ 0.85, review 0.60–0.85, rejected < 0.60), and every claim links to its sentence and page.

**Q13. How does the lexicon grow automatically?**
The harvester downloads open-access papers; GLiNER finds new terms; a term is added only if it occurs in ≥ 3 papers from ≥ 2 alloys with a consistent type, is not junk or an abbreviation, and a large LLM confirms it. Result from 30 papers: 12 new terms, TEST F1 0.507 → 0.510.

**Q14. Is the paper downloading legal?**
Yes: only open-access papers with Creative-Commons licences, through the OpenAlex API; Elsevier/ScienceDirect are excluded because their terms forbid automated downloads; requests are slow and polite.

**Q15. How much does it cost?**
Nothing. Local models (Ollama, GLiNER), free API tiers (Groq, OpenAlex key), Neo4j Community, Python open-source libraries, a MacBook Air M4 (16 GB).

**Q16. How fast is it?**
Hybrid extraction ≈ 3 minutes per paper (was 17 minutes); rules-only takes seconds; GLiNER-large discovery ≈ 1–2 minutes per paper.

**Q17. What is the weakest part?**
Relation extraction (F1 ≈ 0.3). A larger free model (Groq 120B) already improved the pilot from 0.273 to 0.329; this is the next focus.

**Q18. How is your work related to Prof. Tewari's paper?**
Both extract material properties and structural features from literature with LLM-based methods. AM-KG adds an evidence-backed graph, unit normalisation, an honest benchmark, and automatic lexicon growth, all free; the next stage adds agents/research assistant and a PSP dataset.

**Q19. What is new (novel) here?**
(1) Evidence-backed, verifiable LPBF knowledge graph; (2) standards-based AM lexicon that grows automatically with multi-paper + LLM verification; (3) honest comparison of rules vs. zero-shot NER vs. LLMs on a fixed DEV/TEST silver benchmark; (4) fully free and local.

**Q20. What did you learn from the GLiNER experiments?**
Domain labels and context matter most; large models discover more; small LLM checkers do not help; rules tuned on DEV overfit; zero-shot NER is best used for discovery.

**Q21. Can it scale to thousands of papers?**
Yes: each paper is processed independently, results are cached, the harvester and discovery are resumable, and Neo4j handles large graphs. The limits are free download sources and LLM daily quotas.

**Q22. Which AI models are used?**
GLiNER large v2.1 (entities, discovery), Ollama qwen2.5:3b (local LLM), Groq gpt-oss-120b (relations pilot, lexicon verification).

**Q23. What are entities and relations? Give an example.**
Sentence: "Increasing laser power to 200 W reduced porosity in Ti-6Al-4V." Entities: laser power (Parameter), porosity (Defect), Ti-6Al-4V (Material); value: 200 W → laser power; relation: laser power → AFFECTS (decrease) → porosity, with the sentence as evidence.

**Q24. What happens with different names for the same thing?**
Normalisation maps them to one canonical name (SLM, L-PBF → LPBF; Ti64 → Ti-6Al-4V) and units to SI-style units.

**Q25. What will you do next?**
Scale to 100+ papers, final full evaluation, improve relations, fine-tune GLiNER, build the PSP dataset and research assistant, deploy a demo, and write the paper.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| LPBF / L-PBF / SLM | Laser Powder Bed Fusion – metal 3D printing with a laser melting powder layer by layer |
| PSP | Process–Structure–Property relationships |
| Entity | A named thing: material, parameter, property, defect… |
| Relation / claim | A link between entities with evidence, e.g. laser power AFFECTS porosity |
| Lexicon | Our dictionary of AM terms and synonyms |
| Normalisation | Mapping different names/units to one standard |
| NER | Named-Entity Recognition (finding entities in text) |
| Zero-shot | A model works on new labels without training on them |
| GLiNER | Free zero-shot NER model |
| LLM | Large Language Model (qwen, gpt-oss) |
| Ollama | Free tool to run LLMs locally |
| Groq | Free-tier cloud API for large open models |
| Neo4j / Cypher | Graph database / its query language |
| Silver standard | LLM-made answer key (not human-verified) |
| DEV / TEST | Development papers (tuning) / held-out papers (final check) |
| Precision / Recall / F1 | Correctness / coverage / their balance |
| Threshold | Minimum confidence to keep a prediction |
| OpenAlex | Free database/API of scholarly papers |
| Overfitting | Rules that work on the papers they were built from but not on new ones |

---

## 15. Where things are (file map) and main commands

| Path | What |
|---|---|
| `main.py` | Command-line entry (extract, build-graph, evaluate, benchmark, ask…) |
| `app/streamlit_app.py` | Web app incl. Neo4j Explorer |
| `config/lexicon.yaml` | Curated lexicon (174 concepts) |
| `config/lexicon_auto.yaml` | Automatically discovered terms (12 + alias) |
| `config/settings.yaml` | All settings (models, thresholds, paths) |
| `src/` | Pipeline code (ingestion, extraction, normalisation, quality, graph, llm, evaluation) |
| `scripts/gliner_sweep.py` | GLiNER experiments (models, context, checker, DEV/TEST) |
| `scripts/entity_gaps.py`, `scripts/gliner_error_analysis.py` | Error analyses |
| `scripts/harvest_oa.py` | Free open-access paper harvester |
| `scripts/lexicon_discovery.py` | Automatic lexicon growth (rule v1.1) |
| `docs/` | Architecture, ontology, benchmark, silver guideline, Neo4j setup |
| `data/annotations_silver/` | Silver standard |
| `data/experiments/` | Experiment results (not in git) |
| `data/harvest/` | Harvested papers + discovery results (not in git) |

```bash
cd ~/"mtech project Additive manufacturing/am-kg" && conda deactivate; source .venv/bin/activate
python -m pytest -q                                              # run all tests
python main.py extract data/papers/lpbf --mode hybrid            # extract the 10 papers
python main.py build-graph --backend neo4j --real-only           # load Neo4j
streamlit run app/streamlit_app.py                               # web app
python scripts/harvest_oa.py --max 100                           # download open-access papers
python scripts/lexicon_discovery.py --check                      # grow the lexicon + DEV check
python scripts/lexicon_discovery.py --check-only --check-split test   # final TEST number
```

*Secrets (GROQ_API_KEY, NEO4J_PASSWORD, OPENALEX_API_KEY) live only in `.env`, which is never shared or committed.*
