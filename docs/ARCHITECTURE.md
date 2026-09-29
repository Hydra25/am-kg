# Architecture

## Design principles

1. **Evidence first.** Every claim stores paper, page, section, sentence id and the verbatim sentence or table row. The quality controller re-checks that text against the parsed document. A claim that can't be verified is rejected.
2. **The LLM proposes and the code verifies.** Qwen output is used only after checks: entity text must occur in the cited sentence, values must occur and parse with a known unit, the unit's dimension must fit the entity, endpoints must be in the sentence, and the relation must satisfy the ontology.
3. **Epistemic types are never mixed.** Each claim is one of `EXPLICIT`, `EXTRACTED`, `DERIVED` or `INTERPRETED` (see ONTOLOGY.md). `INTERPRETED` text only ever appears in assistant answers, with its own label.
4. **Reproducible.** Ids are deterministic (hashes of paper, sentence and entities). Each extraction records the pipeline version, mode, model name, prompt hashes, lexicon hash and thresholds.
5. **Degrades gracefully.** Without Ollama the pipeline runs rule-based. Without Neo4j it uses a local graph store. Without GROBID it parses with PyMuPDF. Without Chroma it searches with BM25.

## Data flow

```
PDF / .txt / .md
  │  src/ingestion/loader.py         GROBID (optional) or PyMuPDF (src/pdf_parser)
  ▼                                  columns, headers/footers, ligatures, tables, metadata
ParsedDocument ──► data/processed/<paper_id>.json
  │  src/section_parser              heading detection, section typing, page-tagged sentences
  │                                  table rows -> "row sentences" (source_type=table)
  ▼
Rule extraction (always)             src/entity_extraction/rule_based.py + matcher.py
  │  lexicon matching (longest match), units, value->entity assignment by physical dimension
  │  src/relation_extraction/rule_based.py
  │     EXPLICIT: causal/effect predicates, passive voice, "effect of X on Y", negation, hedging,
  │               coordination, background/citation detection
  │     EXTRACTED: HAS_VALUE, HAS_PARAMETER, HAS_CONDITION, PROCESSES, FABRICATES,
  │                PRODUCES, CHARACTERIZES (with explicit context resolution)
  ▼
LLM extraction (hybrid/llm modes)    src/entity_extraction/llm_extractor.py
  │  chunked prompts (prompts/*.txt) -> strict JSON -> repair/retry -> verification
  ▼
Derived relations                    src/relation_extraction/derived.py
  │  paper-level STUDIES / USES_PROCESS, monotonic trends across table rows
  ▼
Normalisation                        src/normalization (lexicon, overrides, fuzzy, LLM-choice, units)
Merge + quality control              src/pipeline.py, src/quality/validator.py
  ▼
ExtractionResult ──► data/extracted/<paper_id>.json   (includes rejected claims for audit)
  │
  ▼  src/knowledge_graph/kb.py        corpus view + human review decisions + entity merges
KnowledgeBase
  ├─► Neo4jGraphStore / LocalGraphStore (src/knowledge_graph/store.py)
  ├─► Named queries (queries.py: Cypher + equivalent Python)
  ├─► ResearchAssistant (src/retrieval): entity linking → graph claims + passages → Qwen → citation check
  ├─► Gap analysis (src/analysis/gaps.py)
  └─► Evaluation (src/evaluation/metrics.py)
```

## Module map

| Folder | Responsibility |
|---|---|
| `src/config.py` | settings.yaml + `.env`, all paths relative to the project root |
| `src/ontology/` | entity/relation enums, allowed signatures, Pydantic models |
| `src/ingestion/` | input discovery, parsing dispatch, metadata, table rows, demo PDF generator |
| `src/pdf_parser/` | PyMuPDF parser (layout, tables), GROBID client + TEI parser |
| `src/section_parser/` | heading detection and section typing, sentence splitting |
| `src/entity_extraction/` | lexicon matcher, rule extractor, LLM extractor |
| `src/relation_extraction/` | rule relations, derived relations, findings |
| `src/normalization/` | lexicon, entity normaliser, units, text utilities |
| `src/llm/` | Ollama client (swappable provider), JSON repair/validation/retry, prompt library |
| `src/quality/` | accept / review / reject rules |
| `src/evidence/` | evidence verification and provenance formatting |
| `src/review/` | append-only human review log |
| `src/knowledge_graph/` | KB, graph export, Neo4j + local stores, named queries |
| `src/retrieval/` | BM25 / Chroma passage index, Graph-RAG assistant |
| `src/analysis/` | experimental research-gap candidates |
| `src/evaluation/` | P/R/F1, normalisation, evidence and graph-quality metrics |
| `app/` | Streamlit UI |

## Scaling to hundreds or thousands of papers

- Extraction is per paper and writes one JSON file per paper. `process_many` skips a bad PDF and continues. Runs can be parallelised across processes (for example `xargs -P`) because nothing is shared.
- Rule extraction takes about 0.1 s per paper. The bottleneck is the LLM: roughly 3 calls per ~2,200-character chunk. Use `--mode rule` for a fast first pass, or run hybrid overnight.
- Neo4j loading uses batched `UNWIND` with uniqueness constraints and indexes on claim endpoints.
- Extension points:
  - persist the BM25 index (it is rebuilt in memory today)
  - switch to `VECTOR_BACKEND=chroma` for large corpora
  - replace the lexicon matcher's per-alias regexes with a single Aho–Corasick automaton if matching becomes a bottleneck

## Known limitations (read before publishing results)

- **Scanned PDFs** need OCR (for example `ocrmypdf`) before ingestion. The parser warns when little text is found.
- **Tables:** tables without ruling lines may be missed by `find_tables()`. GROBID (or Camelot or a table-transformer model) can be plugged in through `extract_tables()`.
- **The rule relations are English-pattern based.** They favour precision. Long-distance and cross-sentence relations are only partly covered (previous-sentence process resolution). The LLM step exists to raise recall.
- **Paper-level context:** a paper that studies several materials makes value-to-material attribution ambiguous. Queries flag this as `context_ambiguous`.
- **LLM confidences** from a 3B model are poorly calibrated. The final confidence therefore combines the model score with evidence overlap and normalisation quality.
- **Scores on the synthetic demo corpus** only show that the pipeline works. The demo text and the lexicon were written together, so those scores are optimistic. Report scores on manually annotated real papers.
