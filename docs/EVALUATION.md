# Evaluation protocol

`python main.py evaluate` compares the knowledge base with the gold annotations in `data/annotations/*.json`. It writes `data/eval/evaluation_report.{md,json}`.

## Annotating real papers (do this for your thesis)

The recommended route is Label Studio with pre-annotations and the `benchmark` command: see [BENCHMARK.md](BENCHMARK.md). The manual JSON format below is what `annotate-import` produces.

1. Pick 5–10 real AM papers that you are allowed to use. Put the PDFs in `data/papers/` and run extraction.
2. For each paper, create `data/annotations/<pdf-file-stem>.json` in the format of `demo_01_lpbf_ti64.json`:
   - `entities`: the document-level set of `{type, canonical}`
   - `relations`: `{subject, relation, object, evidence}`, one per distinct relation; evidence is one supporting sentence
   - `values`: `{entity, normalized_value | normalized_min + normalized_max, normalized_unit}`
   - `normalization`: `{surface, type, canonical}` pairs taken from the paper
3. Annotate **from the paper, before looking at the system output**. Ideally have a second annotator do 2 papers and report inter-annotator agreement (Cohen's κ on relations).
4. Use the canonical names from `config/lexicon.yaml`. If an entity is not in the lexicon, write the name a domain expert would use.

## Metrics

| Metric | Definition |
|---|---|
| Entity P/R/F1 | set match on (type, canonical name) per paper (document level) |
| Relation P/R/F1 (strict) | set match on (subject, relation, object) canonical triples; DERIVED excluded by default |
| Relation (lenient pair) | the unordered entity pair matches, ignoring the relation label |
| Value P/R/F1 | (entity, normalised value or range, unit) within 1 % tolerance |
| Normalisation accuracy | normaliser output on the gold `surface → canonical` pairs |
| Evidence accuracy | share of matched gold relations with at least one supporting claim citing the annotated sentence (≥ 60 % token overlap) |
| Citation support rate | share of predicted claims whose evidence text contains both endpoint surface forms |
| Graph quality | nodes, aggregated relationships, duplicate-entity rate, unsupported-relation rate (evidence not verifiable in the source), new entities pending review |

Options:

- `--accepted-only` evaluates only auto-accepted claims. Use it to see the precision/recall trade-off at your thresholds.
- `--include-derived` also scores DERIVED relations.

To compare extractors, run `extract --mode rule`, then `--mode hybrid`, then `--mode llm` into separate copies of the project (or back up `data/extracted` between runs), and evaluate each.

## Current numbers

The only gold annotations shipped are for 2 **DEMO / SYNTHETIC** documents. Scores on them, for example entity F1 = 1.0 and relation F1 ≈ 0.95, **only show that the pipeline works**. The demo text and the lexicon were written together, so they are optimistic. They are not a scientific result.
