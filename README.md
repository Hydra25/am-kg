# AM-KG: Additive Manufacturing Knowledge Graph & Research Assistant

This project turns Additive Manufacturing (AM) research papers into an **evidence-backed knowledge graph**. Every fact in the graph can be traced back to paper → page → section → sentence or table row.

```
PDF → text & sections → entities → relations → normalisation → quality control
    → Knowledge Graph (Neo4j or local) → queries / Graph-RAG assistant (local Qwen) / gap analysis
```

- **Local and free.** It uses Ollama with `qwen2.5:3b` (configurable) and no paid API.
- **Works without an LLM.** A deterministic rule-based extractor runs first. The LLM adds recall in `hybrid` mode, and its proposals are *verified against the text* before they are used.
- **Works without Neo4j.** A local graph store is used automatically when `NEO4J_URI` is empty.
- **Keeps knowledge types apart.** Claims are marked `EXPLICIT`, `EXTRACTED` or `DERIVED`. LLM explanations appear only as labelled `INTERPRETED` text.
- **Human in the loop.** You can accept, reject, edit or merge in the UI. Every decision goes to an append-only audit log.

> The 5 papers in `data/demo/` (rendered to `data/papers/*.pdf`) are **DEMO / SYNTHETIC** texts written to test the pipeline. They are **not real research** and every one is labelled as such. Put real papers in `data/papers/` for scientific work.

---

## 1. Setup (macOS, Apple Silicon)

```bash
cd "$HOME/mtech project Additive manufacturing/am-kg"

# Python 3.10+ (3.11 or 3.12 recommended; `brew install python@3.12` if needed)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# optional: dense retrieval + notebook
pip install -r requirements-optional.txt

cp .env.example .env        # then edit .env if needed
```

## 2. Start Ollama and pull Qwen

```bash
# Either open the Ollama app, or run the server in its own terminal:
ollama serve
ollama pull qwen2.5:3b
# optional, only if VECTOR_BACKEND=chroma:
ollama pull nomic-embed-text
```

To use another local model, set `OLLAMA_MODEL=qwen2.5:7b` (or any Ollama model) in `.env`.

## 3. Start Neo4j (optional)

Option A, Docker:

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/amkg-password neo4j:5
```

Option B, Homebrew:

```bash
brew install neo4j && neo4j start
# then open http://localhost:7474 and set a password
```

Then put this in `.env`:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=amkg-password
```

Leave `NEO4J_URI` empty to use the local graph store.

Check that everything is reachable:

```bash
python main.py check
```

## 4. Run the pipeline

```bash
# (a) create the DEMO/SYNTHETIC PDFs (optional; skip if you only use real papers)
python main.py make-demo

# (b) parse only (sections, sentences, tables) -> data/processed/
python main.py parse data/papers/demo_01_lpbf_ti64.pdf

# (c) extract one PDF or a whole folder -> data/extracted/<paper_id>.json
python main.py extract data/papers/demo_01_lpbf_ti64.pdf
python main.py extract data/papers --mode auto     # auto = hybrid if Ollama is up, else rule
python main.py extract data/papers --mode rule     # fast, deterministic
python main.py extract data/papers --mode hybrid   # rule + Qwen (slower; ~3 LLM calls per chunk)

# Every LLM run prints where the time went, e.g.
#   LLM time 812.4s | chunks 27/33 sent (6 skipped by gating) | cache hits 0 | tokens in/out 61234/18873
#     - entities      calls=27  cached=0   time=  402.1s  out_tokens=9921
#     - relations     calls=26  ...
# The same numbers are stored in data/extracted/<paper_id>.json under "stats".

# (d) build the graph (Neo4j if configured, else data/graph/graph.json)
python main.py build-graph

# or extract + build in one step
python main.py process data/papers
```

### Faster extraction (Phase 1)

| Mode / option | What proposes entities | Relations & findings | Speed |
|---|---|---|---|
| `--mode rule` | lexicon + regex | regex rules | seconds, no findings beyond rules |
| `--mode gliner` | lexicon + GLiNER | regex rules | ~1 min, no LLM needed |
| `--mode hybrid --entities gliner` | lexicon + GLiNER | rules + LLM | recommended |
| `--mode hybrid --entities llm` | lexicon + LLM prompt | rules + LLM | slowest (original) |

GLiNER is optional: `pip install -r requirements-optional.txt`. The model (~0.5 GB) downloads once from Hugging Face.
With `extraction.entity_backend: auto` it is used automatically once it is installed.

To use a free cloud LLM (Groq, Cerebras, Gemini, OpenRouter) instead of Ollama, set `LLM_PROVIDER`, `LLM_MODEL` and `LLM_API_KEY` in `.env` (see `.env.example`).
Ollama stays the default. Only send published papers to a cloud service.

### Benchmark (Phase 2)

Build a gold standard with Label Studio and compare the extraction modes. See [docs/BENCHMARK.md](docs/BENCHMARK.md).

```bash
python main.py annotate-export data/papers/lpbf      # tasks + labeling config -> data/labelstudio/
python main.py annotate-import export.json           # Label Studio export -> data/annotations/
python main.py benchmark data/papers/lpbf            # -> data/benchmark/results.md
```

## 5. Explore

```bash
python main.py stats
python main.py queries                                  # list named queries
python main.py query processes_for_material --param material=Ti6Al4V
python main.py query parameters_for_process --param process=SLM --show-cypher
python main.py query studies_linking --param a="laser power" --param b=porosity
python main.py query property_values_for_material --param material=Ti-6Al-4V --param property="tensile strength"
python main.py query papers_with_material_and_process --param material=Ti64 --param process=LPBF
python main.py query relations_between --param a="scan speed" --param b="tensile strength"
python main.py query papers_reporting_both --param a=porosity --param b="laser power"
python main.py query understudied_combinations
python main.py why <claim_id>                           # "why does the KG believe this?"

python main.py ask "What parameters affect porosity in LPBF Ti-6Al-4V?"
python main.py ask "Which parameters affect tensile strength of Ti-6Al-4V in LPBF?" --no-llm

python main.py review list                              # claims needing review
python main.py review accept <claim_id> --note "checked p.5"
python main.py review merge "Parameter:preheat temp" --into "Parameter:preheating temperature"
python main.py build-graph                              # apply review decisions to the graph

python main.py evaluate                                 # needs data/annotations/*.json
python main.py gaps --exclude-synthetic                 # experimental
```

Entity parameters accept any spelling the normaliser knows: `Ti6Al4V`, `Ti–6Al–4V`, `SLM`, `laser powder bed fusion`, and so on.

## 6. Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app has these pages:

- **Upload & Extract:** entities, relations, confidence and evidence for a new paper
- **Review:** accept, reject, edit or merge
- **Knowledge Graph:** interactive graph
- **Graph Explorer:** choose a material, process, parameter, property or defect and see what it connects to
- **Research Assistant**
- **Paper Evidence:** page, section and sentence for every claim, plus "why"
- **Queries:** named queries, with their Cypher
- **Research Gaps**
- **Evaluation**

## 7. Tests

```bash
python -m pytest
```

99 tests cover:

- units and normalisation
- sections and sentences
- the worked example from the brief
- causal direction, negation and hedging
- JSON repair and retry
- LLM hallucination guards (with a fake LLM)
- quality-control thresholds
- the PDF end-to-end pipeline
- local queries and review decisions
- the Neo4j loader (fake driver)
- assistant citation checking
- gap wording, evaluation and GROBID TEI parsing
- the Ollama HTTP client and hybrid mode (fake Ollama server)
- LLM speed features: timing/token metering, the disk cache, rule-first gating and per-stage token caps
- the GLiNER entity backend (fake model) and the junk-entity filter
- the OpenAI-compatible cloud client (fake server, rate-limit retry, key from env only)
- the Label Studio round trip (export with pre-annotations, import to gold) and the mode benchmark

**No test needs a running Ollama or Neo4j.**

## 8. Project layout

```
config/settings.yaml      thresholds, modes, paths          config/lexicon.yaml  canonical entities + aliases
prompts/*.txt             LLM prompts (hash-recorded)       main.py              CLI
src/                      pipeline modules (see docs/ARCHITECTURE.md)
app/streamlit_app.py      UI                                notebooks/           walkthrough notebook
data/papers  processed  extracted  graph  review  annotations  eval  demo
docs/                     ARCHITECTURE, ONTOLOGY, NEO4J_SCHEMA, EVALUATION, cypher/*.cypher, examples/
tests/
```

## 9. Configuration reference

- `config/settings.yaml`:
  - `quality.accept_threshold`: 0.85 → accepted
  - `quality.review_threshold`: 0.60 → review; anything lower is rejected
  - `extraction.mode`, `extraction.chunk_chars`
  - `extraction.llm_skip_sections`: sections parsed but never sent to the LLM (front matter, title)
  - `extraction.llm_gate`: send a chunk to the LLM only if the rule-based pass found an AM entity or a number with a unit in it (abstract and conclusion are always sent)
  - `llm.cache`: reuse answers for identical prompts from `data/cache/llm/`. A re-run of an unchanged paper makes no model calls. Disable with `LLM_CACHE=false`.
  - `llm.num_predict` (max output tokens) and `llm.keep_alive` (how long Ollama keeps the model loaded)
  - `llm.num_ctx`: 8192. Ollama's default of 2048 silently truncates prompts.
  - `graph.insert_review_items`
  - `retrieval.vector_backend`
  - `gaps.*`
- `.env` overrides connections and modes: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `NEO4J_*`, `GROBID_URL`, `EXTRACTION_MODE`, `GRAPH_BACKEND`, `VECTOR_BACKEND`, `AMKG_ACCEPT_THRESHOLD`, `AMKG_REVIEW_THRESHOLD`.
- Optional GROBID for better scientific-PDF structure: `docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1`, then set `GROBID_URL=http://localhost:8070`.

## 10. Honest status

**Verified** in development:

- the full rule-based pipeline on the demo PDFs
- the local graph store and all named queries
- the review workflow
- the assistant (graph-only mode, and LLM mode with a fake model)
- gap analysis and evaluation
- the Streamlit pages (with Streamlit's AppTest)
- the Ollama client and hybrid pipeline against a fake Ollama HTTP server
- the Neo4j loader's generated Cypher and parameters (against a fake driver)

**Not yet verified:**

- **Real Qwen output quality.** Run `extract --mode hybrid` on your papers and look at `data/extracted/llm_audit/` for dropped proposals.
- **A live Neo4j server.** Run `build-graph` once with Neo4j up. The Cypher follows Neo4j 5 syntax; please report any error.

See the limitations section in `docs/ARCHITECTURE.md`.
