# Neo4j: run the knowledge graph locally (free)

The pipeline works without Neo4j (local JSON graph). Neo4j adds the interactive graph browser and Cypher queries.
Everything below is free: Neo4j Community Edition runs on your own Mac.

## 1. Install and start (macOS, Homebrew)

```bash
brew install neo4j                                   # Community Edition (+ Java, installed automatically)
neo4j-admin dbms set-initial-password 'choose-a-password'   # once, before the first start (min. 8 characters)
neo4j start                                          # or: brew services start neo4j  (starts at login)
neo4j status
```

Neo4j Browser: <http://localhost:7474>, Bolt: `bolt://localhost:7687`, user `neo4j`.
Stop it with `neo4j stop` (or `brew services stop neo4j`).

Alternatives: Neo4j Desktop (free app), or Neo4j AuraDB Free (cloud, no install; use its `neo4j+s://…` URI).

## 2. Tell the project where Neo4j is (`.env`, never in git)

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=choose-a-password
```

Check: `python main.py check` should print `Neo4j: connected (ok)`.

## 3. Load the graph

```bash
python main.py build-graph --backend neo4j --real-only    # --real-only leaves out the synthetic DEMO papers
```

The command deletes the previous AM-KG nodes and loads papers, entities, values, claims (with evidence),
aggregated relationships and findings. Rejected claims are not loaded. On the 10 LPBF papers this takes seconds.

## 4. Explore

1. Open <http://localhost:7474> and log in.
2. Colours: drag `docs/cypher/style.grass` onto the Browser window (one colour per entity type, names as captions).
3. Paste the queries from `docs/cypher/demo.cypher` one at a time (Cmd+Enter):
   overview → papers/alloys → parameter–property map → what affects porosity → evidence sentences → values → research gaps.

Graph model: [NEO4J_SCHEMA.md](NEO4J_SCHEMA.md). More queries: `python main.py queries` and
[cypher/queries.cypher](cypher/queries.cypher).
