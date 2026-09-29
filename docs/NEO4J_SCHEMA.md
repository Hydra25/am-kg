# Neo4j schema

Created by `python main.py build-graph` (with `NEO4J_URI` set). Constraints and indexes: [cypher/schema.cypher](cypher/schema.cypher). Example queries: [cypher/queries.cypher](cypher/queries.cypher).

## Nodes

| Label(s) | Key | Main properties |
|---|---|---|
| `:Paper` | `paper_id` | title, authors, year, journal, doi, source_file, abstract, is_synthetic |
| `:Entity:<Type>` (for example `:Entity:Material`) | `id` = `"<Type>:<canonical name>"` | canonical_name, entity_type, aliases, alias_text, family, in_lexicon, normalization_confidence, dimensions, paper_count |
| `:Value:ParameterValue` / `:Value:PropertyValue` | `id` | raw, value, value_min, value_max, uncertainty, operator, unit, normalized_value, normalized_min, normalized_max, normalized_unit, dimension, paper_id, of_entity, context_materials, context_processes, state |
| `:Claim` | `claim_id` | relation, subject_id/name/type, object_id/name/type, **evidence**, page, section, section_type, sentence_id, context_sentence_ids, source_type, table_id, confidence, status, method, knowledge_type, trend, negated, hedged, claim_type, claim_origin, qc_reasons, reviewed, attributes_json |
| `:Finding` | `finding_id` | text (verbatim), paper_id, page, section, finding_type, confidence, status |

## Relationships

```
(:Paper)-[:SUPPORTS]->(:Claim)-[:SUBJECT]->(subject node)
                        (:Claim)-[:OBJECT]->(object node)
(:Paper)-[:REPORTS]->(:Finding)-[:EXPRESSES]->(:Claim)
(:Paper)-[:MENTIONS {count}]->(:Entity)
(:Paper)-[:STUDIES]->(:Material)          (:Paper)-[:USES_PROCESS]->(:Process)
(:Process)-[:PROCESSES]->(:Material)      (:Process)-[:HAS_PARAMETER]->(:Parameter)
(:Parameter)-[:HAS_VALUE]->(:ParameterValue)   (:Property)-[:HAS_VALUE]->(:PropertyValue)
(:Parameter)-[:AFFECTS|INFLUENCES]->(:Property|:Defect|:Microstructure)
(:PostProcessing)-[:IMPROVES|CHANGES|AFFECTS|INFLUENCES]->(...)
(:CharacterizationMethod)-[:CHARACTERIZES]->(...)
```

The semantic edges (`PROCESSES`, `AFFECTS`, …) are **aggregates**: one edge for each (subject, relation, object), with these properties:

- `claim_ids`
- `paper_ids`
- `support_count`
- `max_confidence`
- `status` (accepted if any supporting claim is accepted)
- `knowledge_types`
- `trends`
- `negated_count`
- `hedged_count`

The evidence itself lives on the `:Claim` nodes. That is how the graph answers "why does it believe this?":

```cypher
MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.subject_id = 'Parameter:laser power' AND c.object_id = 'Defect:porosity'
RETURN p.title, c.page, c.section, c.evidence, c.knowledge_type, c.status
```

Rejected claims are never loaded into Neo4j. They stay in `data/extracted/*.json` as an audit trail. Claims with status `review` are loaded with `status='review'` (set `graph.insert_review_items: false` to exclude them). Filter with `WHERE r.status IN ['accepted']` for accepted-only analyses.
