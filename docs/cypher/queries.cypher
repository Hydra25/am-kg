// AM-KG example Cypher queries (Neo4j 5.x)
// Parameters: $statuses = ['accepted','review'] (or ['accepted']), $effect_rels = ['AFFECTS','INFLUENCES','CAUSES','IMPROVES','CHANGES','PRODUCES']
// Entity parameters are canonical ids, e.g. $material_id = 'Material:Ti-6Al-4V', $process_id = 'Process:LPBF'.
// In Neo4j Browser set them first, e.g.:  :param statuses => ['accepted','review']

// ---- processes_for_material: 1. All AM processes used for a material
MATCH (pr:Process)-[r:PROCESSES]->(m:Material {id: $material_id})
WHERE r.status IN $statuses
RETURN pr.canonical_name AS process, size(r.paper_ids) AS papers, r.support_count AS claims, r.status AS status
ORDER BY papers DESC, claims DESC;

// ---- parameters_for_process: 2. All parameters reported for a process (with example values)
MATCH (pr:Process {id: $process_id})-[r:HAS_PARAMETER]->(pa:Parameter)
WHERE r.status IN $statuses
OPTIONAL MATCH (pa)-[hv:HAS_VALUE]->(v:ParameterValue)
WHERE v.paper_id IN r.paper_ids AND hv.status IN $statuses
RETURN pa.canonical_name AS parameter, size(r.paper_ids) AS papers, r.support_count AS claims,
       collect(DISTINCT v.raw)[..10] AS example_values
ORDER BY papers DESC, claims DESC;

// ---- studies_linking: 3. Studies with an evidence-backed claim connecting two entities (e.g. laser power & porosity)
MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.status IN $statuses
  AND ((c.subject_id = $a_id AND c.object_id = $b_id) OR (c.subject_id = $b_id AND c.object_id = $a_id))
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, c.relation AS relation,
       c.subject_name AS subject, c.object_name AS object, c.trend AS trend, c.negated AS negated,
       c.hedged AS hedged, c.knowledge_type AS knowledge_type, c.evidence AS evidence, c.page AS page,
       c.section AS section, c.confidence AS confidence, c.status AS status, c.claim_id AS claim_id,
       p.is_synthetic AS synthetic
ORDER BY year DESC;

// ---- property_values_for_material: 4. Reported values of a property (e.g. tensile strength) for a material
MATCH (p:Paper)-[s:STUDIES]->(m:Material {id: $material_id})
WHERE s.status IN $statuses
MATCH (p)-[:SUPPORTS]->(c:Claim {relation: 'HAS_VALUE', subject_id: $property_id})
WHERE c.status IN $statuses
MATCH (v:Value {id: c.object_id})
OPTIONAL MATCH (p)-[s2:STUDIES]->(m2:Material) WHERE s2.status IN $statuses
WITH p, c, v, count(DISTINCT m2) AS n_materials
RETURN p.paper_id AS paper_id, p.title AS title, v.raw AS value, v.normalized_value AS normalized_value,
       v.normalized_min AS normalized_min, v.normalized_max AS normalized_max, v.normalized_unit AS normalized_unit,
       v.state AS material_state, c.evidence AS evidence, c.page AS page, c.section AS section,
       c.status AS status, c.claim_id AS claim_id, n_materials > 1 AS context_ambiguous;

// ---- papers_with_material_and_process: 5. Papers using both a material and a process
MATCH (p:Paper)-[s:STUDIES]->(:Material {id: $material_id}), (p)-[u:USES_PROCESS]->(:Process {id: $process_id})
WHERE s.status IN $statuses AND u.status IN $statuses
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, p.doi AS doi, p.is_synthetic AS synthetic;

// ---- relations_between: 6. Direct claims and two-hop paths between two entities (e.g. scan speed & tensile strength)
CALL {
  MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
  WHERE c.status IN $statuses AND c.relation IN $effect_rels
    AND ((c.subject_id = $a_id AND c.object_id = $b_id) OR (c.subject_id = $b_id AND c.object_id = $a_id))
  RETURN 'direct claim' AS path_type, null AS via, c.relation AS rel1, null AS rel2,
         [p.paper_id] AS papers1, null AS papers2, c.evidence AS evidence
  UNION
  MATCH (a:Entity {id: $a_id})-[r1]-(x:Entity)-[r2]-(b:Entity {id: $b_id})
  WHERE type(r1) IN $effect_rels AND type(r2) IN $effect_rels
    AND r1.status IN $statuses AND r2.status IN $statuses AND x <> a AND x <> b
  RETURN 'two-hop path (DERIVED, not stated by a paper)' AS path_type, x.canonical_name AS via,
         type(r1) AS rel1, type(r2) AS rel2, r1.paper_ids AS papers1, r2.paper_ids AS papers2, null AS evidence
}
RETURN path_type, via, rel1, rel2, papers1, papers2, evidence;

// ---- papers_reporting_both: 7. Papers mentioning two entities together (e.g. porosity and laser power)
MATCH (p:Paper)-[ma:MENTIONS]->(:Entity {id: $a_id}), (p)-[mb:MENTIONS]->(:Entity {id: $b_id})
OPTIONAL MATCH (p)-[:SUPPORTS]->(c:Claim)
WHERE c.status IN $statuses
  AND ((c.subject_id = $a_id AND c.object_id = $b_id) OR (c.subject_id = $b_id AND c.object_id = $a_id))
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, ma.count AS mentions_a, mb.count AS mentions_b,
       count(c) AS direct_claims, p.is_synthetic AS synthetic
ORDER BY direct_claims DESC;

// ---- understudied_combinations: 8. Material x Process x Parameter x Property/Defect combinations with paper counts
MATCH (p:Paper)-[s:STUDIES]->(m:Material), (p)-[u:USES_PROCESS]->(pr:Process), (p)-[:SUPPORTS]->(c:Claim)
WHERE s.status IN $statuses AND u.status IN $statuses AND c.status IN $statuses
  AND c.relation IN ['AFFECTS', 'INFLUENCES'] AND c.subject_type = 'Parameter'
WITH m.canonical_name AS material, pr.canonical_name AS process, c.subject_name AS parameter,
     c.object_name AS response, collect(DISTINCT p.paper_id) AS papers
RETURN material, process, parameter, response, size(papers) AS n_papers, papers
ORDER BY n_papers ASC, material, process;

// ---- why_claim: Provenance: why does the graph believe this claim?
MATCH (p:Paper)-[:SUPPORTS]->(c:Claim {claim_id: $claim_id})
RETURN p.paper_id AS paper_id, p.title AS title, c.relation AS relation, c.subject_name AS subject,
       c.object_name AS object, c.evidence AS evidence, c.page AS page, c.section AS section,
       c.sentence_id AS sentence_id, c.source_type AS source_type, c.method AS method,
       c.knowledge_type AS knowledge_type, c.confidence AS confidence, c.status AS status,
       c.qc_reasons AS qc_reasons, c.reviewed AS reviewed;

// ---- entity_neighbors: Graph explorer: all relations of one entity
MATCH (e:Entity {id: $entity_id})-[r]-(n)
WHERE NOT type(r) IN ['MENTIONS', 'SUBJECT', 'OBJECT', 'STUDIES', 'USES_PROCESS'] AND r.status IN $statuses
RETURN type(r) AS relation, CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction,
       coalesce(n.canonical_name, n.raw) AS neighbor, n.entity_type AS neighbor_type, n.id AS neighbor_id,
       r.support_count AS claims, size(r.paper_ids) AS papers
ORDER BY relation, papers DESC;

// ---- ad hoc: everything the graph knows about laser power -> porosity, with evidence
MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.subject_id = 'Parameter:laser power' AND c.object_id = 'Defect:porosity'
RETURN p.title, c.relation, c.trend, c.negated, c.page, c.section, c.evidence, c.status;

// ---- ad hoc: visualise the neighbourhood of Ti-6Al-4V in Neo4j Browser
MATCH (m:Material {id: 'Material:Ti-6Al-4V'})<-[r:PROCESSES]-(pr:Process)-[r2:HAS_PARAMETER]->(pa:Parameter)
RETURN m, r, pr, r2, pa;
