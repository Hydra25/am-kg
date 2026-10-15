// AM-KG demo queries for Neo4j Browser (http://localhost:7474)
// Build the graph first:  python main.py build-graph --backend neo4j --real-only
// Paste ONE query at a time into the Browser and press Ctrl/Cmd+Enter.
// Queries that RETURN nodes/relationships/paths are drawn as a graph; the others as a table.
// Edge property `status`: 'accepted' (confidence >= 0.85) or 'review' (0.60-0.85). Rejected claims are not loaded.

// ---- 0. What is in the graph? (table)
MATCH (n) UNWIND labels(n) AS label
RETURN label, count(*) AS nodes ORDER BY nodes DESC;

// ---- 1. The papers, their alloys and processes (graph)
MATCH (p:Paper)-[r:STUDIES|USES_PROCESS]->(e:Entity)
WHERE r.status = 'accepted'
RETURN p, r, e;

// ---- 2. Process -> parameter -> property/defect map, effects reported by at least 2 papers (graph)
MATCH (a:Entity)-[r:AFFECTS|INFLUENCES|CAUSES|IMPROVES|CHANGES]->(b:Entity)
WHERE size(r.paper_ids) >= 2
RETURN a, r, b;

// ---- 3. Everything the graph knows that affects porosity (graph)
MATCH (x:Entity)-[r:AFFECTS|INFLUENCES|CAUSES]->(d:Entity {id: 'Defect:porosity'})
RETURN x, r, d;

// ---- 4. Why does the graph believe "energy density influences porosity"? Evidence sentences (table)
MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.subject_id = 'Parameter:volumetric energy density' AND c.object_id = 'Defect:porosity'
RETURN p.title AS paper, c.page AS page, c.section AS section, c.relation AS relation,
       c.evidence AS evidence_sentence, round(c.confidence * 100) / 100 AS confidence, c.status AS status, c.method AS method
ORDER BY confidence DESC;

// ---- 5. The same claims as a graph: paper -> claim -> subject / object (graph)
MATCH path = (p:Paper)-[:SUPPORTS]->(c:Claim)-[:SUBJECT|OBJECT]->(e)
WHERE c.subject_id = 'Parameter:volumetric energy density' AND c.object_id = 'Defect:porosity'
RETURN path;

// ---- 6. Which alloys have evidence that energy density influences porosity? (table)
MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.subject_id = 'Parameter:volumetric energy density' AND c.object_id = 'Defect:porosity'
MATCH (p)-[s:STUDIES]->(m:Material) WHERE s.status = 'accepted'
RETURN m.canonical_name AS alloy, count(DISTINCT p) AS papers, collect(DISTINCT p.title)[..3] AS titles
ORDER BY papers DESC;

// ---- 7. Laser power values reported in the papers, unit-normalised (table)
MATCH (:Parameter {id: 'Parameter:laser power'})-[:HAS_VALUE]->(v:ParameterValue)
MATCH (p:Paper {paper_id: v.paper_id})
RETURN p.title AS paper, v.raw AS as_written, v.normalized_value AS W, v.normalized_min AS min_W, v.normalized_max AS max_W
ORDER BY coalesce(v.normalized_value, v.normalized_min);

// ---- 8. Tensile strength values reported for Ti-6Al-4V, with the sentence (table)
MATCH (p:Paper)-[s:STUDIES]->(:Material {id: 'Material:Ti-6Al-4V'}) WHERE s.status = 'accepted'
MATCH (p)-[:SUPPORTS]->(c:Claim {relation: 'HAS_VALUE', subject_id: 'Property:tensile strength'})
MATCH (v:Value {id: c.object_id})
RETURN p.title AS paper, v.raw AS value, v.normalized_value AS MPa, c.page AS page, c.evidence AS evidence
ORDER BY MPa DESC;

// ---- 9. Everything around one alloy within two hops (graph) - change the alloy id to explore
MATCH (p:Paper)-[s:STUDIES]->(m:Material {id: 'Material:Ti-6Al-4V'}) WHERE s.status = 'accepted'
MATCH (p)-[:MENTIONS]->(e:Entity)-[r:AFFECTS|INFLUENCES|CAUSES|IMPROVES|CHANGES]->(f:Entity)
WHERE r.status = 'accepted'
RETURN m, s, p, e, r, f LIMIT 150;

// ---- 10. Research gaps: effects reported by only one paper (weak evidence, worth new experiments) (table)
MATCH (a:Entity)-[r:AFFECTS|INFLUENCES|CAUSES|IMPROVES|CHANGES]->(b:Entity)
WHERE size(r.paper_ids) = 1 AND r.status = 'accepted'
RETURN a.canonical_name AS cause, type(r) AS relation, b.canonical_name AS effect,
       round(r.max_confidence * 100) / 100 AS confidence
ORDER BY confidence DESC LIMIT 25;

// ---- 11. Accepted, high-confidence core graph (graph)
MATCH (a:Entity)-[r]->(b:Entity)
WHERE type(r) IN ['PROCESSES','HAS_PARAMETER','AFFECTS','INFLUENCES','CAUSES','IMPROVES','CHANGES','CHARACTERIZES']
  AND r.status = 'accepted' AND r.max_confidence >= 0.85
RETURN a, r, b LIMIT 300;
