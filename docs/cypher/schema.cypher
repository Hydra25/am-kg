// Constraints and indexes created by `python main.py build-graph`
CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT value_id IF NOT EXISTS FOR (v:Value) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE;
CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (f:Finding) REQUIRE f.finding_id IS UNIQUE;
CREATE INDEX claim_subject IF NOT EXISTS FOR (c:Claim) ON (c.subject_id);
CREATE INDEX claim_object IF NOT EXISTS FOR (c:Claim) ON (c.object_id);
CREATE INDEX claim_status IF NOT EXISTS FOR (c:Claim) ON (c.status);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name);
CREATE INDEX value_paper IF NOT EXISTS FOR (v:Value) ON (v.paper_id);
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.canonical_name, e.alias_text];
