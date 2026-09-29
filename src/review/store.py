"""Human-in-the-loop review decisions (append-only JSONL audit log).

Actions
  accept  - mark a claim/finding/entity as verified
  reject  - remove a claim/finding from the graph (kept in the audit log)
  edit    - change fields of a claim (relation type, endpoints, value) or an entity name
  merge   - merge one canonical entity into another (stored as a normalisation
            override, so future extractions map it automatically)

Decisions never modify the extracted JSON files; they are applied on load by
the KnowledgeBase, so every correction is reversible and auditable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..ontology.models import stable_id, utcnow

ACTIONS = {"accept", "reject", "edit", "merge"}
KINDS = {"claim", "entity", "finding"}


@dataclass
class ReviewDecision:
    target_kind: str
    target_id: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    reviewer: str = "reviewer"
    note: Optional[str] = None
    timestamp: str = field(default_factory=utcnow)
    decision_id: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if self.target_kind not in KINDS:
            raise ValueError(f"target_kind must be one of {KINDS}")
        if not self.decision_id:
            self.decision_id = stable_id("d", self.target_kind, self.target_id, self.action, self.timestamp)


class ReviewStore:
    def __init__(self, review_dir: Path) -> None:
        self.dir = Path(review_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "decisions.jsonl"

    def add(self, decision: ReviewDecision) -> ReviewDecision:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
        return decision

    def all(self) -> list[ReviewDecision]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(ReviewDecision(**json.loads(line)))
        return out

    def latest(self) -> dict[tuple[str, str], ReviewDecision]:
        """Most recent non-merge decision per target (later decisions override earlier ones)."""
        out: dict[tuple[str, str], ReviewDecision] = {}
        for d in self.all():
            if d.action != "merge":
                out[(d.target_kind, d.target_id)] = d
        return out

    def merges(self) -> dict[str, str]:
        return {d.target_id: d.payload["into"] for d in self.all()
                if d.action == "merge" and d.payload.get("into")}
