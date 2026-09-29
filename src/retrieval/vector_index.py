"""Passage retrieval over the parsed papers.

Backends
  bm25   - pure-Python Okapi BM25 (default, no extra dependency, works offline)
  chroma - Chroma vector DB with embeddings from Ollama (e.g. nomic-embed-text)
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..ingestion.loader import load_parsed
from ..normalization.text import normalize_chars
from ..ontology.models import ParsedDocument

log = logging.getLogger(__name__)
_STOP = set("""a an the of in on for to and or with by from at as is are was were be been this that these those
it its into than then which what how does do did can could may might we our their there here using used via
""".split())


@dataclass
class Passage:
    passage_id: str
    paper_id: str
    title: Optional[str]
    page: Optional[int]
    section: str
    section_type: str
    text: str
    sentence_ids: list[str]
    is_synthetic: bool = False


def tokenize(text: str) -> list[str]:
    t = normalize_chars(text).lower()
    return [w for w in re.findall(r"[a-z0-9α-ω']+(?:-[a-z0-9]+)*", t) if w not in _STOP and len(w) > 1]


def build_passages(docs: Iterable[ParsedDocument], window: int = 2,
                   skip: Iterable[str] = ("references", "acknowledgements")) -> list[Passage]:
    skip = set(skip)
    out: list[Passage] = []
    for doc in docs:
        by_sec: dict[str, list] = defaultdict(list)
        for s in doc.sentences:
            if s.section_type not in skip:
                by_sec[s.section_id].append(s)
        for sents in by_sec.values():
            step = max(1, window)
            for i in range(0, len(sents), step):
                chunk = sents[i:i + window]
                out.append(Passage(
                    passage_id=f"{chunk[0].sentence_id}", paper_id=doc.paper.paper_id, title=doc.paper.title,
                    page=chunk[0].page, section=chunk[0].section_title, section_type=chunk[0].section_type,
                    text=" ".join(s.text for s in chunk), sentence_ids=[s.sentence_id for s in chunk],
                    is_synthetic=doc.paper.is_synthetic))
    return out


def load_docs(processed_dir: Path) -> list[ParsedDocument]:
    return [load_parsed(p) for p in sorted(Path(processed_dir).glob("paper_*.json"))]


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.text + " " + p.section)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 6, boost: Optional[Callable[[Passage], float]] = None) -> list[tuple[Passage, float]]:
        q = tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf.get(t, 0) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1)))
            if s > 0:
                if boost:
                    s *= boost(self.passages[i])
                scores.append((self.passages[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class ChromaIndex:
    """Optional dense retrieval. Requires `pip install chromadb` and an embedding
    function (default: Ollama /api/embed with OLLAMA_EMBED_MODEL)."""

    def __init__(self, passages: list[Passage], persist_dir: Path, embed: Callable[[list[str]], list[list[float]]],
                 collection: str = "am_passages") -> None:
        import chromadb
        self.passages = {p.passage_id: p for p in passages}
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.embed = embed
        existing = set(self.col.get(include=[])["ids"]) if self.col.count() else set()
        new = [p for p in passages if p.passage_id not in existing]
        for i in range(0, len(new), 64):
            batch = new[i:i + 64]
            self.col.add(ids=[p.passage_id for p in batch], documents=[p.text for p in batch],
                         embeddings=self.embed([p.text for p in batch]),
                         metadatas=[{"paper_id": p.paper_id, "section": p.section, "page": p.page or 0} for p in batch])

    def search(self, query: str, k: int = 6, boost: Optional[Callable[[Passage], float]] = None) -> list[tuple[Passage, float]]:
        res = self.col.query(query_embeddings=self.embed([query]), n_results=min(k * 3, max(1, self.col.count())))
        out = []
        for pid, dist in zip(res["ids"][0], res["distances"][0]):
            p = self.passages.get(pid)
            if p is None:
                continue
            score = 1.0 - float(dist)
            out.append((p, score * (boost(p) if boost else 1.0)))
        out.sort(key=lambda x: -x[1])
        return out[:k]
