"""Prompt templates live in prompts/*.txt so they can be versioned and edited
without touching code. Their SHA-1 hashes are recorded in each extraction's
provenance for reproducibility."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from ..config import get_settings


class PromptLibrary:
    def __init__(self, prompts_dir: Path | None = None) -> None:
        self.dir = Path(prompts_dir) if prompts_dir else get_settings().path("prompts_dir")

    @lru_cache(maxsize=32)
    def raw(self, name: str) -> str:
        return (self.dir / f"{name}.txt").read_text(encoding="utf-8")

    def render(self, name: str, **kwargs: str) -> str:
        return self.raw(name).format(**kwargs)

    def hash(self, name: str) -> str:
        return hashlib.sha1(self.raw(name).encode("utf-8")).hexdigest()[:12]

    def hashes(self) -> dict[str, str]:
        return {p.stem: self.hash(p.stem) for p in sorted(self.dir.glob("*.txt"))}
