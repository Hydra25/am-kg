"""Central configuration: config/settings.yaml + environment variables (.env).

Nothing here is hard-coded to a machine: every path is resolved relative to the
project root, and every connection setting can be overridden from the
environment. No paid API keys are used anywhere.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_VERSION = "0.1.0"


class PathsConfig(BaseModel):
    papers_dir: str = "data/papers"
    processed_dir: str = "data/processed"
    extracted_dir: str = "data/extracted"
    graph_dir: str = "data/graph"
    review_dir: str = "data/review"
    annotations_dir: str = "data/annotations"
    index_dir: str = "data/index"
    demo_dir: str = "data/demo"
    lexicon_file: str = "config/lexicon.yaml"
    prompts_dir: str = "prompts"
    logs_dir: str = "logs"
    cache_dir: str = "data/cache"


class PdfConfig(BaseModel):
    parser: str = "auto"
    extract_tables: bool = True
    remove_headers_footers: bool = True


class ExtractionConfig(BaseModel):
    mode: str = "auto"
    chunk_chars: int = 2200
    skip_sections: list[str] = Field(default_factory=lambda: ["references", "acknowledgements"])
    llm_findings: bool = True
    llm_normalization: bool = True
    # Extra sections that are parsed and rule-extracted but NOT sent to the LLM.
    llm_skip_sections: list[str] = Field(default_factory=lambda: ["front_matter", "title"])
    # Rule-first gating: only send a chunk to the LLM if the rule-based extractor
    # found at least one AM entity or a number with a unit in it.
    llm_gate: bool = True
    # Who proposes entities beyond the lexicon (in hybrid/llm/gliner modes):
    #   auto   = GLiNER if the `gliner` package + model are available, else the LLM prompt
    #   gliner = GLiNER model (fast, local, no generation)
    #   llm    = LLM entity prompt (slow; the original behaviour)
    #   none   = lexicon only
    entity_backend: str = "auto"


class LLMConfig(BaseModel):
    # ollama (default, local, free) | groq | cerebras | gemini | openrouter | lmstudio | openai_compat
    provider: str = "ollama"
    api_base: str = ""               # only for provider openai_compat (others have a built-in URL)
    api_key_env: str = "LLM_API_KEY"  # NAME of the env variable holding the key (key itself lives in .env)
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.0
    num_ctx: int = 8192
    timeout_s: int = 240
    max_retries: int = 2
    structured_output: bool = True
    # For "thinking" models (qwen3.x, gemma4 ...): False disables the slow reasoning
    # trace. None = do not send the flag (needed for older models such as qwen2.5).
    think: Optional[bool] = None
    num_predict: int = 1500        # max output tokens per call
    # Tighter caps per extraction stage (stops runaway generations early).
    num_predict_by_stage: dict[str, int] = Field(default_factory=lambda: {
        "entities": 1000, "relations": 800, "findings": 400, "normalization": 100})
    keep_alive: str = "30m"        # how long Ollama keeps the model loaded after a call
    cache: bool = True             # reuse answers for identical prompts (data/cache/llm)


class GlinerConfig(BaseModel):
    model: str = "urchade/gliner_medium-v2.1"   # Apache-2.0, ~0.5 GB, downloaded once
    threshold: float = 0.5
    device: str = "cpu"            # "mps" uses the Apple GPU (try it if cpu is slow)
    batch_size: int = 8
    labels: dict[str, str] = Field(default_factory=dict)   # empty = built-in AM labels


class QualityConfig(BaseModel):
    accept_threshold: float = 0.85
    review_threshold: float = 0.60
    require_evidence: bool = True
    require_evidence_in_source: bool = True


class Neo4jConfig(BaseModel):
    uri: str = ""
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @property
    def enabled(self) -> bool:
        return bool(self.uri)


class GraphConfig(BaseModel):
    backend: str = "auto"
    insert_review_items: bool = True


class RetrievalConfig(BaseModel):
    vector_backend: str = "bm25"
    top_k_passages: int = 6
    top_k_claims: int = 20
    passage_window: int = 2


class GapsConfig(BaseModel):
    min_pair_papers: int = 2
    max_context_papers: int = 1


class Settings(BaseModel):
    root: Path = PROJECT_ROOT
    paths: PathsConfig = Field(default_factory=PathsConfig)
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gliner: GlinerConfig = Field(default_factory=GlinerConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    gaps: GapsConfig = Field(default_factory=GapsConfig)
    grobid_url: str = ""
    log_level: str = "INFO"

    def path(self, key: str) -> Path:
        """Absolute path for a key of `paths` (directories are created on demand)."""
        rel = getattr(self.paths, key)
        p = Path(rel)
        p = p if p.is_absolute() else self.root / p
        if key.endswith("_dir"):
            p.mkdir(parents=True, exist_ok=True)
        return p


def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    return val if val not in (None, "") else None


def load_settings(config_file: Optional[Path] = None, root: Optional[Path] = None) -> Settings:
    root = Path(root) if root else PROJECT_ROOT
    if load_dotenv is not None:
        load_dotenv(root / ".env", override=False)
    config_file = Path(config_file) if config_file else root / "config" / "settings.yaml"
    raw: dict[str, Any] = {}
    if config_file.exists():
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    llm_raw = dict(raw.get("llm", {}))
    settings = Settings(
        root=root,
        paths=PathsConfig(**raw.get("paths", {})),
        pdf=PdfConfig(**raw.get("pdf", {})),
        extraction=ExtractionConfig(**raw.get("extraction", {})),
        llm=LLMConfig(**llm_raw),
        gliner=GlinerConfig(**raw.get("gliner", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        graph=GraphConfig(**raw.get("graph", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        gaps=GapsConfig(**raw.get("gaps", {})),
    )
    # ---- environment overrides -------------------------------------------
    overrides: list[tuple[str, Any, str, type]] = [
        ("OLLAMA_BASE_URL", settings.llm, "base_url", str),
        ("OLLAMA_MODEL", settings.llm, "model", str),
        ("MODEL_NAME", settings.llm, "model", str),  # alias used in the project brief
        ("OLLAMA_EMBED_MODEL", settings.llm, "embed_model", str),
        ("LLM_PROVIDER", settings.llm, "provider", str),
        ("LLM_API_BASE", settings.llm, "api_base", str),
        ("LLM_MODEL", settings.llm, "model", str),
        ("NEO4J_URI", settings.neo4j, "uri", str),
        ("NEO4J_USERNAME", settings.neo4j, "username", str),
        ("NEO4J_PASSWORD", settings.neo4j, "password", str),
        ("NEO4J_DATABASE", settings.neo4j, "database", str),
        ("EXTRACTION_MODE", settings.extraction, "mode", str),
        ("ENTITY_BACKEND", settings.extraction, "entity_backend", str),
        ("GRAPH_BACKEND", settings.graph, "backend", str),
        ("VECTOR_BACKEND", settings.retrieval, "vector_backend", str),
        ("AMKG_ACCEPT_THRESHOLD", settings.quality, "accept_threshold", float),
        ("AMKG_REVIEW_THRESHOLD", settings.quality, "review_threshold", float),
        ("PDF_PARSER", settings.pdf, "parser", str),
    ]
    for env_name, obj, attr, cast in overrides:
        val = _env(env_name)
        if val is not None:
            setattr(obj, attr, cast(val))
    cache = _env("LLM_CACHE")
    if cache is not None:
        settings.llm.cache = cache.strip().lower() in ("1", "true", "yes", "on")
    think = _env("OLLAMA_THINK")
    if think is not None:
        settings.llm.think = think.strip().lower() in ("1", "true", "yes", "on")
    settings.grobid_url = _env("GROBID_URL") or ""
    settings.log_level = _env("LOG_LEVEL") or "INFO"
    if settings.quality.review_threshold > settings.quality.accept_threshold:
        raise ValueError("quality.review_threshold must be <= quality.accept_threshold")
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def setup_logging(level: str | None = None, log_file: Path | None = None) -> None:
    lvl = getattr(logging, (level or get_settings().log_level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
