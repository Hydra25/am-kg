"""Shared fixtures. Tests never need Ollama, Neo4j or GROBID: the LLM and the
Neo4j driver are replaced by deterministic fakes."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.normalization.entity_normalizer import EntityNormalizer  # noqa: E402
from src.normalization.lexicon import Lexicon  # noqa: E402


@pytest.fixture(scope="session")
def lexicon() -> Lexicon:
    return Lexicon.load(ROOT / "config" / "lexicon.yaml")


@pytest.fixture()
def normalizer(lexicon) -> EntityNormalizer:
    return EntityNormalizer(lexicon)


@pytest.fixture()
def tmp_project(tmp_path, monkeypatch):
    """A throw-away project root with config, prompts and demo text."""
    for d in ("config", "prompts"):
        shutil.copytree(ROOT / d, tmp_path / d)
    shutil.copytree(ROOT / "data" / "demo", tmp_path / "data" / "demo")
    shutil.copytree(ROOT / "data" / "annotations", tmp_path / "data" / "annotations")
    for var in ("NEO4J_URI", "EXTRACTION_MODE", "GRAPH_BACKEND", "VECTOR_BACKEND", "GROBID_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")  # guaranteed unreachable
    settings = load_settings(root=tmp_path)
    settings.graph.backend = "local"
    return settings


@pytest.fixture()
def demo_corpus(tmp_project):
    """Render the DEMO/SYNTHETIC PDFs, extract them (rule mode) and return settings."""
    from src.ingestion.demo_pdf import build_demo_pdfs
    from src.pipeline import ExtractionPipeline
    pdfs = build_demo_pdfs(tmp_project.path("demo_dir"), tmp_project.path("papers_dir"))
    pipe = ExtractionPipeline(tmp_project, mode="rule")
    for p in pdfs:
        pipe.process_file(p)
    return tmp_project
