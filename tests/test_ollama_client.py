"""Exercises the real HTTP client + hybrid pipeline against a tiny fake Ollama server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm.client import OllamaClient


def _answer(user: str) -> str:
    if "Extract Additive Manufacturing entities" in user:
        return json.dumps({"entities": [{"text": "laser power", "type": "Parameter", "sentence": "S1", "confidence": 0.9}]})
    if "Extract relations" in user:
        return '```json\n{"relations": [],}\n```'          # malformed on purpose -> repaired
    if "Identify which numbered sentences" in user:
        return json.dumps({"findings": []})
    return "ok"


class Handler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5:3b"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append(payload)
        user = payload["messages"][-1]["content"]
        self._send({"message": {"role": "assistant", "content": _answer(user)}})


@pytest.fixture()
def fake_ollama():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_client_payload(fake_ollama):
    c = OllamaClient(fake_ollama, "qwen2.5:3b", num_ctx=8192)
    assert c.is_available()
    assert OllamaClient(fake_ollama, "llama3:8b").is_available() is False
    out = c.chat("sys", "hello", schema={"type": "object"})
    req = Handler.requests[-1]
    assert out == "ok" and req["model"] == "qwen2.5:3b" and req["stream"] is False
    assert req["options"]["num_ctx"] == 8192 and req["options"]["temperature"] == 0.0
    assert req["format"] == {"type": "object"}


def test_hybrid_pipeline_with_fake_ollama(tmp_project, fake_ollama):
    from src.pipeline import ExtractionPipeline
    tmp_project.llm.base_url = fake_ollama
    pipe = ExtractionPipeline(tmp_project, mode="auto")
    assert pipe.mode == "hybrid"
    res = pipe.process_file(tmp_project.path("demo_dir") / "demo_01_lpbf_ti64.md")
    assert res.stats.llm_calls > 0 and res.provenance.llm_model == "qwen2.5:3b"
    assert res.provenance.prompt_hashes
    assert any(m.method == "rule+llm" for m in res.entities)
    assert res.stats.accepted > 20
