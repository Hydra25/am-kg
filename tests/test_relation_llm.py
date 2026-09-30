"""Separate (bigger) model for the relation prompt only: the 3B-vs-large relation experiment."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm.client import FakeLLM
from src.llm.openai_compat import OpenAICompatClient, reasoning_defaults


class Handler(BaseHTTPRequestHandler):
    payloads: list = []
    reject_field = None          # 400 once if this field is in the request

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "big-model"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.payloads.append(payload)
        if Handler.reject_field and Handler.reject_field in payload:
            self._send(400, {"error": {"message": f"property '{Handler.reject_field}' is unsupported"}})
            return
        self._send(200, {"choices": [{"message": {"content": '{"relations": []}'}}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 5}})


@pytest.fixture()
def server():
    Handler.payloads, Handler.reject_field = [], None
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_reasoning_defaults():
    assert reasoning_defaults("groq", "openai/gpt-oss-120b") == {"reasoning_effort": "low", "include_reasoning": False}
    assert reasoning_defaults("groq", "qwen/qwen3.8-27b")["reasoning_format"] == "hidden"
    assert reasoning_defaults("ollama", "qwen2.5:3b") == {}


def test_unsupported_reasoning_field_is_dropped(server):
    Handler.reject_field = "include_reasoning"
    c = OpenAICompatClient(server, "big-model", extra_body={"reasoning_effort": "low", "include_reasoning": False})
    assert c.chat("s", "u") == '{"relations": []}'
    assert "include_reasoning" in Handler.payloads[0] and "include_reasoning" not in Handler.payloads[-1]
    assert Handler.payloads[-1]["reasoning_effort"] == "low"


def test_pipeline_sends_only_relations_to_the_relation_model(tmp_project, server, monkeypatch):
    from src.ingestion.loader import parse_document
    from src.llm import openai_compat
    from src.pipeline import ExtractionPipeline
    from tests.test_speed import _llm_answer

    monkeypatch.setitem(openai_compat.PROVIDER_URLS, "lmstudio", server)   # lmstudio needs no key
    tmp_project.llm.cache = False
    tmp_project.extraction.relation_llm = "lmstudio:big-model"
    main = FakeLLM(_llm_answer)
    pipe = ExtractionPipeline(tmp_project, llm=main, mode="hybrid")
    doc = parse_document(tmp_project.path("demo_dir") / "demo_01_lpbf_ti64.md", tmp_project)
    res = pipe.extract(doc)
    assert Handler.payloads, "relation prompts must go to the relation model"
    assert all("relation" in p["messages"][1]["content"].lower() for p in Handler.payloads)
    assert res.provenance.relation_llm_model == "lmstudio:big-model"
    assert res.stats.llm_by_provider.get("lmstudio", 0) == len(Handler.payloads)
    assert res.stats.llm_by_stage["relations"]["calls"] == len(Handler.payloads)
