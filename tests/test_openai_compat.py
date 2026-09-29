"""OpenAI-compatible provider (Groq / Cerebras / Gemini / LM Studio) against a fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm.client import LLMError
from src.llm.openai_compat import OpenAICompatClient, make_llm_client


class Handler(BaseHTTPRequestHandler):
    requests: list = []
    rate_limit_first = 0      # how many 429s to return before answering

    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        ok = self.headers.get("Authorization") == "Bearer test-key"
        self._send(200 if ok else 401, {"data": [{"id": "llama-test"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append({"payload": payload, "auth": self.headers.get("Authorization")})
        if Handler.rate_limit_first > 0:
            Handler.rate_limit_first -= 1
            self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
            return
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
                         "usage": {"prompt_tokens": 50, "completion_tokens": 7}})


@pytest.fixture()
def server():
    Handler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_chat_sends_json_mode_and_reads_usage(server):
    c = OpenAICompatClient(server, "llama-test", api_key="test-key", num_predict=300)
    assert c.is_available()
    assert c.chat("sys", "Return JSON") == '{"ok": true}'
    req = Handler.requests[-1]
    assert req["auth"] == "Bearer test-key"
    assert req["payload"]["response_format"] == {"type": "json_object"}
    assert req["payload"]["max_tokens"] == 300 and req["payload"]["temperature"] == 0.0
    assert c.last_usage == {"prompt_tokens": 50, "output_tokens": 7}


def test_rate_limit_is_retried(server):
    Handler.rate_limit_first = 2
    c = OpenAICompatClient(server, "llama-test", api_key="test-key")
    assert c.chat("s", "u") == '{"ok": true}'
    assert len(Handler.requests) == 3


def test_key_comes_from_environment_only(tmp_project, monkeypatch, server):
    tmp_project.llm.provider = "openai_compat"
    tmp_project.llm.api_base = server
    tmp_project.llm.model = "llama-test"
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        make_llm_client(tmp_project)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = make_llm_client(tmp_project)
    assert isinstance(client, OpenAICompatClient) and client.is_available()


def test_pipeline_runs_on_cloud_provider(tmp_project, monkeypatch, server):
    from src.ingestion.loader import parse_document
    from src.pipeline import ExtractionPipeline

    tmp_project.llm.provider = "openai_compat"
    tmp_project.llm.api_base = server
    tmp_project.llm.model = "llama-test"
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    pipe = ExtractionPipeline(tmp_project, mode="hybrid")
    doc = parse_document(tmp_project.path("demo_dir") / "demo_01_lpbf_ti64.md", tmp_project)
    res = pipe.extract(doc)
    assert res.stats.llm_calls > 0 and res.stats.llm_prompt_tokens > 0
    assert res.provenance.llm_model == "llama-test"
