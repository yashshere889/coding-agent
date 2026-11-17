"""Drive OpenAICompatibleModel against a local stand-in for vLLM's /v1/chat/completions."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from conftest import run

from codeagent import Agent, OpenAICompatibleModel, PermissionEngine, ToolExecutor, ToolRegistry, default_tools


def completion(message: dict, finish: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def start_server(responses: list[dict], seen: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            payload = json.dumps(responses.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_round_trip_through_openai_client(ctx):
    responses = [
        completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "chatcmpl-tool-a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "README.md"}'}},
                    {"id": "chatcmpl-tool-b", "type": "function", "function": {"name": "glob", "arguments": '{"pattern": "**/*.py"}'}},
                ],
            },
            "tool_calls",
        ),
        completion({"role": "assistant", "content": "The project has one Python file."}, "stop"),
    ]
    seen: list[dict] = []
    server = start_server(responses, seen)
    try:
        model = OpenAICompatibleModel("fake", base_url=f"http://127.0.0.1:{server.server_port}/v1", temperature=0.2)
        registry = ToolRegistry(default_tools())
        agent = Agent(model, registry, ToolExecutor(registry, PermissionEngine()), ctx)
        answer = run(agent.send("what is here?"))
    finally:
        server.shutdown()

    assert answer == "The project has one Python file."
    first, second = seen
    assert first["tool_choice"] == "auto" and first["temperature"] == 0.2
    assert [t["function"]["name"] for t in first["tools"]][:2] == ["read_file", "edit_file"]
    # Second request: assistant message with tool_calls, then one tool message per call.
    assistant, tool_a, tool_b = second["messages"][-3:]
    assert assistant["role"] == "assistant" and len(assistant["tool_calls"]) == 2
    assert (tool_a["tool_call_id"], tool_b["tool_call_id"]) == ("chatcmpl-tool-a", "chatcmpl-tool-b")
    assert "# demo" in tool_a["content"] and "src/app.py" in tool_b["content"]
