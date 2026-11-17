from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codeagent import (  # noqa: E402
    AssistantTurn,
    PermissionEngine,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    default_tools,
)


def run(coro):
    return asyncio.run(coro)


def call(name: str, i: int = 0, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}_{i}", name=name, arguments=json.dumps(args))


class ScriptedModel:
    """A fake ChatModel that plays back a list of turns and records what it was sent."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def complete(self, messages, tools):
        self.requests.append(([dict(m) for m in messages], tools))
        return self.turns.pop(0)


def turn(text: str = "", calls: list[ToolCall] | None = None) -> AssistantTurn:
    calls = calls or []
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
        ]
    return AssistantTurn(text, calls, "tool_calls" if calls else "stop", message)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "README.md").write_text("# demo\n")
    return tmp_path


@pytest.fixture
def make_executor():
    def factory(tools=None, **perm_kwargs):
        registry = ToolRegistry(tools if tools is not None else default_tools())
        return ToolExecutor(registry, PermissionEngine(**perm_kwargs))

    return factory


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)
