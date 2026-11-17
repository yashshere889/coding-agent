"""Model adapters. The agent only needs ``ChatModel.complete(messages, tools) -> AssistantTurn``.

``OpenAICompatibleModel`` talks to any OpenAI-compatible ``/v1/chat/completions`` server, which
is what ``vllm serve`` exposes. Start vLLM with tool calling turned on, for example:

    vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --enable-auto-tool-choice --tool-call-parser qwen3_coder

The parser name depends on the model family (hermes, llama3_json, mistral, qwen3_coder, ...);
check the vLLM tool-calling docs for yours. Without a working parser the model's tool calls
arrive as plain text and nothing runs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .executor import ToolCall


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    message: dict[str, Any]  # the assistant message to append to the history, as-is
    usage: dict[str, Any] = field(default_factory=dict)


class ChatModel(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn: ...


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        *,
        max_tokens: int = 8192,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 600,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.parallel_tool_calls = parallel_tool_calls
        self.extra_body = extra_body or {}

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        calls = []
        for tc in msg.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            args = fn.arguments if isinstance(fn.arguments, str) else json.dumps(fn.arguments or {})
            calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:16]}", name=fn.name, arguments=args))

        message: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls
            ]
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return AssistantTurn(msg.content or "", calls, choice.finish_reason, message, usage)
