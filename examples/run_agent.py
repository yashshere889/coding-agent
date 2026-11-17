#!/usr/bin/env python3
"""Interactive coding agent on a vLLM (or any OpenAI-compatible) server.

    python examples/run_agent.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --workspace ~/src/myproject

Ctrl-C interrupts the current request; Ctrl-D or 'exit' quits.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codeagent import (  # noqa: E402
    Agent,
    ApprovalRequest,
    ApprovalResponse,
    Hooks,
    OpenAICompatibleModel,
    PermissionEngine,
    PreToolUseOutcome,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    default_tools,
)
from codeagent.permissions import MODES  # noqa: E402


def print_event(event: dict) -> None:
    kind = event["type"]
    if kind == "assistant_text":
        print(f"\n{event['text']}")
    elif kind == "tool_start":
        print(f"  -> {event['summary']}")
    elif kind == "tool_end" and event["is_error"]:
        print(f"     (error from {event['name']})")


async def terminal_approver(req: ApprovalRequest) -> ApprovalResponse:
    print(f"\n  ? {req.summary}\n    reason: {req.reason}")
    answer = await asyncio.to_thread(input, f"    allow? [y]es / [n]o / [a]lways allow '{req.suggested_rule}': ")
    answer = answer.strip().lower()
    if answer in ("a", "always"):
        return ApprovalResponse(True, remember_rule=req.suggested_rule)
    if answer in ("y", "yes"):
        return ApprovalResponse(True)
    feedback = await asyncio.to_thread(input, "    tell the agent why (optional): ")
    return ApprovalResponse(False, feedback=feedback.strip())


def build_hooks() -> Hooks:
    hooks = Hooks()

    @hooks.pre_tool_use("bash")
    def no_push(tool, inp, ctx):  # example policy: the agent never pushes
        if "git push" in inp["command"]:
            return PreToolUseOutcome(block="pushing is done by the user, not the agent")
        return None

    return hooks


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name as served by vLLM")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--workspace", default=".", type=Path)
    ap.add_argument("--mode", default="default", choices=MODES)
    ap.add_argument("--allow", action="append", default=[], help="allow rule, e.g. 'bash(npm test*)'; repeatable")
    ap.add_argument("--deny", action="append", default=[], help="deny rule, e.g. 'bash(rm *)'; repeatable")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    ctx = ToolContext(workspace=args.workspace)
    registry = ToolRegistry(default_tools())
    permissions = PermissionEngine(mode=args.mode, allow=args.allow, deny=args.deny, approver=terminal_approver)
    executor = ToolExecutor(registry, permissions, build_hooks(), on_event=print_event)
    model = OpenAICompatibleModel(
        args.model, args.base_url, args.api_key, max_tokens=args.max_tokens, temperature=args.temperature
    )
    agent = Agent(model, registry, executor, ctx, on_event=print_event)

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, ctx.cancel.set)
    print(f"workspace: {ctx.workspace}   mode: {args.mode}   (Ctrl-C interrupts, Ctrl-D quits)")
    while True:
        try:
            text = await asyncio.to_thread(input, "\n> ")
        except EOFError:
            break
        if text.strip() in ("exit", "quit"):
            break
        if not text.strip():
            continue
        ctx.cancel.clear()
        reply = await agent.send(text)
        if reply.startswith("["):  # status lines; normal replies were already printed as events
            print(reply)


if __name__ == "__main__":
    asyncio.run(main())
