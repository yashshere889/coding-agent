# codeagent: a tool system for coding agents

A small, model-agnostic tool layer and agent loop for building a coding agent on an
open-weight model served by vLLM. The design follows the patterns production coding agents use:
per-call safety flags, schema validation, a layered permission engine, parallel scheduling of
safe calls, hooks, output budgets, and an "every call gets a result" invariant.

```
codeagent/
  tool.py         Tool base class, ToolInput, ToolResult, ToolContext, CancelToken
  registry.py     tool lookup, the `tools` array for the model, deferred tools + tool_search
  permissions.py  deny rules -> tool check -> modes -> hooks -> allow rules -> ask a person
  hooks.py        pre_tool_use (block / rewrite / vote) and post_tool_use (inspect / replace)
  executor.py     scheduling (parallel vs serial) and the per-call pipeline
  llm.py          OpenAI-compatible adapter (vLLM); swap in any ChatModel
  loop.py         the agent loop
  tools/          read_file, edit_file, write_file, glob, grep, bash
examples/run_agent.py   interactive terminal agent
tests/                  45 tests; no GPU or model needed
```

## Run it

1. Serve a model with tool calling turned on. The parser depends on the model family; see vLLM's
   tool-calling docs. For example:

   ```bash
   vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --enable-auto-tool-choice --tool-call-parser qwen3_coder
   ```

2. Install and run:

   ```bash
   python -m venv .venv && .venv/bin/pip install -e ".[dev]"
   ```

   ```bash
   .venv/bin/python examples/run_agent.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --workspace ~/src/myproject --allow 'bash(pytest*)'
   ```

3. Run the tests: `.venv/bin/python -m pytest -q`

## How a tool call flows

```
model turn with tool_calls
  │  partition: consecutive concurrency-safe calls form a parallel batch, others run alone
  ▼  for each call:
  lookup (unknown name -> "did you mean …") → parse JSON (double-encoded tolerated)
  → pydantic validation (errors listed per field) → tool.validate (e.g. "read the file first")
  → pre_tool_use hooks (block / rewrite input / vote) → permission engine
  → tool.call with timeout + cancellation → post_tool_use hooks
  → output budget (big output saved to .agent/tool-results/, preview returned)
  ▼
one `tool` message per call, in the original order → next model request
```

Every failure becomes an error result the model can read and react to. Nothing is dropped,
because a missing tool result breaks the conversation.

## Adding a tool

```python
from pydantic import Field
from codeagent import Tool, ToolInput, ToolContext, ToolError

class RunTestsInput(ToolInput):
    path: str = Field("tests", description="Test file or directory")
    keyword: str | None = Field(None, description="Only run tests matching this -k expression")

class RunTests(Tool[RunTestsInput]):
    name = "run_tests"
    description = "Run the project's pytest suite and return the summary."
    Input = RunTestsInput
    timeout_s = 600

    def is_read_only(self, inp):          # auto-allowed in default mode, allowed in plan mode
        return True

    def is_concurrency_safe(self, inp):   # two test runs could clash; keep them serial
        return False

    async def call(self, inp: RunTestsInput, ctx: ToolContext) -> str:
        ...  # raise ToolError("message") to return an error to the model
```

Register it with `ToolRegistry([*default_tools(), RunTests()])`. Set `defer_loading = True` (and
a `search_hint`) for rarely used tools. They are then hidden behind `tool_search` until the model
loads them, which keeps the prompt short.

## Permissions

| Mode | Behaviour |
|---|---|
| `default` | read-only calls run; everything else needs an allow rule or a person |
| `accept_edits` | also auto-allows file edits inside the workspace |
| `plan` | read-only calls only; everything else is denied |
| `bypass` | everything runs except what deny rules block (use inside a sandbox only) |

Rules: `bash`, `bash(npm test*)`, `edit_file(src/*)`. Deny rules always win. A rule with a pattern
allows a bash call only if every simple command in it matches, so `bash(git *)` does not cover
`git status && rm -rf ~`. Commands using `$(…)` or backticks always go to a person. The bash
analyser is a heuristic, not a sandbox: for untrusted work, run the agent in a container.

## Tips for open-weight models

* Keep the always-loaded tool list short (6–10). Defer the rest behind `tool_search`.
* Flat schemas work best; `clean_schema` already strips titles and `Optional` unions.
* Error messages are written as instructions ("Read X first", "include more context"). Small
  models recover far better from those than from stack traces.
* Keep the system prompt and tool order fixed across turns so vLLM's prefix caching hits.
  Loading a deferred tool changes the tool block, which re-renders the prompt once.
* If tool calls show up as plain text, the vLLM `--tool-call-parser` doesn't match the model.

## Not built yet (natural next steps)

* **Context management:** summarise or drop old tool results when the conversation nears
  `--max-model-len`.
* **Streaming:** start safe tools while the model is still streaming the rest of its turn.
* **Sub-agents:** a tool whose `call` runs a nested `Agent` with a narrower tool set.
* **Persistent shell state:** `cd` and exported variables don't carry over between bash calls.
* **MCP:** adapt MCP server tools into `Tool` objects.
