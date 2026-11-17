"""A small, model-agnostic tool system for coding agents."""

from .executor import ToolCall, ToolExecutor, ToolOutcome
from .hooks import Hooks, PreToolUseOutcome
from .llm import AssistantTurn, ChatModel, OpenAICompatibleModel
from .loop import DEFAULT_SYSTEM_PROMPT, Agent
from .permissions import ApprovalRequest, ApprovalResponse, PermissionEngine, Rule
from .registry import ToolRegistry
from .tool import CancelToken, PermissionDecision, Tool, ToolContext, ToolError, ToolInput, ToolResult
from .tools import default_tools

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Agent",
    "ApprovalRequest",
    "ApprovalResponse",
    "AssistantTurn",
    "CancelToken",
    "ChatModel",
    "Hooks",
    "OpenAICompatibleModel",
    "PermissionDecision",
    "PermissionEngine",
    "PreToolUseOutcome",
    "Rule",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolExecutor",
    "ToolInput",
    "ToolOutcome",
    "ToolRegistry",
    "ToolResult",
    "default_tools",
]
