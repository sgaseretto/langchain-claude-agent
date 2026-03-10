"""Type aliases and constants for langchain-claude-agent."""

from __future__ import annotations

from typing import Any, Literal

# SDK model names
DEFAULT_MODEL = "sonnet"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
MCP_SERVER_NAME = "langchain-tools"
MCP_SERVER_VERSION = "1.0.0"

# Tool name prefix for MCP tools
TOOL_NAME_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Thinking configuration type (dict matching ThinkingConfigAdaptive/Enabled/Disabled)
ThinkingConfig = dict[str, Any]

# Effort level type
EffortLevel = Literal["low", "medium", "high", "max"]

# Output format for structured output
OutputFormat = dict[str, Any]
