"""Type aliases and constants for langchain-claude-agent."""

from __future__ import annotations

# SDK model names
DEFAULT_MODEL = "sonnet"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
MCP_SERVER_NAME = "langchain-tools"
MCP_SERVER_VERSION = "1.0.0"

# Tool name prefix for MCP tools
TOOL_NAME_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
