"""Convert LangChain BaseTool objects to SDK-compatible tool specifications.

This module provides utilities for extracting schemas from LangChain tools
and wrapping them in a plain dataclass (SDKToolSpec) that holds the name,
description, schema, and async handler needed by the Claude Agent SDK.

No SDK imports are used here -- the actual ``@tool`` / ``create_sdk_mcp_server``
wrapping happens downstream in ``chat_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.tools import BaseTool

# JSON-schema type string -> Python type
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


@dataclass
class SDKToolSpec:
    """Container for a single tool converted to SDK-compatible format.

    Attributes:
        name: Tool name as it will appear in the SDK.
        description: Human-readable description of what the tool does.
        schema: Mapping of parameter names to Python types
            (e.g. ``{"a": int, "b": int}``).
        handler: Async callable that accepts ``dict[str, Any]`` args and
            returns ``{"content": [{"type": "text", "text": "..."}]}``.
    """

    name: str
    description: str
    schema: dict[str, type]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _get_json_schema(tool: BaseTool) -> dict[str, Any]:
    """Return a JSON-schema mapping for a LangChain tool input.

    Args:
        tool: A LangChain ``BaseTool`` instance.

    Returns:
        A JSON-schema dictionary describing the tool input.
    """
    args_schema = tool.args_schema
    if args_schema is None:
        return {}

    if isinstance(args_schema, dict):
        return args_schema

    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()

    input_schema = tool.get_input_schema()
    if hasattr(input_schema, "model_json_schema"):
        return input_schema.model_json_schema()

    return {}


def get_tool_schema(lc_tool: BaseTool) -> dict[str, type]:
    """Extract a simple type-mapping schema from a LangChain tool.

    Uses the tool's JSON schema when available and maps JSON-schema type
    strings to Python types.

    Args:
        lc_tool: A LangChain BaseTool instance.

    Returns:
        A dict mapping parameter names to Python types, e.g.
        ``{"a": int, "b": int}``.
    """
    json_schema = _get_json_schema(lc_tool)
    properties: dict[str, Any] = json_schema.get("properties", {})

    result: dict[str, type] = {}
    for param_name, param_info in properties.items():
        json_type = param_info.get("type", "string")
        result[param_name] = _JSON_TYPE_MAP.get(json_type, str)

    return result


def convert_langchain_tool_to_sdk(lc_tool: BaseTool) -> SDKToolSpec:
    """Convert a single LangChain BaseTool into an SDKToolSpec.

    The returned spec contains an async handler that delegates to the
    tool's ``ainvoke`` (preferred) with a fallback to ``invoke``.

    Args:
        lc_tool: A LangChain BaseTool instance.

    Returns:
        An SDKToolSpec dataclass with name, description, schema, and
        async handler ready for the Claude Agent SDK.
    """
    schema = get_tool_schema(lc_tool)

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        """Invoke the LangChain tool and wrap the result for the SDK.

        Args:
            args: Dictionary of arguments to pass to the tool.

        Returns:
            SDK-formatted result dict with content list.
        """
        result = await lc_tool.ainvoke(args)
        return {"content": [{"type": "text", "text": str(result)}]}

    return SDKToolSpec(
        name=lc_tool.name,
        description=lc_tool.description,
        schema=schema,
        handler=_handler,
    )


def convert_langchain_tools(tools: list[BaseTool]) -> list[SDKToolSpec]:
    """Batch-convert a list of LangChain tools to SDK tool specs.

    Args:
        tools: A list of LangChain BaseTool instances.

    Returns:
        A list of SDKToolSpec instances, one per input tool.
    """
    return [convert_langchain_tool_to_sdk(t) for t in tools]
