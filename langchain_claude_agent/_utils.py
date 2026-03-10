"""Message conversion and usage mapping utilities for langchain-claude-agent."""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def extract_system_message(
    messages: list[BaseMessage],
) -> tuple[str | None, list[BaseMessage]]:
    """Extract and concatenate system messages from a message list.

    Iterates through the messages, collects all SystemMessage contents, and
    returns them joined by newlines. Non-system messages are returned as the
    remaining list.

    Args:
        messages: A list of LangChain BaseMessage instances.

    Returns:
        A tuple of (system_prompt, remaining_messages) where system_prompt
        is the concatenated system text or None if no system messages exist.
    """
    system_parts: list[str] = []
    remaining: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(str(msg.content))
        else:
            remaining.append(msg)

    system = "\n".join(system_parts) if system_parts else None
    return system, remaining


def convert_messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Convert a list of LangChain messages into a single prompt string.

    Each message type is formatted with a role prefix. SystemMessages are
    skipped (they should be extracted separately via extract_system_message).

    Args:
        messages: A list of LangChain BaseMessage instances.

    Returns:
        A newline-joined string with role-prefixed message contents.
    """
    lines: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            if name:
                lines.append(f"Tool Result ({name}): {msg.content}")
            else:
                lines.append(f"Tool Result: {msg.content}")
        else:
            lines.append(f"{msg.type}: {msg.content}")

    return "\n".join(lines)


def has_multimodal_content(messages: list[BaseMessage]) -> bool:
    """Check if any message contains multimodal content (images).

    Args:
        messages: A list of LangChain BaseMessage instances.

    Returns:
        True if any message has list-based content with image blocks.
    """
    for msg in messages:
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") in (
                    "image_url",
                    "image",
                ):
                    return True
    return False


def _convert_image_block(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain image_url block to SDK image format.

    Handles both data URIs (``data:image/jpeg;base64,...``) and plain
    base64 strings.

    Args:
        block: A LangChain image_url content block.

    Returns:
        An SDK-formatted image block with ``type``, ``source`` keys.
    """
    url = block.get("image_url", {}).get("url", "")

    # Parse data URI: data:image/jpeg;base64,<data>
    match = re.match(r"data:(image/\w+);base64,(.+)", url)
    if match:
        media_type = match.group(1)
        data = match.group(2)
    else:
        # Assume raw base64 JPEG if no data URI prefix
        media_type = "image/jpeg"
        data = url

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def _convert_message_content(content: Any) -> Any:
    """Convert LangChain message content to SDK streaming message format.

    Handles both plain strings and multimodal content lists.

    Args:
        content: Message content — either a string or a list of content blocks.

    Returns:
        A string or a list of SDK-formatted content blocks.
    """
    if isinstance(content, str):
        return content

    sdk_blocks = []
    for block in content:
        if not isinstance(block, dict):
            sdk_blocks.append({"type": "text", "text": str(block)})
        elif block.get("type") == "text":
            sdk_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image_url":
            sdk_blocks.append(_convert_image_block(block))
        elif block.get("type") == "image":
            # Already in SDK format
            sdk_blocks.append(block)
        else:
            sdk_blocks.append({"type": "text", "text": str(block)})
    return sdk_blocks


def convert_messages_to_sdk_streaming(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """Convert LangChain messages to SDK streaming input message dicts.

    Each message is converted to the SDK's streaming format:
    ``{"type": "user", "message": {"role": "user", "content": ...}}``.
    Supports multimodal content (images).

    Args:
        messages: A list of LangChain BaseMessage instances (system messages
            should already be extracted).

    Returns:
        A list of SDK streaming message dicts.
    """
    sdk_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            sdk_messages.append(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": _convert_message_content(msg.content),
                    },
                }
            )
        elif isinstance(msg, AIMessage):
            sdk_messages.append(
                {
                    "type": "user",
                    "message": {
                        "role": "assistant",
                        "content": _convert_message_content(msg.content),
                    },
                }
            )
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            text = (
                f"Tool Result ({name}): {msg.content}"
                if name
                else f"Tool Result: {msg.content}"
            )
            sdk_messages.append(
                {
                    "type": "user",
                    "message": {"role": "user", "content": text},
                }
            )
        else:
            sdk_messages.append(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": _convert_message_content(msg.content),
                    },
                }
            )
    return sdk_messages


def map_sdk_usage(sdk_usage: dict | None) -> dict:
    """Map Claude Agent SDK usage data to LangChain usage metadata format.

    Translates the SDK's usage dictionary into the format expected by
    LangChain's AIMessage usage_metadata, including optional cache details.

    Args:
        sdk_usage: A dictionary of usage data from the SDK response, or None.

    Returns:
        A dictionary with input_tokens, output_tokens, total_tokens, and
        optionally input_token_details for cache metrics. Returns an empty
        dict if sdk_usage is None or empty.
    """
    if not sdk_usage:
        return {}

    input_tokens = sdk_usage.get("input_tokens", 0)
    output_tokens = sdk_usage.get("output_tokens", 0)

    result: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    cache_read = sdk_usage.get("cache_read_input_tokens")
    cache_creation = sdk_usage.get("cache_creation_input_tokens")

    if cache_read is not None or cache_creation is not None:
        details: dict = {}
        if cache_read is not None:
            details["cache_read"] = cache_read
        if cache_creation is not None:
            details["cache_creation"] = cache_creation
        result["input_token_details"] = details

    return result


def check_claude_agent_sdk_credentials() -> tuple[bool, str]:
    """Check if Claude Agent SDK credentials are available.

    Checks in order:
        1. ANTHROPIC_API_KEY environment variable
        2. Cloud provider env vars (Bedrock, Vertex, Foundry)
        3. SDK probe via query() with max_turns=0

    Returns:
        Tuple of (success, message) where success is True if credentials
        are found and message describes the credential source.
    """
    # Check API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "ANTHROPIC_API_KEY found in environment"

    # Check cloud provider credentials
    for env_var, provider in [
        ("CLAUDE_CODE_USE_BEDROCK", "Amazon Bedrock"),
        ("CLAUDE_CODE_USE_VERTEX", "Google Vertex AI"),
        ("CLAUDE_CODE_USE_FOUNDRY", "Microsoft Azure Foundry"),
    ]:
        if os.environ.get(env_var):
            return True, f"{provider} credentials configured via {env_var}"

    # Probe SDK directly
    try:
        import asyncio

        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import query as sdk_query

        async def _probe():
            options = ClaudeAgentOptions(model="sonnet", max_turns=0)
            try:
                async for _ in sdk_query(prompt="test", options=options):
                    pass
            except Exception as e:
                err_str = str(e).lower()
                if "max_turns" in err_str or "turn" in err_str:
                    return True, "claude_agent_sdk: auth verified (max_turns reached)"
                if any(k in err_str for k in ("login", "auth", "credential")):
                    return False, f"claude_agent_sdk: auth failed: {e}"
                return True, f"claude_agent_sdk: probe completed with: {e}"
            return True, "claude_agent_sdk: probe completed successfully"

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_probe())
        finally:
            loop.close()
    except ImportError:
        return False, "claude_agent_sdk: not installed"
    except Exception as e:
        return False, f"claude_agent_sdk: probe failed: {e}"
