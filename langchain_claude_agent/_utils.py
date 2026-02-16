"""Message conversion and usage mapping utilities for langchain-claude-agent."""

from __future__ import annotations

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
