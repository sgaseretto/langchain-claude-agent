"""Tests for message conversion and usage mapping utilities."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_claude_agent._utils import (
    convert_messages_to_prompt,
    extract_system_message,
    map_sdk_usage,
)


class TestExtractSystemMessage:
    """Tests for extract_system_message."""

    def test_no_system_message(self):
        """When no system messages are present, return None and all messages."""
        msgs = [HumanMessage(content="hello")]
        system, remaining = extract_system_message(msgs)
        assert system is None
        assert len(remaining) == 1
        assert remaining[0].content == "hello"

    def test_single_system_message(self):
        """A single system message should be extracted."""
        msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="hi"),
        ]
        system, remaining = extract_system_message(msgs)
        assert system == "You are helpful."
        assert len(remaining) == 1
        assert isinstance(remaining[0], HumanMessage)

    def test_multiple_system_messages_concatenated(self):
        """Multiple system messages should be concatenated with newline."""
        msgs = [
            SystemMessage(content="Be helpful."),
            SystemMessage(content="Be concise."),
            HumanMessage(content="hi"),
        ]
        system, remaining = extract_system_message(msgs)
        assert system == "Be helpful.\nBe concise."
        assert len(remaining) == 1

    def test_system_message_not_at_start(self):
        """System messages anywhere in the list should be extracted."""
        msgs = [
            HumanMessage(content="hello"),
            SystemMessage(content="You are helpful."),
            AIMessage(content="hi there"),
        ]
        system, remaining = extract_system_message(msgs)
        assert system == "You are helpful."
        assert len(remaining) == 2
        assert isinstance(remaining[0], HumanMessage)
        assert isinstance(remaining[1], AIMessage)


class TestConvertMessages:
    """Tests for convert_messages_to_prompt."""

    def test_single_human_message(self):
        """A single human message should be converted."""
        msgs = [HumanMessage(content="hello")]
        result = convert_messages_to_prompt(msgs)
        assert result == "Human: hello"

    def test_human_ai_conversation(self):
        """A human-ai conversation should alternate correctly."""
        msgs = [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
        result = convert_messages_to_prompt(msgs)
        assert result == "Human: hello\nAssistant: hi there"

    def test_tool_message(self):
        """A tool message with a name should include the name."""
        msgs = [
            ToolMessage(content="result data", name="search", tool_call_id="tc_1"),
        ]
        result = convert_messages_to_prompt(msgs)
        assert result == "Tool Result (search): result data"

    def test_tool_message_without_name(self):
        """A tool message without a name should omit the parenthetical."""
        msgs = [
            ToolMessage(content="result data", tool_call_id="tc_1"),
        ]
        result = convert_messages_to_prompt(msgs)
        assert result == "Tool Result: result data"

    def test_empty_messages(self):
        """An empty message list should produce an empty string."""
        result = convert_messages_to_prompt([])
        assert result == ""

    def test_system_messages_skipped(self):
        """System messages should be skipped in the prompt."""
        msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="hello"),
        ]
        result = convert_messages_to_prompt(msgs)
        assert result == "Human: hello"


class TestMapUsage:
    """Tests for map_sdk_usage."""

    def test_empty_usage(self):
        """An empty dict should return an empty dict."""
        result = map_sdk_usage({})
        assert result == {}

    def test_none_usage(self):
        """None should return an empty dict."""
        result = map_sdk_usage(None)
        assert result == {}

    def test_basic_usage(self):
        """Basic usage with input and output tokens should map correctly."""
        sdk_usage = {"input_tokens": 10, "output_tokens": 20}
        result = map_sdk_usage(sdk_usage)
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_cache_usage(self):
        """Cache usage fields should appear in input_token_details."""
        sdk_usage = {
            "input_tokens": 50,
            "output_tokens": 25,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10,
        }
        result = map_sdk_usage(sdk_usage)
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 25
        assert result["total_tokens"] == 75
        assert "input_token_details" in result
        details = result["input_token_details"]
        assert details["cache_read"] == 30
        assert details["cache_creation"] == 10
