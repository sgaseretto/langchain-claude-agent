"""Tests for ChatClaudeAgSDK."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel

from langchain_claude_agent.chat_model import ChatClaudeAgSDK, _schema_to_output_format

# Patch targets for SDK functions used by chat_model
_PATCH_QUERY = "langchain_claude_agent.chat_model.sdk_query"
_PATCH_TOOL_DECORATOR = "langchain_claude_agent.chat_model.sdk_tool_decorator"
_PATCH_MCP_SERVER = "langchain_claude_agent.chat_model.create_sdk_mcp_server"

# ---------------------------------------------------------------------------
# Helpers: mock SDK messages
# ---------------------------------------------------------------------------


def _make_assistant_message(text: str) -> AssistantMessage:
    """Create an SDK AssistantMessage with a single text block.

    Args:
        text: The text content for the assistant message block.

    Returns:
        An ``AssistantMessage`` instance with one ``TextBlock``.
    """
    return AssistantMessage(content=[TextBlock(text=text)], model="sonnet")


def _make_assistant_message_with_thinking(
    text: str,
    thinking: str = "Let me think...",
    signature: str = "sig123",
) -> AssistantMessage:
    """Create an SDK AssistantMessage with thinking + text blocks.

    Args:
        text: The text content for the text block.
        thinking: The thinking content.
        signature: The thinking block signature.

    Returns:
        An ``AssistantMessage`` with a ``ThinkingBlock`` followed by a ``TextBlock``.
    """
    return AssistantMessage(
        content=[
            ThinkingBlock(thinking=thinking, signature=signature),
            TextBlock(text=text),
        ],
        model="sonnet",
    )


def _make_result_message(
    usage: dict | None = None,
    total_cost: float | None = None,
    structured_output: object | None = None,
) -> ResultMessage:
    """Create an SDK ResultMessage.

    Args:
        usage: Token usage dictionary.
        total_cost: Total cost in USD.
        structured_output: Optional structured output data.

    Returns:
        A ``ResultMessage`` instance with the given usage and cost.
    """
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="test-session",
        total_cost_usd=total_cost or 0.001,
        usage=usage or {"input_tokens": 10, "output_tokens": 5},
    )
    if structured_output is not None:
        msg.structured_output = structured_output
    return msg


def _make_stream_event(text: str) -> StreamEvent:
    """Create an SDK StreamEvent with a text_delta content block delta.

    Args:
        text: The text content for the text_delta event.

    Returns:
        A ``StreamEvent`` instance with a ``content_block_delta`` event.
    """
    return StreamEvent(
        uuid="test-uuid",
        session_id="test-session",
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    )


# ---------------------------------------------------------------------------
# TestChatClaudeAgSDKInit
# ---------------------------------------------------------------------------


class TestChatClaudeAgSDKInit:
    """Tests for ChatClaudeAgSDK initialisation and properties."""

    def test_default_model(self):
        """Default model should be 'sonnet'."""
        agent = ChatClaudeAgSDK()
        assert agent.model == "sonnet"

    def test_custom_model(self):
        """A custom model string should be stored."""
        agent = ChatClaudeAgSDK(model="opus")
        assert agent.model == "opus"

    def test_llm_type(self):
        """The _llm_type property should return 'claude-agent-sdk'."""
        agent = ChatClaudeAgSDK()
        assert agent._llm_type == "claude-agent-sdk"

    def test_default_permission_mode(self):
        """Default permission mode should be 'bypassPermissions'."""
        agent = ChatClaudeAgSDK()
        assert agent.permission_mode == "bypassPermissions"

    def test_custom_fields(self):
        """Custom fields should be stored correctly."""
        agent = ChatClaudeAgSDK(
            model="haiku",
            max_turns=5,
            max_budget_usd=1.0,
            allowed_tools=["Read", "Bash"],
            system_prompt="Be helpful.",
            cwd="/tmp",
        )
        assert agent.max_turns == 5
        assert agent.max_budget_usd == 1.0
        assert agent.allowed_tools == ["Read", "Bash"]
        assert agent.system_prompt == "Be helpful."
        assert agent.cwd == "/tmp"

    def test_init_with_thinking_config(self):
        """Thinking config should be stored."""
        thinking = {"type": "enabled", "budget_tokens": 10000}
        agent = ChatClaudeAgSDK(thinking=thinking)
        assert agent.thinking == thinking
        assert agent.thinking["budget_tokens"] == 10000

    def test_init_with_effort(self):
        """Effort level should be stored."""
        agent = ChatClaudeAgSDK(effort="high")
        assert agent.effort == "high"

    def test_init_defaults_thinking_and_effort_none(self):
        """Thinking and effort should default to None."""
        agent = ChatClaudeAgSDK()
        assert agent.thinking is None
        assert agent.effort is None


# ---------------------------------------------------------------------------
# TestBuildOptions
# ---------------------------------------------------------------------------


class TestBuildOptions:
    """Tests for _build_options helper."""

    def test_build_options_basic(self):
        """Options should reflect the agent's configuration."""
        agent = ChatClaudeAgSDK(model="opus", max_turns=3)
        options = agent._build_options(system_prompt="test prompt")
        assert options.model == "opus"
        assert options.system_prompt == "test prompt"
        assert options.max_turns == 3

    def test_build_options_fallback_system_prompt(self):
        """When system_prompt arg is None, use self.system_prompt."""
        agent = ChatClaudeAgSDK(system_prompt="default prompt")
        options = agent._build_options(system_prompt=None)
        assert options.system_prompt == "default prompt"

    def test_build_options_override_system_prompt(self):
        """An explicit system_prompt arg should override self.system_prompt."""
        agent = ChatClaudeAgSDK(system_prompt="default prompt")
        options = agent._build_options(system_prompt="override prompt")
        assert options.system_prompt == "override prompt"

    def test_build_options_passes_thinking_and_effort(self):
        """Thinking and effort should be passed to ClaudeAgentOptions."""
        thinking = {"type": "enabled", "budget_tokens": 5000}
        agent = ChatClaudeAgSDK(thinking=thinking, effort="high")
        options = agent._build_options()
        assert options.thinking == thinking
        assert options.effort == "high"

    def test_build_options_passes_output_format(self):
        """Output format should be passed to ClaudeAgentOptions."""
        agent = ChatClaudeAgSDK()
        output_format = {"type": "json_schema", "schema": {"type": "object"}}
        options = agent._build_options(output_format=output_format)
        assert options.output_format == output_format


# ---------------------------------------------------------------------------
# TestChatClaudeAgSDKGenerate
# ---------------------------------------------------------------------------


class TestChatClaudeAgSDKGenerate:
    """Tests for the async generation path (query)."""

    @pytest.mark.asyncio
    async def test_agenerate_basic(self):
        """A basic query should return an AIMessage with the assistant text."""
        assistant_msg = _make_assistant_message("Hello, world!")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        ai_msg = result.generations[0].message
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_agenerate_with_system_message(self):
        """A SystemMessage should be extracted and passed as the system prompt."""
        assistant_msg = _make_assistant_message("response")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query) as mock_q:
            await agent._agenerate(
                [
                    SystemMessage(content="You are a pirate."),
                    HumanMessage(content="Hi"),
                ]
            )
            call_kwargs = mock_q.call_args
            options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
            assert options.system_prompt == "You are a pirate."

    @pytest.mark.asyncio
    async def test_agenerate_returns_usage_metadata(self):
        """Usage metadata should be mapped from the ResultMessage."""
        assistant_msg = _make_assistant_message("answer")
        result_msg = _make_result_message(
            usage={"input_tokens": 100, "output_tokens": 50}
        )

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        ai_msg = result.generations[0].message
        assert ai_msg.usage_metadata["input_tokens"] == 100
        assert ai_msg.usage_metadata["output_tokens"] == 50
        assert ai_msg.usage_metadata["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_agenerate_fallback_system_prompt(self):
        """When no SystemMessage is present, self.system_prompt should be used."""
        assistant_msg = _make_assistant_message("ok")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK(system_prompt="default system")
        with patch(_PATCH_QUERY, side_effect=_mock_query) as mock_q:
            await agent._agenerate([HumanMessage(content="Hi")])
            call_kwargs = mock_q.call_args
            options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
            assert options.system_prompt == "default system"

    @pytest.mark.asyncio
    async def test_agenerate_system_message_overrides_default(self):
        """A SystemMessage in the input should override self.system_prompt."""
        assistant_msg = _make_assistant_message("ok")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK(system_prompt="default system")
        with patch(_PATCH_QUERY, side_effect=_mock_query) as mock_q:
            await agent._agenerate(
                [
                    SystemMessage(content="override system"),
                    HumanMessage(content="Hi"),
                ]
            )
            call_kwargs = mock_q.call_args
            options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
            assert options.system_prompt == "override system"

    @pytest.mark.asyncio
    async def test_agenerate_multiple_text_blocks(self):
        """Multiple text blocks in a single AssistantMessage should be concatenated."""
        msg = AssistantMessage(
            content=[TextBlock(text="Hello "), TextBlock(text="world!")],
            model="sonnet",
        )
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "Hello world!"

    @pytest.mark.asyncio
    async def test_agenerate_response_metadata_contains_model(self):
        """Response metadata should include the model name."""
        assistant_msg = _make_assistant_message("hi")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK(model="haiku")
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        ai_msg = result.generations[0].message
        assert ai_msg.response_metadata["model"] == "haiku"

    @pytest.mark.asyncio
    async def test_agenerate_extracts_thinking_blocks(self):
        """Thinking blocks should be extracted into additional_kwargs."""
        msg = _make_assistant_message_with_thinking(
            text="The answer is 555.",
            thinking="15 * 37 = 555",
            signature="sig-abc",
        )
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield msg
            yield result_msg

        agent = ChatClaudeAgSDK(thinking={"type": "enabled", "budget_tokens": 5000})
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="What is 15*37?")])

        ai_msg = result.generations[0].message
        assert ai_msg.content == "The answer is 555."
        assert "thinking" in ai_msg.additional_kwargs
        blocks = ai_msg.additional_kwargs["thinking"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "15 * 37 = 555"
        assert blocks[0]["signature"] == "sig-abc"

    @pytest.mark.asyncio
    async def test_agenerate_no_thinking_no_additional_kwargs(self):
        """When no thinking blocks, additional_kwargs should not have 'thinking'."""
        msg = _make_assistant_message("plain response")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        ai_msg = result.generations[0].message
        assert "thinking" not in ai_msg.additional_kwargs

    @pytest.mark.asyncio
    async def test_agenerate_passes_output_format_to_options(self):
        """Output format from kwargs should be passed to _build_options."""
        msg = _make_assistant_message('{"name": "test"}')
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query) as mock_q:
            await agent._agenerate(
                [HumanMessage(content="Hi")],
                output_format={"type": "json_schema", "schema": {"type": "object"}},
            )
            call_kwargs = mock_q.call_args
            options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
            assert options.output_format == {
                "type": "json_schema",
                "schema": {"type": "object"},
            }

    @pytest.mark.asyncio
    async def test_agenerate_uses_structured_output_from_result(self):
        """When ResultMessage has structured_output, it should be used as content."""
        msg = _make_assistant_message("ignored text")
        structured = {"name": "Alice", "age": 30}
        result_msg = _make_result_message(structured_output=structured)

        async def _mock_query(**kwargs):
            yield msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        ai_msg = result.generations[0].message
        assert json.loads(ai_msg.content) == structured


# ---------------------------------------------------------------------------
# TestChatClaudeAgSDKTools
# ---------------------------------------------------------------------------


@lc_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class TestChatClaudeAgSDKTools:
    """Tests for bind_tools and tool calling via query()."""

    def test_bind_tools_returns_runnable(self):
        """bind_tools should return a Runnable with invoke/ainvoke."""
        llm = ChatClaudeAgSDK()
        bound = llm.bind_tools([add])
        assert hasattr(bound, "invoke")
        assert hasattr(bound, "ainvoke")

    def test_bind_tools_stores_tools_in_kwargs(self):
        """bind_tools should store tools in the RunnableBinding kwargs."""
        llm = ChatClaudeAgSDK()
        bound = llm.bind_tools([add])
        assert "tools" in bound.kwargs
        assert bound.kwargs["tools"] == [add]

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_attaches_mcp_server(self):
        """_agenerate with tools should attach MCP server to options and call query()."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="What is 3 + 4?")]

        captured_kwargs = {}

        async def mock_query(**kwargs):
            captured_kwargs.update(kwargs)
            yield _make_assistant_message("The sum of 3 and 4 is 7.")
            yield _make_result_message()

        mock_mcp = MagicMock()

        with (
            patch(_PATCH_QUERY, side_effect=mock_query),
            patch(_PATCH_TOOL_DECORATOR, side_effect=lambda n, d, s: lambda fn: fn),
            patch(_PATCH_MCP_SERVER, return_value=mock_mcp),
        ):
            result = await llm._agenerate(messages, tools=[add])

        assert "7" in result.generations[0].message.content
        options = captured_kwargs["options"]
        assert "langchain-tools" in options.mcp_servers
        assert options.mcp_servers["langchain-tools"] is mock_mcp

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_returns_usage(self):
        """_agenerate with tools should propagate usage metadata."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="What is 3 + 4?")]

        async def mock_query(**kwargs):
            yield _make_assistant_message("7")
            yield _make_result_message(
                usage={"input_tokens": 200, "output_tokens": 100}
            )

        with (
            patch(_PATCH_QUERY, side_effect=mock_query),
            patch(_PATCH_TOOL_DECORATOR, side_effect=lambda n, d, s: lambda fn: fn),
            patch(_PATCH_MCP_SERVER, return_value=MagicMock()),
        ):
            result = await llm._agenerate(messages, tools=[add])

        ai_msg = result.generations[0].message
        assert ai_msg.usage_metadata["input_tokens"] == 200
        assert ai_msg.usage_metadata["output_tokens"] == 100

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_sets_allowed_tools(self):
        """_agenerate with tools should add tool names to allowed_tools."""
        llm = ChatClaudeAgSDK(allowed_tools=["Read"])
        messages = [HumanMessage(content="What is 3 + 4?")]

        captured_kwargs = {}

        async def mock_query(**kwargs):
            captured_kwargs.update(kwargs)
            yield _make_assistant_message("7")
            yield _make_result_message()

        with (
            patch(_PATCH_QUERY, side_effect=mock_query),
            patch(_PATCH_TOOL_DECORATOR, side_effect=lambda n, d, s: lambda fn: fn),
            patch(_PATCH_MCP_SERVER, return_value=MagicMock()),
        ):
            await llm._agenerate(messages, tools=[add])

        options = captured_kwargs["options"]
        assert "mcp__langchain-tools__add" in options.allowed_tools
        assert "Read" in options.allowed_tools

    @pytest.mark.asyncio
    async def test_agenerate_without_tools_has_no_mcp(self):
        """_agenerate without tools should not attach MCP servers."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="Hello")]

        captured_kwargs = {}

        async def mock_query(**kwargs):
            captured_kwargs.update(kwargs)
            yield _make_assistant_message("Hi")
            yield _make_result_message()

        with patch(_PATCH_QUERY, side_effect=mock_query):
            await llm._agenerate(messages)

        options = captured_kwargs["options"]
        assert not options.mcp_servers


# ---------------------------------------------------------------------------
# TestChatClaudeAgSDKSync
# ---------------------------------------------------------------------------


class TestChatClaudeAgSDKSync:
    """Tests for the sync _generate wrapper."""

    def test_sync_generate(self):
        """The sync _generate should delegate to _agenerate."""
        assistant_msg = _make_assistant_message("sync result")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgSDK()
        with patch(_PATCH_QUERY, side_effect=_mock_query):
            result = agent._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "sync result"


# ---------------------------------------------------------------------------
# TestChatClaudeAgSDKStream
# ---------------------------------------------------------------------------


class TestChatClaudeAgSDKStream:
    """Tests for _stream and _astream."""

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self):
        """_astream should yield ChatGenerationChunk for each text delta."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="Tell me a story")]

        async def mock_query(*args, **kwargs):
            yield _make_stream_event("Once ")
            yield _make_stream_event("upon ")
            yield _make_stream_event("a time")
            yield _make_result_message(usage={"input_tokens": 10, "output_tokens": 6})

        with patch(_PATCH_QUERY, mock_query):
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk)

        text_chunks = [c for c in chunks if c.message.content]
        assert len(text_chunks) == 3
        full_text = "".join(c.message.content for c in text_chunks)
        assert full_text == "Once upon a time"

    @pytest.mark.asyncio
    async def test_astream_final_chunk_has_usage(self):
        """The final chunk from _astream should carry usage metadata."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="hello")]

        async def mock_query(*args, **kwargs):
            yield _make_stream_event("hi")
            yield _make_result_message(usage={"input_tokens": 50, "output_tokens": 25})

        with patch(_PATCH_QUERY, mock_query):
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk)

        last_chunk = chunks[-1]
        assert last_chunk.message.usage_metadata["input_tokens"] == 50

    @pytest.mark.asyncio
    async def test_astream_calls_run_manager(self):
        """_astream should call run_manager.on_llm_new_token for each delta."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="hello")]

        async def mock_query(*args, **kwargs):
            yield _make_stream_event("hi")
            yield _make_result_message()

        mock_manager = AsyncMock()

        with patch(_PATCH_QUERY, mock_query):
            async for _ in llm._astream(messages, run_manager=mock_manager):
                pass

        mock_manager.on_llm_new_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_astream_uses_include_partial_messages(self):
        """_astream should set include_partial_messages=True in options."""
        llm = ChatClaudeAgSDK()
        messages = [HumanMessage(content="hello")]

        captured_kwargs = {}

        async def mock_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield _make_result_message()

        with patch(_PATCH_QUERY, mock_query):
            async for _ in llm._astream(messages):
                pass

        options = captured_kwargs.get("options")
        assert options.include_partial_messages is True

    @pytest.mark.asyncio
    async def test_astream_with_assistant_message_fallback(self):
        """When SDK returns AssistantMessage (e.g. with thinking), _astream should handle it."""
        llm = ChatClaudeAgSDK(thinking={"type": "enabled", "budget_tokens": 5000})
        messages = [HumanMessage(content="Think about this")]

        msg = _make_assistant_message_with_thinking(
            text="The result",
            thinking="Deep thought...",
            signature="sig-xyz",
        )

        async def mock_query(*args, **kwargs):
            yield msg
            yield _make_result_message()

        with patch(_PATCH_QUERY, mock_query):
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk)

        # Should have at least one text chunk and one usage chunk
        text_chunks = [c for c in chunks if c.message.content]
        assert len(text_chunks) >= 1
        assert text_chunks[0].message.content == "The result"
        # First text chunk should carry thinking blocks
        assert "thinking" in text_chunks[0].message.additional_kwargs
        thinking = text_chunks[0].message.additional_kwargs["thinking"]
        assert thinking[0]["thinking"] == "Deep thought..."


# ---------------------------------------------------------------------------
# TestSchemaToOutputFormat
# ---------------------------------------------------------------------------


class _TestModel(BaseModel):
    """A test Pydantic model for structured output."""

    name: str
    age: int


class TestSchemaToOutputFormat:
    """Tests for the _schema_to_output_format helper."""

    def test_pydantic_schema(self):
        """Pydantic model should produce a json_schema output format."""
        fmt = _schema_to_output_format(_TestModel)
        assert fmt["type"] == "json_schema"
        schema = fmt["schema"]
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_dict_schema(self):
        """A plain dict should be used directly as the JSON schema."""
        raw_schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        fmt = _schema_to_output_format(raw_schema)
        assert fmt["type"] == "json_schema"
        assert fmt["schema"] == raw_schema


# ---------------------------------------------------------------------------
# TestWithStructuredOutput
# ---------------------------------------------------------------------------


class TestWithStructuredOutput:
    """Tests for with_structured_output method."""

    def test_returns_runnable(self):
        """with_structured_output should return a Runnable."""
        llm = ChatClaudeAgSDK()
        chain = llm.with_structured_output(_TestModel)
        assert hasattr(chain, "invoke")
        assert hasattr(chain, "ainvoke")
