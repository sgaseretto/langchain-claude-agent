"""Tests for ChatClaudeAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

from langchain_claude_agent.chat_model import ChatClaudeAgent

# ---------------------------------------------------------------------------
# Helpers: mock SDK messages
# ---------------------------------------------------------------------------


def _make_assistant_message(text: str) -> MagicMock:
    """Create a mock SDK AssistantMessage.

    Args:
        text: The text content for the assistant message block.

    Returns:
        A MagicMock whose type name is ``AssistantMessage`` and whose
        ``content`` attribute contains a single text block.
    """
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    type(msg).__name__ = "AssistantMessage"
    # Ensure it has the attributes our duck-typing helpers check
    # AssistantMessage has `content` but NOT `usage` / `subtype`
    del msg.usage
    del msg.subtype
    return msg


def _make_result_message(
    usage: dict | None = None,
    total_cost: float | None = None,
) -> MagicMock:
    """Create a mock SDK ResultMessage.

    Args:
        usage: Token usage dictionary.
        total_cost: Total cost in USD.

    Returns:
        A MagicMock whose type name is ``ResultMessage`` with ``usage``,
        ``total_cost_usd``, and ``subtype`` attributes.
    """
    msg = MagicMock()
    msg.usage = usage or {"input_tokens": 10, "output_tokens": 5}
    msg.total_cost_usd = total_cost or 0.001
    msg.subtype = "success"
    type(msg).__name__ = "ResultMessage"
    return msg


def _make_stream_event(text: str) -> MagicMock:
    """Create a mock SDK StreamEvent with text delta.

    Args:
        text: The text content for the text_delta event.

    Returns:
        A MagicMock whose ``event`` attribute is a dict with
        ``content_block_delta`` type, and whose ``content``, ``usage``,
        and ``subtype`` attributes are removed to avoid duck-typing
        misclassification.
    """
    evt = MagicMock()
    evt.event = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
    }
    # Remove default MagicMock attrs that would confuse duck typing
    del evt.content
    del evt.usage
    del evt.subtype
    return evt


# ---------------------------------------------------------------------------
# TestChatClaudeAgentInit
# ---------------------------------------------------------------------------


class TestChatClaudeAgentInit:
    """Tests for ChatClaudeAgent initialisation and properties."""

    def test_default_model(self):
        """Default model should be 'sonnet'."""
        agent = ChatClaudeAgent()
        assert agent.model == "sonnet"

    def test_custom_model(self):
        """A custom model string should be stored."""
        agent = ChatClaudeAgent(model="opus")
        assert agent.model == "opus"

    def test_llm_type(self):
        """The _llm_type property should return 'claude-agent-sdk'."""
        agent = ChatClaudeAgent()
        assert agent._llm_type == "claude-agent-sdk"

    def test_default_permission_mode(self):
        """Default permission mode should be 'bypassPermissions'."""
        agent = ChatClaudeAgent()
        assert agent.permission_mode == "bypassPermissions"

    def test_custom_fields(self):
        """Custom fields should be stored correctly."""
        agent = ChatClaudeAgent(
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


# ---------------------------------------------------------------------------
# TestBuildOptions
# ---------------------------------------------------------------------------


class TestBuildOptions:
    """Tests for _build_options helper."""

    def test_build_options_basic(self):
        """Options should reflect the agent's configuration."""
        agent = ChatClaudeAgent(model="opus", max_turns=3)
        options = agent._build_options(system_prompt="test prompt")
        assert options.model == "opus"
        assert options.system_prompt == "test prompt"
        assert options.max_turns == 3

    def test_build_options_fallback_system_prompt(self):
        """When system_prompt arg is None, use self.system_prompt."""
        agent = ChatClaudeAgent(system_prompt="default prompt")
        options = agent._build_options(system_prompt=None)
        assert options.system_prompt == "default prompt"

    def test_build_options_override_system_prompt(self):
        """An explicit system_prompt arg should override self.system_prompt."""
        agent = ChatClaudeAgent(system_prompt="default prompt")
        options = agent._build_options(system_prompt="override prompt")
        assert options.system_prompt == "override prompt"


# ---------------------------------------------------------------------------
# TestChatClaudeAgentGenerate
# ---------------------------------------------------------------------------


class TestChatClaudeAgentGenerate:
    """Tests for the async generation path (query)."""

    @pytest.mark.asyncio
    async def test_agenerate_basic(self):
        """A basic query should return an AIMessage with the assistant text."""
        assistant_msg = _make_assistant_message("Hello, world!")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgent()
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ):
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

        agent = ChatClaudeAgent()
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ) as mock_q:
            await agent._agenerate(
                [
                    SystemMessage(content="You are a pirate."),
                    HumanMessage(content="Hi"),
                ]
            )
            # Check that the options passed to sdk_query include the system prompt
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

        agent = ChatClaudeAgent()
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ):
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

        agent = ChatClaudeAgent(system_prompt="default system")
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ) as mock_q:
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

        agent = ChatClaudeAgent(system_prompt="default system")
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ) as mock_q:
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
        block1 = MagicMock()
        block1.text = "Hello "
        block2 = MagicMock()
        block2.text = "world!"
        msg = MagicMock()
        msg.content = [block1, block2]
        type(msg).__name__ = "AssistantMessage"
        del msg.usage
        del msg.subtype

        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield msg
            yield result_msg

        agent = ChatClaudeAgent()
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ):
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

        agent = ChatClaudeAgent(model="haiku")
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ):
            result = await agent._agenerate([HumanMessage(content="Hi")])

        ai_msg = result.generations[0].message
        assert ai_msg.response_metadata["model"] == "haiku"


# ---------------------------------------------------------------------------
# TestChatClaudeAgentWithClient
# ---------------------------------------------------------------------------


class TestChatClaudeAgentWithClient:
    """Tests for the client path (tool calling)."""

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_uses_client_path(self):
        """Passing tools should route to _agenerate_with_client."""
        agent = ChatClaudeAgent()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def mock_receive():
            yield _make_assistant_message("client response")
            yield _make_result_message()

        mock_client.receive_response = mock_receive

        with patch(
            "langchain_claude_agent.chat_model.ClaudeSDKClient",
            return_value=mock_client,
        ):
            with patch(
                "langchain_claude_agent.chat_model.sdk_tool_decorator",
                side_effect=lambda n, d, s: lambda fn: fn,
            ):
                with patch(
                    "langchain_claude_agent.chat_model.create_sdk_mcp_server",
                    return_value=MagicMock(),
                ):
                    result = await agent._agenerate(
                        [HumanMessage(content="Hi")],
                        tools=[MagicMock(spec=["name", "description", "args_schema"])],
                    )

        assert "client response" in result.generations[0].message.content


# ---------------------------------------------------------------------------
# TestChatClaudeAgentSync
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestChatClaudeAgentTools
# ---------------------------------------------------------------------------


@lc_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class TestChatClaudeAgentTools:
    """Tests for bind_tools and tool calling."""

    def test_bind_tools_returns_runnable(self):
        """bind_tools should return a Runnable with invoke/ainvoke."""
        llm = ChatClaudeAgent()
        bound = llm.bind_tools([add])
        assert hasattr(bound, "invoke")
        assert hasattr(bound, "ainvoke")

    def test_bind_tools_stores_tools_in_kwargs(self):
        """bind_tools should store tools in the RunnableBinding kwargs."""
        llm = ChatClaudeAgent()
        bound = llm.bind_tools([add])
        assert "tools" in bound.kwargs
        assert bound.kwargs["tools"] == [add]

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_uses_client(self):
        """_agenerate with tools should route through ClaudeSDKClient."""
        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="What is 3 + 4?")]

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def mock_receive():
            yield _make_assistant_message("The sum of 3 and 4 is 7.")
            yield _make_result_message()

        mock_client.receive_response = mock_receive

        with patch(
            "langchain_claude_agent.chat_model.ClaudeSDKClient",
            return_value=mock_client,
        ):
            with patch(
                "langchain_claude_agent.chat_model.sdk_tool_decorator",
                side_effect=lambda n, d, s: lambda fn: fn,
            ):
                with patch(
                    "langchain_claude_agent.chat_model.create_sdk_mcp_server",
                    return_value=MagicMock(),
                ):
                    result = await llm._agenerate(messages, tools=[add])

        assert "7" in result.generations[0].message.content

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_returns_usage(self):
        """_agenerate with tools should propagate usage metadata."""
        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="What is 3 + 4?")]

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def mock_receive():
            yield _make_assistant_message("7")
            yield _make_result_message(
                usage={"input_tokens": 200, "output_tokens": 100}
            )

        mock_client.receive_response = mock_receive

        with patch(
            "langchain_claude_agent.chat_model.ClaudeSDKClient",
            return_value=mock_client,
        ):
            with patch(
                "langchain_claude_agent.chat_model.sdk_tool_decorator",
                side_effect=lambda n, d, s: lambda fn: fn,
            ):
                with patch(
                    "langchain_claude_agent.chat_model.create_sdk_mcp_server",
                    return_value=MagicMock(),
                ):
                    result = await llm._agenerate(messages, tools=[add])

        ai_msg = result.generations[0].message
        assert ai_msg.usage_metadata["input_tokens"] == 200
        assert ai_msg.usage_metadata["output_tokens"] == 100


# ---------------------------------------------------------------------------
# TestChatClaudeAgentSync
# ---------------------------------------------------------------------------


class TestChatClaudeAgentSync:
    """Tests for the sync _generate wrapper."""

    def test_sync_generate(self):
        """The sync _generate should delegate to _agenerate."""
        assistant_msg = _make_assistant_message("sync result")
        result_msg = _make_result_message()

        async def _mock_query(**kwargs):
            yield assistant_msg
            yield result_msg

        agent = ChatClaudeAgent()
        with patch(
            "langchain_claude_agent.chat_model.sdk_query",
            side_effect=_mock_query,
        ):
            result = agent._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "sync result"


# ---------------------------------------------------------------------------
# TestChatClaudeAgentStream
# ---------------------------------------------------------------------------


class TestChatClaudeAgentStream:
    """Tests for _stream and _astream."""

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self):
        """_astream should yield ChatGenerationChunk for each text delta."""
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="Tell me a story")]

        async def mock_query(*args, **kwargs):
            yield _make_stream_event("Once ")
            yield _make_stream_event("upon ")
            yield _make_stream_event("a time")
            yield _make_result_message(usage={"input_tokens": 10, "output_tokens": 6})

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
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
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="hello")]

        async def mock_query(*args, **kwargs):
            yield _make_stream_event("hi")
            yield _make_result_message(usage={"input_tokens": 50, "output_tokens": 25})

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk)

        # Last chunk should have usage metadata
        last_chunk = chunks[-1]
        assert last_chunk.message.usage_metadata["input_tokens"] == 50

    @pytest.mark.asyncio
    async def test_astream_calls_run_manager(self):
        """_astream should call run_manager.on_llm_new_token for each delta."""
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="hello")]

        async def mock_query(*args, **kwargs):
            yield _make_stream_event("hi")
            yield _make_result_message()

        mock_manager = AsyncMock()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            async for _ in llm._astream(messages, run_manager=mock_manager):
                pass

        mock_manager.on_llm_new_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_astream_uses_include_partial_messages(self):
        """_astream should set include_partial_messages=True in options."""
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="hello")]

        captured_kwargs = {}

        async def mock_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield _make_result_message()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            async for _ in llm._astream(messages):
                pass

        options = captured_kwargs.get("options")
        assert options.include_partial_messages is True
