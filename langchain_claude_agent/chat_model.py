"""ChatClaudeAgent - LangChain BaseChatModel wrapping the Claude Agent SDK."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, Optional

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server
from claude_agent_sdk import query as sdk_query
from claude_agent_sdk import tool as sdk_tool_decorator
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable

from langchain_claude_agent._tool_converter import convert_langchain_tools
from langchain_claude_agent._types import (
    DEFAULT_MODEL,
    DEFAULT_PERMISSION_MODE,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    TOOL_NAME_PREFIX,
)
from langchain_claude_agent._utils import (
    convert_messages_to_prompt,
    extract_system_message,
    map_sdk_usage,
)


def _is_assistant_message(msg: Any) -> bool:
    """Check whether *msg* looks like an SDK AssistantMessage (duck-typing).

    Args:
        msg: An object yielded by ``sdk_query``.

    Returns:
        True when *msg* carries ``content`` but lacks ``usage`` and ``subtype``
        (which are present only on ResultMessage).
    """
    return hasattr(msg, "content") and not hasattr(msg, "usage")


def _is_result_message(msg: Any) -> bool:
    """Check whether *msg* looks like an SDK ResultMessage (duck-typing).

    Args:
        msg: An object yielded by ``sdk_query``.

    Returns:
        True when *msg* carries both ``usage`` and ``subtype`` attributes.
    """
    return hasattr(msg, "usage") and hasattr(msg, "subtype")


def _is_stream_event(msg: Any) -> bool:
    """Check whether *msg* looks like an SDK StreamEvent.

    Args:
        msg: An object yielded by ``sdk_query``.

    Returns:
        True when *msg* has an ``event`` attribute that is a dict.
    """
    return hasattr(msg, "event") and isinstance(getattr(msg, "event", None), dict)


class ChatClaudeAgent(BaseChatModel):
    """LangChain chat model wrapping the Claude Agent SDK.

    Uses ``query()`` for basic text generation and ``ClaudeSDKClient``
    when tools are bound via ``bind_tools()``.

    Args:
        model: SDK model name (e.g. "sonnet", "opus", "haiku").
        max_turns: Maximum agentic turns.
        max_budget_usd: Budget limit in USD.
        allowed_tools: SDK built-in tools (e.g. ["Read", "Bash"]).
        system_prompt: Default system prompt (overridden by SystemMessage).
        permission_mode: SDK permission mode.
        cwd: Working directory for the SDK.
    """

    model: str = DEFAULT_MODEL
    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    allowed_tools: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    permission_mode: str = DEFAULT_PERMISSION_MODE
    cwd: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "claude-agent-sdk"

    def _build_options(
        self,
        system_prompt: str | None = None,
        *,
        include_partial_messages: bool = False,
    ) -> ClaudeAgentOptions:
        """Build a ``ClaudeAgentOptions`` from the agent's configuration.

        Args:
            system_prompt: Explicit system prompt that takes priority over
                ``self.system_prompt``.  When *None*, ``self.system_prompt``
                is used as a fallback.
            include_partial_messages: Whether the SDK should yield partial
                (streaming) messages.

        Returns:
            A fully-populated ``ClaudeAgentOptions`` instance.
        """
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            allowed_tools=self.allowed_tools or [],
            permission_mode=self.permission_mode,
            cwd=self.cwd,
            include_partial_messages=include_partial_messages,
        )

    # --------------------------------------------------------------------- #
    # Tool binding
    # --------------------------------------------------------------------- #

    def bind_tools(
        self,
        tools: list,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind LangChain tools for autonomous SDK execution.

        Args:
            tools: List of LangChain tool instances.
            tool_choice: Not used (SDK handles tool selection autonomously).
            **kwargs: Additional arguments passed to bind().

        Returns:
            A Runnable with tools stored in kwargs.
        """
        return self.bind(tools=tools, **kwargs)

    # --------------------------------------------------------------------- #
    # Async generation
    # --------------------------------------------------------------------- #

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async entry-point: route to query or client depending on tools.

        Args:
            messages: LangChain messages forming the conversation.
            stop: Optional stop sequences (unused by SDK but part of the API).
            run_manager: LangChain async callback manager.
            **kwargs: Extra keyword arguments.  When ``tools`` is present, the
                client path is used.

        Returns:
            A ``ChatResult`` with one ``ChatGeneration`` containing an
            ``AIMessage``.
        """
        tools = kwargs.pop("tools", None)
        if tools:
            return await self._agenerate_with_client(
                messages, tools, stop, run_manager, **kwargs
            )
        return await self._agenerate_with_query(messages, stop, run_manager, **kwargs)

    async def _agenerate_with_query(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the SDK ``query()`` async generator.

        Args:
            messages: LangChain messages forming the conversation.
            stop: Optional stop sequences.
            run_manager: LangChain async callback manager.
            **kwargs: Extra keyword arguments (forwarded but currently unused).

        Returns:
            A ``ChatResult`` wrapping the assistant's text response and usage
            metadata.
        """
        system_prompt, chat_messages = extract_system_message(messages)
        prompt = convert_messages_to_prompt(chat_messages)
        options = self._build_options(system_prompt)

        result_text = ""
        usage_data: dict = {}

        async for message in sdk_query(prompt=prompt, options=options):
            if _is_assistant_message(message):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text
            elif _is_result_message(message):
                usage_data = message.usage or {}

        ai_msg = AIMessage(
            content=result_text,
            usage_metadata=map_sdk_usage(usage_data),
            response_metadata={"model": self.model},
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate_with_client(
        self,
        messages: list[BaseMessage],
        tools: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate using ClaudeSDKClient with bound tools.

        Converts LangChain tools to SDK MCP tools, creates an MCP server,
        and runs a ClaudeSDKClient session to get the response.

        Args:
            messages: LangChain messages.
            tools: LangChain tool instances to convert and bind.
            stop: Stop sequences.
            run_manager: Callback manager.
            **kwargs: Extra arguments.

        Returns:
            ChatResult with AIMessage.
        """
        system_prompt, chat_messages = extract_system_message(messages)
        prompt = convert_messages_to_prompt(chat_messages)

        # Convert LangChain tools to SDK tool specs
        sdk_tool_specs = convert_langchain_tools(tools)

        # Create SDK @tool decorated functions from our specs
        sdk_tools = []
        for spec in sdk_tool_specs:
            decorated = sdk_tool_decorator(spec.name, spec.description, spec.schema)(
                spec.handler
            )
            sdk_tools.append(decorated)

        # Create MCP server with the tools
        mcp_server = create_sdk_mcp_server(
            name=MCP_SERVER_NAME,
            version=MCP_SERVER_VERSION,
            tools=sdk_tools,
        )
        tool_names = [f"{TOOL_NAME_PREFIX}{spec.name}" for spec in sdk_tool_specs]

        # Build options with MCP server
        options = self._build_options(system_prompt)
        options.mcp_servers = {MCP_SERVER_NAME: mcp_server}
        options.allowed_tools = tool_names + (self.allowed_tools or [])

        result_text = ""
        usage_data: dict = {}

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if _is_assistant_message(message):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += block.text
                elif _is_result_message(message):
                    usage_data = message.usage or {}

        ai_msg = AIMessage(
            content=result_text,
            usage_metadata=map_sdk_usage(usage_data),
            response_metadata={"model": self.model},
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # --------------------------------------------------------------------- #
    # Sync generation
    # --------------------------------------------------------------------- #

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous wrapper around :meth:`_agenerate`.

        Args:
            messages: LangChain messages forming the conversation.
            stop: Optional stop sequences.
            run_manager: LangChain sync callback manager.
            **kwargs: Extra keyword arguments forwarded to ``_agenerate``.

        Returns:
            A ``ChatResult`` produced by the async generation path.
        """
        return asyncio.run(self._agenerate(messages, stop, **kwargs))

    # --------------------------------------------------------------------- #
    # Streaming
    # --------------------------------------------------------------------- #

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream tokens asynchronously using SDK partial messages.

        Args:
            messages: LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager for on_llm_new_token.
            **kwargs: Additional arguments.

        Yields:
            ChatGenerationChunk with AIMessageChunk for each text delta.
        """
        system_prompt, chat_messages = extract_system_message(messages)
        prompt = convert_messages_to_prompt(chat_messages)
        options = self._build_options(system_prompt, include_partial_messages=True)

        async for message in sdk_query(prompt=prompt, options=options):
            if _is_stream_event(message):
                event = message.event
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk
            elif _is_result_message(message):
                usage = map_sdk_usage(message.usage)
                if usage:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content="", usage_metadata=usage)
                    )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens synchronously by wrapping the async stream.

        Args:
            messages: LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager.
            **kwargs: Additional arguments.

        Yields:
            ChatGenerationChunk for each text delta.
        """
        loop = asyncio.new_event_loop()
        try:
            aiter = self._astream(messages, stop, **kwargs)
            while True:
                try:
                    chunk = loop.run_until_complete(aiter.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
