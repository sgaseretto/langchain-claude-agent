"""ChatClaudeAgSDK - LangChain BaseChatModel wrapping the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Iterator, Optional, Type, Union

import nest_asyncio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ThinkingBlock,
    create_sdk_mcp_server,
)
from claude_agent_sdk import query as sdk_query
from claude_agent_sdk import tool as sdk_tool_decorator
from claude_agent_sdk.types import StreamEvent
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_claude_agent._tool_converter import convert_langchain_tools
from langchain_claude_agent._types import (
    DEFAULT_MODEL,
    DEFAULT_PERMISSION_MODE,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    TOOL_NAME_PREFIX,
    EffortLevel,
    OutputFormat,
    ThinkingConfig,
)
from langchain_claude_agent._utils import (
    convert_messages_to_prompt,
    extract_system_message,
    map_sdk_usage,
)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync code, handling existing event loops.

    Uses ``nest_asyncio`` to patch the running loop when called from within
    an existing event loop (e.g. Jupyter notebooks), allowing nested
    ``run_until_complete`` calls on the same loop.

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Patch the running loop to allow nested calls (safe to call multiple times)
    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


def _run_async_iter(aiter: AsyncIterator) -> Iterator:
    """Collect an async iterator from sync code, handling existing event loops.

    Args:
        aiter: An async iterator to consume.

    Yields:
        Items from the async iterator.
    """

    async def _collect():
        return [item async for item in aiter]

    for item in _run_async(_collect()):
        yield item


async def _streaming_prompt(text: str) -> AsyncIterator[dict]:
    """Wrap a plain text prompt as an async generator for SDK streaming input.

    Custom MCP tools require the prompt to be an async generator/iterable
    rather than a plain string.

    Args:
        text: The prompt text to yield as a user message.

    Yields:
        A single user message dict in SDK streaming format.
    """
    yield {
        "type": "user",
        "message": {
            "role": "user",
            "content": text,
        },
    }


def _schema_to_output_format(schema: Any) -> OutputFormat:
    """Convert a schema (Pydantic model, dict, or TypedDict) to SDK output_format.

    Args:
        schema: A Pydantic model class, a JSON schema dict, or a TypedDict class.

    Returns:
        A dict with ``{"type": "json_schema", "schema": ...}`` suitable for
        the SDK's ``output_format`` option.
    """
    if isinstance(schema, dict):
        json_schema = schema
    elif hasattr(schema, "model_json_schema"):
        json_schema = schema.model_json_schema()
    elif hasattr(schema, "__annotations__"):
        # TypedDict - build a simple schema from annotations
        props = {}
        for name, typ in schema.__annotations__.items():
            type_name = getattr(typ, "__name__", str(typ))
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
            }
            props[name] = {"type": type_map.get(type_name, "string")}
        json_schema = {
            "type": "object",
            "properties": props,
            "required": list(schema.__annotations__.keys()),
        }
    else:
        raise ValueError(f"Unsupported schema type: {type(schema)}")

    return {"type": "json_schema", "schema": json_schema}


class ChatClaudeAgSDK(BaseChatModel):
    """LangChain chat model wrapping the Claude Agent SDK.

    All generation paths use ``query()`` from the SDK.  When tools are
    bound via ``bind_tools()``, they are converted to an in-process MCP
    server and attached to the ``ClaudeAgentOptions`` before calling
    ``query()``.

    Args:
        model: SDK model name (e.g. "sonnet", "opus", "haiku").
        max_turns: Maximum agentic turns.
        max_budget_usd: Budget limit in USD.
        allowed_tools: SDK built-in tools (e.g. ["Read", "Bash"]).
        system_prompt: Default system prompt (overridden by SystemMessage).
        permission_mode: SDK permission mode.
        cwd: Working directory for the SDK.
        thinking: Extended thinking configuration. Use
            ``{"type": "enabled", "budget_tokens": N}`` for explicit budget,
            ``{"type": "adaptive"}`` for adaptive, or ``{"type": "disabled"}``.
        effort: Effort level for the model ("low", "medium", "high", "max").
    """

    model: str = DEFAULT_MODEL
    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    allowed_tools: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    permission_mode: str = DEFAULT_PERMISSION_MODE
    cwd: Optional[str] = None
    thinking: Optional[ThinkingConfig] = None
    effort: Optional[EffortLevel] = None

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "claude-agent-sdk"

    def _build_options(
        self,
        system_prompt: str | None = None,
        *,
        include_partial_messages: bool = False,
        output_format: OutputFormat | None = None,
    ) -> ClaudeAgentOptions:
        """Build a ``ClaudeAgentOptions`` from the agent's configuration.

        Args:
            system_prompt: Explicit system prompt that takes priority over
                ``self.system_prompt``.  When *None*, ``self.system_prompt``
                is used as a fallback.
            include_partial_messages: Whether the SDK should yield partial
                (streaming) messages.
            output_format: Optional structured output format dict.

        Returns:
            A fully-populated ``ClaudeAgentOptions`` instance.
        """
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt
            if system_prompt is not None
            else self.system_prompt,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            allowed_tools=self.allowed_tools or [],
            permission_mode=self.permission_mode,
            cwd=self.cwd,
            include_partial_messages=include_partial_messages,
            thinking=self.thinking,
            effort=self.effort,
            output_format=output_format,
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

    def _attach_tools_to_options(
        self, options: ClaudeAgentOptions, tools: list
    ) -> None:
        """Convert LangChain tools to an MCP server and attach to *options*.

        Args:
            options: The ``ClaudeAgentOptions`` to mutate in-place.
            tools: LangChain tool instances to convert and bind.
        """
        sdk_tool_specs = convert_langchain_tools(tools)

        sdk_tools = []
        for spec in sdk_tool_specs:
            decorated = sdk_tool_decorator(spec.name, spec.description, spec.schema)(
                spec.handler
            )
            sdk_tools.append(decorated)

        mcp_server = create_sdk_mcp_server(
            name=MCP_SERVER_NAME,
            version=MCP_SERVER_VERSION,
            tools=sdk_tools,
        )
        tool_names = [f"{TOOL_NAME_PREFIX}{spec.name}" for spec in sdk_tool_specs]

        options.mcp_servers = {MCP_SERVER_NAME: mcp_server}
        options.allowed_tools = tool_names + (self.allowed_tools or [])

    # --------------------------------------------------------------------- #
    # Structured output
    # --------------------------------------------------------------------- #

    def with_structured_output(
        self,
        schema: Union[dict, Type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Return a Runnable that produces structured output.

        Args:
            schema: A Pydantic model class, JSON schema dict, or TypedDict.
            include_raw: If True, return a dict with "raw", "parsed", and
                "parsing_error" keys.
            **kwargs: Additional arguments.

        Returns:
            A Runnable that parses the model output into the given schema.
        """
        output_format = _schema_to_output_format(schema)
        llm = self.bind(output_format=output_format)

        is_pydantic = hasattr(schema, "model_validate")
        if is_pydantic:
            parser = PydanticOutputParser(pydantic_object=schema)
        else:
            parser = JsonOutputParser()

        if include_raw:
            return RunnablePassthrough.assign(
                parsed=lambda x: x,
                parsing_error=lambda x: None,
            ) | {
                "raw": llm,
                "parsed": llm | parser,
                "parsing_error": lambda x: None,
            }
        return llm | parser

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
        """Generate a response using the SDK ``query()`` async generator.

        Args:
            messages: LangChain messages forming the conversation.
            stop: Optional stop sequences (unused by SDK but part of the API).
            run_manager: LangChain async callback manager.
            **kwargs: Extra keyword arguments.

        Returns:
            A ``ChatResult`` with one ``ChatGeneration`` containing an
            ``AIMessage``.
        """
        tools = kwargs.pop("tools", None)
        output_format = kwargs.pop("output_format", None)
        system_prompt, chat_messages = extract_system_message(messages)
        prompt_text = convert_messages_to_prompt(chat_messages)
        options = self._build_options(system_prompt, output_format=output_format)

        has_mcp = False
        if tools:
            self._attach_tools_to_options(options, tools)
            has_mcp = True

        # MCP tools require streaming input (async generator), not a plain string
        prompt: Any = _streaming_prompt(prompt_text) if has_mcp else prompt_text

        result_text = ""
        thinking_blocks: list[dict] = []
        usage_data: dict = {}
        structured_output = None

        async for message in sdk_query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ThinkingBlock):
                        thinking_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.signature,
                            }
                        )
                    elif hasattr(block, "text"):
                        result_text += block.text
            elif isinstance(message, ResultMessage):
                usage_data = message.usage or {}
                if message.structured_output is not None:
                    structured_output = message.structured_output

        # Use structured output as content if available
        content = (
            json.dumps(structured_output)
            if structured_output is not None
            else result_text
        )

        additional_kwargs: dict[str, Any] = {}
        if thinking_blocks:
            additional_kwargs["thinking"] = thinking_blocks

        ai_msg = AIMessage(
            content=content,
            usage_metadata=map_sdk_usage(usage_data),
            response_metadata={"model": self.model},
            additional_kwargs=additional_kwargs,
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
        return _run_async(self._agenerate(messages, stop, **kwargs))

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

        When thinking is enabled, the SDK may return ``AssistantMessage``
        objects instead of ``StreamEvent``s. In that case, thinking blocks
        are emitted in ``additional_kwargs`` and text blocks as content chunks.

        Args:
            messages: LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager for on_llm_new_token.
            **kwargs: Additional arguments.

        Yields:
            ChatGenerationChunk with AIMessageChunk for each text delta.
        """
        tools = kwargs.pop("tools", None)
        output_format = kwargs.pop("output_format", None)
        system_prompt, chat_messages = extract_system_message(messages)
        prompt_text = convert_messages_to_prompt(chat_messages)
        options = self._build_options(
            system_prompt,
            include_partial_messages=True,
            output_format=output_format,
        )

        has_mcp = False
        if tools:
            self._attach_tools_to_options(options, tools)
            has_mcp = True

        # MCP tools require streaming input (async generator), not a plain string
        prompt: Any = _streaming_prompt(prompt_text) if has_mcp else prompt_text

        async for message in sdk_query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
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
            elif isinstance(message, AssistantMessage):
                # Fallback when SDK returns complete messages (e.g. with thinking)
                thinking_blocks: list[dict] = []
                for block in message.content:
                    if isinstance(block, ThinkingBlock):
                        thinking_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.signature,
                            }
                        )
                    elif hasattr(block, "text"):
                        text = block.text
                        additional_kwargs: dict[str, Any] = {}
                        if thinking_blocks:
                            additional_kwargs["thinking"] = thinking_blocks
                            thinking_blocks = []
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=text,
                                additional_kwargs=additional_kwargs,
                            )
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk
            elif isinstance(message, ResultMessage):
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
        yield from _run_async_iter(self._astream(messages, stop, **kwargs))
