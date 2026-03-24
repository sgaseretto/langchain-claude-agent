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
from claude_agent_sdk.types import HookMatcher, StreamEvent, ToolUseBlock, UserMessage
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable

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
    convert_messages_to_sdk_streaming,
    extract_system_message,
    has_multimodal_content,
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


async def _streaming_messages(
    sdk_messages: list[dict],
) -> AsyncIterator[dict]:
    """Yield pre-built SDK streaming message dicts as an async generator.

    Used for multimodal messages that need the streaming input format.

    Args:
        sdk_messages: A list of SDK streaming message dicts.

    Yields:
        Each SDK message dict.
    """
    for msg in sdk_messages:
        yield msg


async def _delegate_langchain_tool_execution(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Block SDK MCP execution so LangChain can handle the tool call itself.

    Args:
        input_data: Hook payload describing the attempted tool invocation.
        tool_use_id: Tool use identifier from the SDK.
        context: Hook execution context.

    Returns:
        Hook output instructing Claude Code not to run the tool locally.
    """
    del input_data, tool_use_id, context
    return {
        "continue_": False,
        "decision": "block",
        "reason": "Tool execution is delegated to the outer LangChain runtime.",
    }


def _coerce_array_tool_arg(value: Any) -> list[Any]:
    """Coerce a model-produced value into a JSON-schema array.

    This is only used for LangChain's synthetic structured-output tools,
    where the model may emit a string for a field declared as ``list[str]``.

    Args:
        value: Raw tool argument value from the model.

    Returns:
        A best-effort list representation.
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        lowered = stripped.lower()
        if lowered in {"[]", "none", "n/a"}:
            return []
        if "no" in lowered and "alert" in lowered:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        separator = ";" if ";" in stripped else ","
        parts = [part.strip() for part in stripped.split(separator)]
        return [part for part in parts if part]

    return [value]


def _coerce_args_to_json_schema(
    args: dict[str, Any],
    json_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Coerce tool arguments to match a JSON schema more closely.

    Args:
        args: Raw tool arguments from the model.
        json_schema: JSON schema for the target tool, if available.

    Returns:
        Tool arguments after lightweight coercion.
    """
    if not json_schema:
        return args

    properties = json_schema.get("properties", {})
    coerced = dict(args)
    for key, value in args.items():
        field_schema = properties.get(key, {})
        if field_schema.get("type") == "array":
            coerced[key] = _coerce_array_tool_arg(value)
    return coerced


def _build_tool_schema_map(tools: list[Any]) -> dict[str, dict[str, Any]]:
    """Build a lookup of prefixed tool names to JSON schemas.

    Args:
        tools: Active bound tools for the current turn.

    Returns:
        Mapping of SDK-visible tool names to JSON schema dicts.
    """
    schema_map: dict[str, dict[str, Any]] = {}
    for tool in tools:
        args_schema = getattr(tool, "args_schema", None)
        if not isinstance(args_schema, dict):
            continue
        prefixed_name = (
            tool.name
            if tool.name.startswith(TOOL_NAME_PREFIX)
            else f"{TOOL_NAME_PREFIX}{tool.name}"
        )
        schema_map[prefixed_name] = args_schema
    return schema_map


def _langchain_tool_call_from_sdk_block(
    block: ToolUseBlock,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an SDK ToolUseBlock to a LangChain tool call.

    Args:
        block: The SDK tool-use block emitted by Claude Code.
        json_schema: Optional JSON schema for the target tool.

    Returns:
        A LangChain-compatible tool call dictionary.
    """
    name = block.name
    if name.startswith(TOOL_NAME_PREFIX):
        name = name[len(TOOL_NAME_PREFIX) :]

    return {
        "name": name,
        "args": _coerce_args_to_json_schema(block.input, json_schema),
        "id": block.id,
        "type": "tool_call",
    }


def _select_bound_tool_names(
    tool_names: list[str],
    tool_choice: str | dict[str, Any] | None,
) -> list[str]:
    """Resolve LangChain tool_choice values to SDK-allowed tool names.

    Args:
        tool_names: Prefixed SDK tool names.
        tool_choice: LangChain tool-choice value.

    Returns:
        The subset of tool names that should remain available to Claude.

    Raises:
        ValueError: If the requested tool choice references an unknown tool.
    """
    if tool_choice in (None, "auto", "any", "required"):
        return tool_names

    if tool_choice == "none":
        return []

    chosen_name: str | None = None
    if isinstance(tool_choice, str):
        chosen_name = tool_choice
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        chosen_name = tool_choice.get("function", {}).get("name")

    if not chosen_name:
        raise ValueError(f"Unsupported tool_choice: {tool_choice!r}")

    prefixed_name = (
        chosen_name
        if chosen_name.startswith(TOOL_NAME_PREFIX)
        else f"{TOOL_NAME_PREFIX}{chosen_name}"
    )
    if prefixed_name not in tool_names:
        raise ValueError(f"Unknown tool_choice: {chosen_name}")

    return [prefixed_name]


def _is_langchain_response_format_tool(tool: Any) -> bool:
    """Return True for LangChain ToolStrategy's synthetic schema tool.

    LangChain currently represents the structured-output tool injected by
    ``create_agent(..., response_format=ToolStrategy(...))`` as a tool whose
    ``args_schema`` is already a JSON schema dictionary.

    Args:
        tool: Tool-like object supplied via ``bind_tools()``.

    Returns:
        True when the tool looks like LangChain's synthetic structured-output
        tool, False otherwise.
    """
    args_schema = getattr(tool, "args_schema", None)
    return isinstance(args_schema, dict) and isinstance(
        args_schema.get("properties"), dict
    )


def _select_tools_for_turn(
    tools: list[Any],
    messages: list[BaseMessage] | None = None,
) -> list[Any]:
    """Select the bound tools that should be visible for the current turn.

    For LangChain ``ToolStrategy`` flows, the synthetic schema tool should only
    be exposed after tool results are already present in the conversation. This
    keeps the SDK from eagerly emitting the schema tool call before real tool
    outputs have been returned to the outer LangChain loop.

    Args:
        tools: The full list of bound LangChain tools.
        messages: Conversation messages for the current turn.

    Returns:
        The subset of tools that should be exposed to the SDK on this turn.
    """
    structured_tools = [tool for tool in tools if _is_langchain_response_format_tool(tool)]
    if not structured_tools:
        return tools

    regular_tools = [tool for tool in tools if not _is_langchain_response_format_tool(tool)]
    has_tool_results = any(isinstance(message, ToolMessage) for message in messages or [])

    if has_tool_results:
        return structured_tools if regular_tools else tools

    return regular_tools or tools


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
        stderr_lines: list[str] | None = None,
    ) -> ClaudeAgentOptions:
        """Build a ``ClaudeAgentOptions`` from the agent's configuration.

        Args:
            system_prompt: Explicit system prompt that takes priority over
                ``self.system_prompt``.  When *None*, ``self.system_prompt``
                is used as a fallback.
            include_partial_messages: Whether the SDK should yield partial
                (streaming) messages.
            output_format: Optional structured output format dict.
            stderr_lines: Optional list that will collect stderr output from
                the CLI process for debugging purposes.

        Returns:
            A fully-populated ``ClaudeAgentOptions`` instance.
        """
        stderr_callback = None
        if stderr_lines is not None:

            def _collect_stderr(line: str) -> None:
                stderr_lines.append(line)

            stderr_callback = _collect_stderr

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
            stderr=stderr_callback,
        )

    # --------------------------------------------------------------------- #
    # Tool binding
    # --------------------------------------------------------------------- #

    def bind_tools(
        self,
        tools: list,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind LangChain tools for provider-style tool calling.

        Args:
            tools: List of LangChain tool instances.
            tool_choice: Optional LangChain tool-choice hint.
            **kwargs: Additional arguments passed to bind().

        Returns:
            A Runnable with tools stored in kwargs.
        """
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)

    def _attach_tools_to_options(
        self,
        options: ClaudeAgentOptions,
        tools: list,
        tool_choice: str | dict[str, Any] | None = None,
        messages: list[BaseMessage] | None = None,
    ) -> list[str]:
        """Convert LangChain tools to an MCP server and attach to *options*.

        Args:
            options: The ``ClaudeAgentOptions`` to mutate in-place.
            tools: LangChain tool instances to convert and bind.
            tool_choice: Optional LangChain tool-choice hint.
            messages: Conversation messages for the current turn.

        Returns:
            The prefixed tool names exposed to Claude for this invocation.
        """
        active_tools = _select_tools_for_turn(tools, messages)
        sdk_tool_specs = convert_langchain_tools(active_tools)

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
        selected_tool_names = _select_bound_tool_names(tool_names, tool_choice)

        if not selected_tool_names:
            options.allowed_tools = self.allowed_tools or []
            return []

        options.mcp_servers = {MCP_SERVER_NAME: mcp_server}
        options.allowed_tools = selected_tool_names + (self.allowed_tools or [])
        options.hooks = {
            "PreToolUse": [
                HookMatcher(
                    matcher=tool_name,
                    hooks=[_delegate_langchain_tool_execution],
                )
                for tool_name in selected_tool_names
            ]
        }
        return selected_tool_names

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

            def _parse_with_raw(input: Any) -> dict:
                """Invoke the LLM, parse, and return raw + parsed + error."""
                raw = llm.invoke(input)
                try:
                    parsed = parser.invoke(raw)
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                except Exception as e:
                    return {"raw": raw, "parsed": None, "parsing_error": e}

            from langchain_core.runnables import RunnableLambda

            return RunnableLambda(_parse_with_raw)
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
        tool_choice = kwargs.pop("tool_choice", None)
        output_format = kwargs.pop("output_format", None)
        system_prompt, chat_messages = extract_system_message(messages)

        stderr_lines: list[str] = []
        options = self._build_options(
            system_prompt, output_format=output_format, stderr_lines=stderr_lines
        )

        use_streaming = False
        selected_bound_tool_names: set[str] = set()
        tool_schema_map: dict[str, dict[str, Any]] = {}
        if tools:
            tool_schema_map = _build_tool_schema_map(
                _select_tools_for_turn(tools, chat_messages)
            )
            selected_bound_tool_names = set(
                self._attach_tools_to_options(options, tools, tool_choice, chat_messages)
            )
            use_streaming = True

        multimodal = has_multimodal_content(chat_messages)
        if multimodal:
            use_streaming = True

        # Build prompt: streaming mode for MCP tools or multimodal content.
        if multimodal:
            sdk_msgs = convert_messages_to_sdk_streaming(chat_messages)
            prompt: Any = _streaming_messages(sdk_msgs)
        elif use_streaming:
            prompt_text = convert_messages_to_prompt(chat_messages)
            prompt = _streaming_prompt(prompt_text)
        else:
            prompt = convert_messages_to_prompt(chat_messages)

        result_text = ""
        thinking_blocks: list[dict] = []
        tool_calls: list[dict[str, Any]] = []
        ignore_assistant_text = False
        usage_data: dict = {}
        structured_output = None

        try:
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
                        elif (
                            isinstance(block, ToolUseBlock)
                            and block.name in selected_bound_tool_names
                        ):
                            tool_calls.append(
                                _langchain_tool_call_from_sdk_block(
                                    block,
                                    tool_schema_map.get(block.name),
                                )
                            )
                            ignore_assistant_text = True
                        elif (
                            isinstance(block, ToolUseBlock)
                            and output_format is not None
                            and block.name == "StructuredOutput"
                        ):
                            structured_output = block.input
                            ignore_assistant_text = True
                        elif hasattr(block, "text"):
                            if not ignore_assistant_text:
                                result_text += block.text
                elif isinstance(message, UserMessage) and tool_calls:
                    ignore_assistant_text = True
                elif isinstance(message, ResultMessage):
                    usage_data = message.usage or {}
                    if message.structured_output is not None and not tool_calls:
                        structured_output = message.structured_output
        except Exception as e:
            if stderr_lines:
                stderr_detail = "\n".join(stderr_lines)
                raise type(e)(f"{e}\nCLI stderr:\n{stderr_detail}") from e
            raise

        # Use structured output as content if available.
        content = (
            json.dumps(structured_output)
            if structured_output is not None and not tool_calls
            else result_text
        )

        additional_kwargs: dict[str, Any] = {}
        if thinking_blocks:
            additional_kwargs["thinking"] = thinking_blocks

        ai_msg = AIMessage(
            content=content,
            tool_calls=tool_calls,
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
        tool_choice = kwargs.pop("tool_choice", None)
        output_format = kwargs.pop("output_format", None)
        system_prompt, chat_messages = extract_system_message(messages)

        stderr_lines: list[str] = []
        options = self._build_options(
            system_prompt,
            include_partial_messages=True,
            output_format=output_format,
            stderr_lines=stderr_lines,
        )

        use_streaming = False
        selected_bound_tool_names: set[str] = set()
        tool_schema_map: dict[str, dict[str, Any]] = {}
        if tools:
            tool_schema_map = _build_tool_schema_map(
                _select_tools_for_turn(tools, chat_messages)
            )
            selected_bound_tool_names = set(
                self._attach_tools_to_options(options, tools, tool_choice, chat_messages)
            )
            use_streaming = True

        multimodal = has_multimodal_content(chat_messages)
        if multimodal:
            use_streaming = True

        # Build prompt: streaming mode for MCP tools or multimodal content.
        if multimodal:
            sdk_msgs = convert_messages_to_sdk_streaming(chat_messages)
            prompt: Any = _streaming_messages(sdk_msgs)
        elif use_streaming:
            prompt_text = convert_messages_to_prompt(chat_messages)
            prompt = _streaming_prompt(prompt_text)
        else:
            prompt = convert_messages_to_prompt(chat_messages)

        msg_source = sdk_query(prompt=prompt, options=options)
        ignore_assistant_text = False

        async for message in msg_source:
            if isinstance(message, StreamEvent):
                if ignore_assistant_text:
                    continue
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
                    elif (
                        isinstance(block, ToolUseBlock)
                        and block.name in selected_bound_tool_names
                    ):
                        ignore_assistant_text = True
                        tool_call = _langchain_tool_call_from_sdk_block(
                            block,
                            tool_schema_map.get(block.name),
                        )
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    {
                                        "name": tool_call["name"],
                                        "args": json.dumps(tool_call["args"]),
                                        "id": block.id,
                                        "index": 0,
                                        "type": "tool_call_chunk",
                                    }
                                ],
                                chunk_position="last",
                            )
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token("", chunk=chunk)
                        yield chunk
                    elif (
                        isinstance(block, ToolUseBlock)
                        and output_format is not None
                        and block.name == "StructuredOutput"
                    ):
                        ignore_assistant_text = True
                    elif hasattr(block, "text"):
                        if ignore_assistant_text:
                            continue
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
            elif isinstance(message, UserMessage):
                if ignore_assistant_text:
                    continue
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
