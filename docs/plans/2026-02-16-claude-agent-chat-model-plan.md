# ChatClaudeAgent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a LangChain `BaseChatModel` that wraps the Claude Agent SDK, with streaming, tool calling, and cost tracking.

**Architecture:** Hybrid approach - `query()` for basic text generation, `ClaudeSDKClient` when tools are bound via `bind_tools()`. The SDK executes tools autonomously. Uses fastcore utilities where applicable.

**Tech Stack:** Python 3.10+, langchain-core, claude-agent-sdk, fastcore, pydantic, ruff, pytest, pytest-asyncio, uv

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `langchain_claude_agent/__init__.py`
- Create: `langchain_claude_agent/_types.py`
- Create: `tests/__init__.py`

**Step 1: Create `pyproject.toml`**

```toml
[project]
name = "langchain-claude-agent"
version = "0.1.0"
description = "Use the Claude Agent SDK as an LLM provider in LangChain/LangGraph"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [
    { name = "sgaseretto" }
]
dependencies = [
    "langchain-core>=0.3.0",
    "claude-agent-sdk",
    "fastcore",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "ruff",
    "pytest",
    "pytest-asyncio",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "B", "I"]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create `.python-version`**

```
3.10
```

**Step 3: Create `langchain_claude_agent/__init__.py`**

```python
"""LangChain chat model wrapping the Claude Agent SDK."""

from langchain_claude_agent.chat_model import ChatClaudeAgent

__all__ = ["ChatClaudeAgent"]
```

Note: This will initially fail to import since `chat_model.py` doesn't exist yet. That's fine - we'll create it in Task 3.

**Step 4: Create `langchain_claude_agent/_types.py`**

```python
"""Type aliases and constants for langchain-claude-agent."""

from __future__ import annotations

# SDK model names
DEFAULT_MODEL = "sonnet"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
MCP_SERVER_NAME = "langchain-tools"
MCP_SERVER_VERSION = "1.0.0"

# Tool name prefix for MCP tools
TOOL_NAME_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
```

**Step 5: Create `tests/__init__.py`**

Empty file.

**Step 6: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed, `uv.lock` created.

**Step 7: Commit**

```bash
git add pyproject.toml .python-version langchain_claude_agent/__init__.py langchain_claude_agent/_types.py tests/__init__.py uv.lock
git commit -m "scaffold project with pyproject.toml, package structure, and dependencies"
```

---

### Task 2: Utils Module - Message Conversion & Usage Mapping

**Files:**
- Create: `tests/test_utils.py`
- Create: `langchain_claude_agent/_utils.py`

**Step 1: Write failing tests for message conversion**

```python
"""Tests for message conversion and usage mapping utilities."""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class TestExtractSystemMessage:
    """Tests for extract_system_message."""

    def test_no_system_message(self):
        from langchain_claude_agent._utils import extract_system_message

        messages = [HumanMessage(content="hello")]
        system, remaining = extract_system_message(messages)
        assert system is None
        assert len(remaining) == 1
        assert remaining[0].content == "hello"

    def test_single_system_message(self):
        from langchain_claude_agent._utils import extract_system_message

        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="hello"),
        ]
        system, remaining = extract_system_message(messages)
        assert system == "You are helpful"
        assert len(remaining) == 1

    def test_multiple_system_messages_concatenated(self):
        from langchain_claude_agent._utils import extract_system_message

        messages = [
            SystemMessage(content="Be helpful."),
            SystemMessage(content="Be concise."),
            HumanMessage(content="hello"),
        ]
        system, remaining = extract_system_message(messages)
        assert system == "Be helpful.\nBe concise."
        assert len(remaining) == 1

    def test_system_message_not_at_start(self):
        from langchain_claude_agent._utils import extract_system_message

        messages = [
            HumanMessage(content="hello"),
            SystemMessage(content="You are helpful"),
        ]
        system, remaining = extract_system_message(messages)
        assert system == "You are helpful"
        assert len(remaining) == 1


class TestConvertMessages:
    """Tests for convert_messages_to_prompt."""

    def test_single_human_message(self):
        from langchain_claude_agent._utils import convert_messages_to_prompt

        messages = [HumanMessage(content="What is 2+2?")]
        prompt = convert_messages_to_prompt(messages)
        assert prompt == "Human: What is 2+2?"

    def test_human_ai_conversation(self):
        from langchain_claude_agent._utils import convert_messages_to_prompt

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "Human: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "Human: How are you?" in prompt

    def test_tool_message(self):
        from langchain_claude_agent._utils import convert_messages_to_prompt

        messages = [
            ToolMessage(content="72 degrees", tool_call_id="call_1", name="weather"),
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "Tool Result (weather): 72 degrees" in prompt

    def test_tool_message_without_name(self):
        from langchain_claude_agent._utils import convert_messages_to_prompt

        messages = [
            ToolMessage(content="result", tool_call_id="call_1"),
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "Tool Result: result" in prompt

    def test_empty_messages(self):
        from langchain_claude_agent._utils import convert_messages_to_prompt

        prompt = convert_messages_to_prompt([])
        assert prompt == ""

    def test_system_messages_skipped(self):
        from langchain_claude_agent._utils import convert_messages_to_prompt

        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "system" not in prompt
        assert "Human: hello" in prompt


class TestMapUsage:
    """Tests for map_sdk_usage."""

    def test_empty_usage(self):
        from langchain_claude_agent._utils import map_sdk_usage

        result = map_sdk_usage({})
        assert result == {}

    def test_none_usage(self):
        from langchain_claude_agent._utils import map_sdk_usage

        result = map_sdk_usage(None)
        assert result == {}

    def test_basic_usage(self):
        from langchain_claude_agent._utils import map_sdk_usage

        usage = {"input_tokens": 100, "output_tokens": 50}
        result = map_sdk_usage(usage)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_cache_usage(self):
        from langchain_claude_agent._utils import map_sdk_usage

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        }
        result = map_sdk_usage(usage)
        assert result["input_token_details"]["cache_read"] == 80
        assert result["input_token_details"]["cache_creation"] == 20
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL - `langchain_claude_agent._utils` does not exist.

**Step 3: Implement `_utils.py`**

```python
"""Message conversion, usage mapping, and credential checking utilities."""

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
    """Extract system messages from a message list.

    Args:
        messages: List of LangChain messages.

    Returns:
        Tuple of (concatenated system prompt or None, remaining messages).
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
    """Convert LangChain messages to a formatted prompt string.

    SystemMessages are skipped (should be extracted separately).
    HumanMessage → "Human: {content}"
    AIMessage → "Assistant: {content}"
    ToolMessage → "Tool Result ({name}): {content}" or "Tool Result: {content}"

    Args:
        messages: List of LangChain messages (excluding system messages).

    Returns:
        Formatted prompt string.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            if name:
                parts.append(f"Tool Result ({name}): {msg.content}")
            else:
                parts.append(f"Tool Result: {msg.content}")
        else:
            parts.append(f"{msg.type}: {msg.content}")
    return "\n".join(parts)


def map_sdk_usage(sdk_usage: dict | None) -> dict:
    """Map Claude Agent SDK usage data to LangChain UsageMetadata format.

    Args:
        sdk_usage: Usage dict from SDK ResultMessage, or None.

    Returns:
        Dict compatible with LangChain's usage_metadata field.
    """
    if not sdk_usage:
        return {}
    input_tokens = sdk_usage.get("input_tokens", 0)
    output_tokens = sdk_usage.get("output_tokens", 0)
    result = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    cache_read = sdk_usage.get("cache_read_input_tokens", 0)
    cache_creation = sdk_usage.get("cache_creation_input_tokens", 0)
    if cache_read or cache_creation:
        result["input_token_details"] = {
            "cache_read": cache_read,
            "cache_creation": cache_creation,
        }
    return result
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: All tests PASS.

**Step 5: Run linter**

Run: `uv run ruff check langchain_claude_agent/_utils.py tests/test_utils.py`
Expected: No errors.

**Step 6: Commit**

```bash
git add langchain_claude_agent/_utils.py tests/test_utils.py
git commit -m "add message conversion and usage mapping utilities with tests"
```

---

### Task 3: Tool Converter Module

**Files:**
- Create: `tests/test_tool_converter.py`
- Create: `langchain_claude_agent/_tool_converter.py`

**Step 1: Write failing tests for tool conversion**

```python
"""Tests for LangChain tool to SDK MCP tool conversion."""

import pytest
from langchain_core.tools import tool as lc_tool


@lc_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@lc_tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


class TestGetToolSchema:
    """Tests for extracting JSON schema from LangChain tools."""

    def test_schema_from_tool_with_args(self):
        from langchain_claude_agent._tool_converter import get_tool_schema

        schema = get_tool_schema(add)
        assert "a" in schema
        assert "b" in schema
        assert schema["a"] is int or schema["a"] == int

    def test_schema_from_tool_with_string_arg(self):
        from langchain_claude_agent._tool_converter import get_tool_schema

        schema = get_tool_schema(greet)
        assert "name" in schema
        assert schema["name"] is str or schema["name"] == str


class TestConvertTool:
    """Tests for converting LangChain tools to SDK tools."""

    def test_convert_preserves_name(self):
        from langchain_claude_agent._tool_converter import convert_langchain_tool_to_sdk

        sdk_tool = convert_langchain_tool_to_sdk(add)
        assert sdk_tool.name == "add"

    def test_convert_preserves_description(self):
        from langchain_claude_agent._tool_converter import convert_langchain_tool_to_sdk

        sdk_tool = convert_langchain_tool_to_sdk(add)
        assert sdk_tool.description == "Add two integers."

    @pytest.mark.asyncio
    async def test_converted_tool_executes(self):
        from langchain_claude_agent._tool_converter import convert_langchain_tool_to_sdk

        sdk_tool = convert_langchain_tool_to_sdk(add)
        result = await sdk_tool.handler({"a": 3, "b": 4})
        assert result["content"][0]["type"] == "text"
        assert "7" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_converted_string_tool_executes(self):
        from langchain_claude_agent._tool_converter import convert_langchain_tool_to_sdk

        sdk_tool = convert_langchain_tool_to_sdk(greet)
        result = await sdk_tool.handler({"name": "Alice"})
        assert "Hello, Alice!" in result["content"][0]["text"]


class TestConvertToolList:
    """Tests for batch tool conversion."""

    def test_convert_multiple_tools(self):
        from langchain_claude_agent._tool_converter import convert_langchain_tools

        sdk_tools = convert_langchain_tools([add, greet])
        assert len(sdk_tools) == 2
        names = {t.name for t in sdk_tools}
        assert names == {"add", "greet"}
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tool_converter.py -v`
Expected: FAIL - module does not exist.

**Step 3: Implement `_tool_converter.py`**

Note: The SDK's `@tool` decorator creates tool objects with `name`, `description`, `schema`, and `handler` attributes. We need to check the actual SDK tool structure at implementation time and adapt. The code below is our best understanding from the docs - the implementer should verify the `@tool` return type and adjust the `SDKTool` dataclass and handler access accordingly.

```python
"""Convert LangChain tools to Claude Agent SDK MCP tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from langchain_core.tools import BaseTool

from claude_agent_sdk import tool as sdk_tool


@dataclass
class SDKTool:
    """Wrapper holding SDK tool metadata and handler."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def get_tool_schema(lc_tool: BaseTool) -> dict[str, Any]:
    """Extract a simple type-mapping schema from a LangChain tool.

    Args:
        lc_tool: A LangChain BaseTool instance.

    Returns:
        Dict mapping argument names to Python types (for SDK @tool simple schema).
    """
    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema is not None:
        json_schema = lc_tool.args_schema.model_json_schema()
        props = json_schema.get("properties", {})
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }
        return {
            name: type_map.get(info.get("type", "string"), str)
            for name, info in props.items()
        }
    return {}


def convert_langchain_tool_to_sdk(lc_tool: BaseTool) -> SDKTool:
    """Convert a single LangChain tool to an SDK MCP tool.

    Args:
        lc_tool: A LangChain BaseTool instance.

    Returns:
        An SDKTool with name, description, schema, and async handler.
    """
    schema = get_tool_schema(lc_tool)

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await lc_tool.ainvoke(args)
        except NotImplementedError:
            result = lc_tool.invoke(args)
        return {"content": [{"type": "text", "text": str(result)}]}

    return SDKTool(
        name=lc_tool.name,
        description=lc_tool.description,
        schema=schema,
        handler=handler,
    )


def convert_langchain_tools(tools: list[BaseTool]) -> list[SDKTool]:
    """Convert a list of LangChain tools to SDK MCP tools.

    Args:
        tools: List of LangChain BaseTool instances.

    Returns:
        List of SDKTool instances.
    """
    return [convert_langchain_tool_to_sdk(t) for t in tools]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_converter.py -v`
Expected: All tests PASS.

**Step 5: Lint**

Run: `uv run ruff check langchain_claude_agent/_tool_converter.py tests/test_tool_converter.py`
Expected: No errors.

**Step 6: Commit**

```bash
git add langchain_claude_agent/_tool_converter.py tests/test_tool_converter.py
git commit -m "add tool converter for LangChain tools to SDK MCP tools with tests"
```

---

### Task 4: Credential Checking

**Files:**
- Add to: `langchain_claude_agent/_utils.py`
- Add to: `tests/test_utils.py`

**Step 1: Write failing test for credential checking**

Add to `tests/test_utils.py`:

```python
import os
from unittest.mock import patch


class TestCheckCredentials:
    """Tests for check_claude_agent_sdk_credentials."""

    def test_api_key_present(self):
        from langchain_claude_agent._utils import check_claude_agent_sdk_credentials

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            ok, msg = check_claude_agent_sdk_credentials()
            assert ok is True
            assert "ANTHROPIC_API_KEY" in msg

    def test_no_credentials(self):
        from langchain_claude_agent._utils import check_claude_agent_sdk_credentials

        with patch.dict(os.environ, {}, clear=True):
            # Remove all known credential env vars
            env = {
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    "ANTHROPIC_API_KEY",
                    "CLAUDE_CODE_USE_BEDROCK",
                    "CLAUDE_CODE_USE_VERTEX",
                    "CLAUDE_CODE_USE_FOUNDRY",
                }
            }
            with patch.dict(os.environ, env, clear=True):
                ok, msg = check_claude_agent_sdk_credentials()
                # Either finds SDK credentials or returns False
                assert isinstance(ok, bool)
                assert isinstance(msg, str)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py::TestCheckCredentials -v`
Expected: FAIL - function does not exist.

**Step 3: Add credential checking to `_utils.py`**

Append to `langchain_claude_agent/_utils.py`:

```python
import os


def check_claude_agent_sdk_credentials() -> tuple[bool, str]:
    """Check if Claude Agent SDK credentials are available.

    Checks in order:
    1. ANTHROPIC_API_KEY environment variable
    2. Cloud provider env vars (Bedrock, Vertex, Foundry)
    3. SDK probe (Claude Code subscription)

    Returns:
        Tuple of (success: bool, message: str).
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
        from claude_agent_sdk import query, ClaudeAgentOptions

        async def _probe():
            options = ClaudeAgentOptions(model="sonnet", max_turns=0)
            try:
                async for _ in query(prompt="test", options=options):
                    pass
            except Exception as e:
                err_str = str(e).lower()
                if "max_turns" in err_str or "turn" in err_str:
                    return True, "claude_agent_sdk: auth verified (max_turns reached)"
                if "login" in err_str or "auth" in err_str or "credential" in err_str:
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add langchain_claude_agent/_utils.py tests/test_utils.py
git commit -m "add credential checking for API key, cloud providers, and SDK probe"
```

---

### Task 5: ChatClaudeAgent Core - Basic invoke (query path)

**Files:**
- Create: `tests/test_chat_model.py`
- Create: `langchain_claude_agent/chat_model.py`
- Modify: `langchain_claude_agent/__init__.py`

**Step 1: Write failing test for basic invocation**

```python
"""Tests for ChatClaudeAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _make_assistant_message(text):
    """Create a mock SDK AssistantMessage."""
    block = MagicMock()
    block.text = text
    type(block).__name__ = "TextBlock"
    # hasattr(block, "text") → True
    msg = MagicMock()
    msg.content = [block]
    msg.__class__.__name__ = "AssistantMessage"
    return msg


def _make_result_message(usage=None, total_cost=None):
    """Create a mock SDK ResultMessage."""
    msg = MagicMock()
    msg.usage = usage or {"input_tokens": 10, "output_tokens": 5}
    msg.total_cost_usd = total_cost or 0.001
    msg.subtype = "success"
    msg.__class__.__name__ = "ResultMessage"
    return msg


class TestChatClaudeAgentInit:
    """Tests for model initialization."""

    def test_default_model(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        assert llm.model == "sonnet"

    def test_custom_model(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent(model="opus")
        assert llm.model == "opus"

    def test_llm_type(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        assert llm._llm_type == "claude-agent-sdk"


class TestChatClaudeAgentGenerate:
    """Tests for _generate and _agenerate via mocked SDK."""

    @pytest.mark.asyncio
    async def test_agenerate_basic(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="What is 2+2?")]

        # Mock the SDK query function
        async def mock_query(*args, **kwargs):
            yield _make_assistant_message("4")
            yield _make_result_message()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            result = await llm._agenerate(messages)

        assert len(result.generations) == 1
        ai_msg = result.generations[0].message
        assert isinstance(ai_msg, AIMessage)
        assert "4" in ai_msg.content

    @pytest.mark.asyncio
    async def test_agenerate_with_system_message(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [
            SystemMessage(content="You are a math tutor"),
            HumanMessage(content="What is 2+2?"),
        ]

        captured_options = {}

        async def mock_query(*args, **kwargs):
            captured_options.update(kwargs)
            yield _make_assistant_message("4")
            yield _make_result_message()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            result = await llm._agenerate(messages)

        # Verify system prompt was passed to SDK
        options = captured_options.get("options")
        assert options is not None
        assert options.system_prompt == "You are a math tutor"

    @pytest.mark.asyncio
    async def test_agenerate_returns_usage_metadata(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="hello")]

        async def mock_query(*args, **kwargs):
            yield _make_assistant_message("hi")
            yield _make_result_message(
                usage={"input_tokens": 100, "output_tokens": 50}
            )

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            result = await llm._agenerate(messages)

        ai_msg = result.generations[0].message
        assert ai_msg.usage_metadata["input_tokens"] == 100
        assert ai_msg.usage_metadata["output_tokens"] == 50
        assert ai_msg.usage_metadata["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_agenerate_fallback_system_prompt(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent(system_prompt="Default system prompt")
        messages = [HumanMessage(content="hello")]

        captured_options = {}

        async def mock_query(*args, **kwargs):
            captured_options.update(kwargs)
            yield _make_assistant_message("hi")
            yield _make_result_message()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            await llm._agenerate(messages)

        options = captured_options.get("options")
        assert options.system_prompt == "Default system prompt"

    @pytest.mark.asyncio
    async def test_agenerate_system_message_overrides_default(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent(system_prompt="Default")
        messages = [
            SystemMessage(content="Override"),
            HumanMessage(content="hello"),
        ]

        captured_options = {}

        async def mock_query(*args, **kwargs):
            captured_options.update(kwargs)
            yield _make_assistant_message("hi")
            yield _make_result_message()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            await llm._agenerate(messages)

        options = captured_options.get("options")
        assert options.system_prompt == "Override"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_model.py -v`
Expected: FAIL - `chat_model` module does not exist.

**Step 3: Implement `chat_model.py` (core, no streaming/tools yet)**

```python
"""ChatClaudeAgent - LangChain BaseChatModel wrapping the Claude Agent SDK."""

from __future__ import annotations

import asyncio
from typing import Any, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from claude_agent_sdk import query as sdk_query
from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage

from langchain_claude_agent._types import DEFAULT_MODEL, DEFAULT_PERMISSION_MODE
from langchain_claude_agent._utils import (
    convert_messages_to_prompt,
    extract_system_message,
    map_sdk_usage,
)


class ChatClaudeAgent(BaseChatModel):
    """LangChain chat model wrapping the Claude Agent SDK.

    Uses query() for basic text generation and ClaudeSDKClient
    when tools are bound via bind_tools().

    Args:
        model: SDK model name (e.g. "sonnet", "opus", "haiku").
        max_turns: Maximum agentic turns. Defaults to None.
        max_budget_usd: Budget limit in USD. Defaults to None.
        allowed_tools: SDK built-in tools (e.g. ["Read", "Bash"]).
        system_prompt: Default system prompt (overridden by SystemMessage).
        permission_mode: SDK permission mode. Defaults to "bypassPermissions".
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
        """Return type of chat model."""
        return "claude-agent-sdk"

    def _build_options(
        self,
        system_prompt: str | None = None,
        *,
        include_partial_messages: bool = False,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from model config.

        Args:
            system_prompt: System prompt to use (from message extraction).
            include_partial_messages: Enable streaming events.

        Returns:
            Configured ClaudeAgentOptions.
        """
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt or self.system_prompt,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            allowed_tools=self.allowed_tools or [],
            permission_mode=self.permission_mode,
            cwd=self.cwd,
            include_partial_messages=include_partial_messages,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion asynchronously.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences (not used by SDK).
            run_manager: Callback manager.
            **kwargs: Additional arguments (tools checked for routing).

        Returns:
            ChatResult with AIMessage.
        """
        tools = kwargs.get("tools")
        if tools:
            return await self._agenerate_with_client(
                messages, tools, stop, run_manager, **kwargs
            )
        return await self._agenerate_with_query(
            messages, stop, run_manager, **kwargs
        )

    async def _agenerate_with_query(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate using SDK query() - no tools path.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager.

        Returns:
            ChatResult with AIMessage.
        """
        system_prompt, chat_messages = extract_system_message(messages)
        prompt = convert_messages_to_prompt(chat_messages)
        options = self._build_options(system_prompt)

        result_text = ""
        usage_data: dict = {}

        async for message in sdk_query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text
            elif isinstance(message, ResultMessage):
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
        """Generate using ClaudeSDKClient - tools path.

        Implemented in Task 6.
        """
        raise NotImplementedError("Tool calling not yet implemented")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion synchronously.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager.

        Returns:
            ChatResult with AIMessage.
        """
        return asyncio.run(self._agenerate(messages, stop, **kwargs))
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_model.py -v`
Expected: All tests PASS.

**Step 5: Lint**

Run: `uv run ruff check langchain_claude_agent/chat_model.py tests/test_chat_model.py`
Expected: No errors.

**Step 6: Commit**

```bash
git add langchain_claude_agent/chat_model.py langchain_claude_agent/__init__.py tests/test_chat_model.py
git commit -m "add ChatClaudeAgent core with basic async invoke via query()"
```

---

### Task 6: Tool Calling via ClaudeSDKClient

**Files:**
- Modify: `langchain_claude_agent/chat_model.py`
- Add to: `tests/test_chat_model.py`

**Step 1: Write failing test for bind_tools and tool invocation**

Add to `tests/test_chat_model.py`:

```python
from langchain_core.tools import tool as lc_tool


@lc_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class TestChatClaudeAgentTools:
    """Tests for bind_tools and tool calling."""

    def test_bind_tools_returns_runnable(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        bound = llm.bind_tools([add])
        # bind_tools returns a RunnableBinding
        assert hasattr(bound, "invoke")
        assert hasattr(bound, "ainvoke")

    @pytest.mark.asyncio
    async def test_agenerate_with_tools_uses_client(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="What is 3 + 4?")]

        # Mock ClaudeSDKClient
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
                "langchain_claude_agent.chat_model.create_sdk_mcp_server",
                return_value=MagicMock(),
            ):
                result = await llm._agenerate(
                    messages, tools=[add]
                )

        assert "7" in result.generations[0].message.content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_model.py::TestChatClaudeAgentTools -v`
Expected: FAIL - `_agenerate_with_client` raises `NotImplementedError`.

**Step 3: Implement bind_tools and _agenerate_with_client**

Add these imports to `chat_model.py`:

```python
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from claude_agent_sdk import ClaudeSDKClient, create_sdk_mcp_server
from langchain_claude_agent._tool_converter import convert_langchain_tools
from langchain_claude_agent._types import MCP_SERVER_NAME, MCP_SERVER_VERSION, TOOL_NAME_PREFIX
```

Add `bind_tools` method to `ChatClaudeAgent`:

```python
    def bind_tools(
        self,
        tools: list[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind LangChain tools to the model for SDK autonomous execution.

        Args:
            tools: List of LangChain tool instances.
            tool_choice: Not used (SDK handles tool selection).
            **kwargs: Additional arguments passed to bind().

        Returns:
            A Runnable with tools stored in kwargs.
        """
        return self.bind(tools=tools, **kwargs)
```

Replace the `_agenerate_with_client` stub:

```python
    async def _agenerate_with_client(
        self,
        messages: list[BaseMessage],
        tools: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate using ClaudeSDKClient with bound tools.

        Args:
            messages: List of LangChain messages.
            tools: List of LangChain tools to convert to SDK MCP tools.
            stop: Stop sequences.
            run_manager: Callback manager.

        Returns:
            ChatResult with AIMessage.
        """
        system_prompt, chat_messages = extract_system_message(messages)
        prompt = convert_messages_to_prompt(chat_messages)

        sdk_tools = convert_langchain_tools(tools)
        sdk_tool_objects = [
            sdk_tool_fn(t.name, t.description, t.schema)(t.handler)
            for t in sdk_tools
        ]
        mcp_server = create_sdk_mcp_server(
            name=MCP_SERVER_NAME,
            version=MCP_SERVER_VERSION,
            tools=sdk_tool_objects,
        )
        tool_names = [f"{TOOL_NAME_PREFIX}{t.name}" for t in sdk_tools]

        options = self._build_options(system_prompt)
        options.mcp_servers = {MCP_SERVER_NAME: mcp_server}
        options.allowed_tools = tool_names + (self.allowed_tools or [])

        result_text = ""
        usage_data: dict = {}

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += block.text
                elif isinstance(message, ResultMessage):
                    usage_data = message.usage or {}

        ai_msg = AIMessage(
            content=result_text,
            usage_metadata=map_sdk_usage(usage_data),
            response_metadata={"model": self.model},
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
```

Note: The exact way to create SDK tool objects from our `SDKTool` dataclass depends on the SDK's `@tool` decorator API. The implementer should verify the actual SDK `@tool` signature and adapt. The pattern may be:

```python
from claude_agent_sdk import tool as sdk_tool_decorator

# Option A: If @tool returns a decorator
sdk_tool_objects = [
    sdk_tool_decorator(t.name, t.description, t.schema)(t.handler)
    for t in sdk_tools
]

# Option B: If @tool takes the function directly
# Adjust as needed based on actual SDK API
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_model.py -v`
Expected: All tests PASS.

**Step 5: Lint**

Run: `uv run ruff check langchain_claude_agent/chat_model.py`
Expected: No errors.

**Step 6: Commit**

```bash
git add langchain_claude_agent/chat_model.py tests/test_chat_model.py langchain_claude_agent/_tool_converter.py
git commit -m "add tool calling via ClaudeSDKClient with bind_tools support"
```

---

### Task 7: Streaming Support

**Files:**
- Modify: `langchain_claude_agent/chat_model.py`
- Add to: `tests/test_chat_model.py`

**Step 1: Write failing test for streaming**

Add to `tests/test_chat_model.py`:

```python
from langchain_core.messages import AIMessageChunk


class TestChatClaudeAgentStream:
    """Tests for _stream and _astream."""

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="Tell me a story")]

        # Mock StreamEvent objects
        def make_stream_event(text):
            evt = MagicMock()
            evt.__class__.__name__ = "StreamEvent"
            evt.event = {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": text},
            }
            return evt

        async def mock_query(*args, **kwargs):
            yield make_stream_event("Once ")
            yield make_stream_event("upon ")
            yield make_stream_event("a time")
            yield _make_result_message(
                usage={"input_tokens": 10, "output_tokens": 6}
            )

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk)

        # Should have text chunks + final usage chunk
        text_chunks = [c for c in chunks if c.message.content]
        assert len(text_chunks) == 3
        full_text = "".join(c.message.content for c in text_chunks)
        assert full_text == "Once upon a time"

    @pytest.mark.asyncio
    async def test_astream_calls_run_manager(self):
        from langchain_claude_agent import ChatClaudeAgent

        llm = ChatClaudeAgent()
        messages = [HumanMessage(content="hello")]

        def make_stream_event(text):
            evt = MagicMock()
            evt.__class__.__name__ = "StreamEvent"
            evt.event = {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": text},
            }
            return evt

        async def mock_query(*args, **kwargs):
            yield make_stream_event("hi")
            yield _make_result_message()

        mock_manager = AsyncMock()

        with patch("langchain_claude_agent.chat_model.sdk_query", mock_query):
            async for _ in llm._astream(
                messages, run_manager=mock_manager
            ):
                pass

        mock_manager.on_llm_new_token.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_model.py::TestChatClaudeAgentStream -v`
Expected: FAIL - `_astream` not overridden (uses default which calls `_stream` which raises).

**Step 3: Implement `_astream` and `_stream`**

Add to `chat_model.py` imports:

```python
from typing import AsyncIterator
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from claude_agent_sdk.types import StreamEvent
```

Add methods to `ChatClaudeAgent`:

```python
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chat completion tokens asynchronously.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager for on_llm_new_token.
            **kwargs: Additional arguments.

        Yields:
            ChatGenerationChunk with AIMessageChunk for each token.
        """
        system_prompt, chat_messages = extract_system_message(messages)
        prompt = convert_messages_to_prompt(chat_messages)
        options = self._build_options(system_prompt, include_partial_messages=True)

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
            elif isinstance(message, ResultMessage):
                usage = map_sdk_usage(message.usage)
                if usage:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="", usage_metadata=usage
                        )
                    )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion tokens synchronously.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager.
            **kwargs: Additional arguments.

        Yields:
            ChatGenerationChunk with AIMessageChunk for each token.
        """
        loop = asyncio.new_event_loop()
        try:
            astream = self._astream(messages, stop, **kwargs)
            while True:
                try:
                    chunk = loop.run_until_complete(astream.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_model.py -v`
Expected: All tests PASS.

**Step 5: Lint**

Run: `uv run ruff check langchain_claude_agent/chat_model.py`
Expected: No errors.

**Step 6: Commit**

```bash
git add langchain_claude_agent/chat_model.py tests/test_chat_model.py
git commit -m "add streaming support with _astream and _stream methods"
```

---

### Task 8: Update __init__.py Exports and README

**Files:**
- Modify: `langchain_claude_agent/__init__.py`
- Modify: `README.md`

**Step 1: Update `__init__.py` with full exports**

```python
"""LangChain chat model wrapping the Claude Agent SDK."""

from langchain_claude_agent.chat_model import ChatClaudeAgent
from langchain_claude_agent._utils import check_claude_agent_sdk_credentials

__all__ = ["ChatClaudeAgent", "check_claude_agent_sdk_credentials"]
```

**Step 2: Update `README.md`**

```markdown
# langchain-claude-agent

Use the Claude Agent SDK as an LLM provider in LangChain/LangGraph.

## Installation

```bash
uv add langchain-claude-agent
```

Or with pip:

```bash
pip install langchain-claude-agent
```

## Quick Start

```python
from langchain_claude_agent import ChatClaudeAgent

llm = ChatClaudeAgent(model="sonnet")
response = llm.invoke("What is the capital of France?")
print(response.content)
```

## Streaming

```python
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## With LangChain Tools

Tools are executed autonomously by the Claude Agent SDK:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Paris?")
print(response.content)
```

## With SDK Built-in Tools

```python
llm = ChatClaudeAgent(
    model="sonnet",
    allowed_tools=["Read", "Bash", "Grep"],
    cwd="/path/to/project",
)
response = llm.invoke("Find all TODO comments in the codebase")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"sonnet"` | Claude model name |
| `max_turns` | `None` | Maximum agentic turns |
| `max_budget_usd` | `None` | Budget limit in USD |
| `allowed_tools` | `None` | SDK built-in tools to enable |
| `system_prompt` | `None` | Default system prompt |
| `permission_mode` | `"bypassPermissions"` | SDK permission mode |
| `cwd` | `None` | Working directory |

## Authentication

Set your API key:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

Also supports Amazon Bedrock, Google Vertex AI, and Microsoft Azure Foundry via their respective environment variables.

Check credentials programmatically:

```python
from langchain_claude_agent import check_claude_agent_sdk_credentials

ok, message = check_claude_agent_sdk_credentials()
print(f"Credentials: {ok} - {message}")
```
```

**Step 3: Verify import works**

Run: `uv run python -c "from langchain_claude_agent import ChatClaudeAgent; print(ChatClaudeAgent)"`
Expected: `<class 'langchain_claude_agent.chat_model.ChatClaudeAgent'>`

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 5: Run linter on everything**

Run: `uv run ruff check langchain_claude_agent/ tests/`
Expected: No errors.

**Step 6: Commit**

```bash
git add langchain_claude_agent/__init__.py README.md
git commit -m "update package exports and add comprehensive README"
```

---

### Task 9: Final Polish - Ruff Format, Lock File, CLAUDE.md

**Files:**
- All source files
- `uv.lock`
- `CLAUDE.md`

**Step 1: Format all code with ruff**

Run: `uv run ruff format langchain_claude_agent/ tests/`
Expected: Files formatted.

**Step 2: Check with ruff**

Run: `uv run ruff check langchain_claude_agent/ tests/ --fix`
Expected: No remaining issues.

**Step 3: Run full test suite one last time**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 4: Verify uv.lock is up to date**

Run: `uv lock`
Expected: Lock file up to date or regenerated.

**Step 5: Commit everything**

```bash
git add -A
git commit -m "format code with ruff, verify lock file, final polish"
```

---

## Task Dependency Graph

```
Task 1 (scaffolding)
  ├── Task 2 (utils) ─── Task 4 (credentials)
  ├── Task 3 (tool converter)
  │         │
  │         └──── Task 6 (tool calling)
  │
  ├── Task 5 (core chat model) ──── Task 7 (streaming)
  │
  └── Task 8 (exports & README)
          │
          └── Task 9 (final polish)
```

Tasks 2, 3, and 5 can run in parallel after Task 1.
Task 4 depends on Task 2. Task 6 depends on Tasks 3 and 5.
Task 7 depends on Task 5. Task 8 depends on all prior tasks.
Task 9 is the final step.
