"""Tests for tool converter: LangChain tools to SDK-compatible tool specs."""

from __future__ import annotations

from langchain_core.tools import tool as lc_tool

from langchain_claude_agent._tool_converter import (
    SDKToolSpec,
    convert_langchain_tool_to_sdk,
    convert_langchain_tools,
    get_tool_schema,
)

# ---------------------------------------------------------------------------
# Fixtures: LangChain tools
# ---------------------------------------------------------------------------


@lc_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@lc_tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


# ---------------------------------------------------------------------------
# TestGetToolSchema
# ---------------------------------------------------------------------------


class TestGetToolSchema:
    """Tests for get_tool_schema."""

    def test_integer_params(self):
        """Integer parameters should map to Python int."""
        schema = get_tool_schema(add)
        assert schema == {"a": int, "b": int}

    def test_string_param(self):
        """String parameters should map to Python str."""
        schema = get_tool_schema(greet)
        assert schema == {"name": str}


# ---------------------------------------------------------------------------
# TestConvertTool
# ---------------------------------------------------------------------------


class TestConvertTool:
    """Tests for convert_langchain_tool_to_sdk."""

    def test_name_preserved(self):
        """The SDK spec should preserve the LangChain tool name."""
        spec = convert_langchain_tool_to_sdk(add)
        assert spec.name == "add"

    def test_description_preserved(self):
        """The SDK spec should preserve the LangChain tool description."""
        spec = convert_langchain_tool_to_sdk(add)
        assert spec.description == "Add two integers."

    def test_schema_preserved(self):
        """The SDK spec should contain the converted schema."""
        spec = convert_langchain_tool_to_sdk(add)
        assert spec.schema == {"a": int, "b": int}

    async def test_handler_execution_add(self):
        """The async handler should invoke the tool and return SDK format."""
        spec = convert_langchain_tool_to_sdk(add)
        result = await spec.handler({"a": 3, "b": 4})
        assert result == {"content": [{"type": "text", "text": "7"}]}

    async def test_handler_execution_greet(self):
        """The async handler should work for string tools too."""
        spec = convert_langchain_tool_to_sdk(greet)
        result = await spec.handler({"name": "Alice"})
        assert result == {"content": [{"type": "text", "text": "Hello, Alice!"}]}

    def test_spec_is_dataclass(self):
        """The returned spec should be an SDKToolSpec instance."""
        spec = convert_langchain_tool_to_sdk(add)
        assert isinstance(spec, SDKToolSpec)


# ---------------------------------------------------------------------------
# TestConvertToolList
# ---------------------------------------------------------------------------


class TestConvertToolList:
    """Tests for convert_langchain_tools batch conversion."""

    def test_converts_multiple_tools(self):
        """Batch conversion should return the correct number of specs."""
        specs = convert_langchain_tools([add, greet])
        assert len(specs) == 2

    def test_names_match(self):
        """Batch conversion should preserve names for all tools."""
        specs = convert_langchain_tools([add, greet])
        names = [s.name for s in specs]
        assert names == ["add", "greet"]

    def test_empty_list(self):
        """An empty tool list should produce an empty spec list."""
        specs = convert_langchain_tools([])
        assert specs == []
