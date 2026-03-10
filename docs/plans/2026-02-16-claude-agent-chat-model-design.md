# Design: LangChain Custom Chat Model for Claude Agent SDK

## Overview

A LangChain `BaseChatModel` implementation (`ChatClaudeAgent`) that wraps the Claude Agent SDK, enabling Claude to be used as a chat model within LangChain/LangGraph workflows. Supports text generation, streaming, tool calling, and cost tracking.

## Architecture

**Hybrid approach:** Uses `query()` for basic text generation (no tools) and `ClaudeSDKClient` when tools are bound via `bind_tools()`. The decision is made internally based on whether tools are present in kwargs.

```
User code
  │
  ├─ model.invoke("hello")          → query()           → AIMessage
  ├─ model.stream("hello")          → query() + stream  → AIMessageChunk*
  └─ model.bind_tools([t]).invoke() → ClaudeSDKClient    → AIMessage
```

## Package Structure

```
langchain_claude_agent/
├── __init__.py              # Public API: ChatClaudeAgent, check_credentials
├── chat_model.py            # ChatClaudeAgent (BaseChatModel subclass)
├── _utils.py                # Message conversion, usage mapping, credentials
├── _tool_converter.py       # LangChain BaseTool → SDK @tool conversion
└── _types.py                # Type aliases and constants

tests/
├── __init__.py
├── test_chat_model.py
├── test_utils.py
└── test_tool_converter.py
```

## ChatClaudeAgent Class

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"sonnet"` | SDK model name |
| `max_turns` | `int \| None` | `None` | Max agentic turns |
| `max_budget_usd` | `float \| None` | `None` | Budget limit in USD |
| `allowed_tools` | `list[str] \| None` | `None` | SDK built-in tools (Read, Bash, etc.) |
| `system_prompt` | `str \| None` | `None` | SDK system prompt (fallback if no SystemMessage) |
| `permission_mode` | `str` | `"bypassPermissions"` | Permission mode for SDK |
| `cwd` | `str \| None` | `None` | Working directory for SDK |

### LangChain Interface Methods

- `_generate()` / `_agenerate()`: Core generation, routes to query() or ClaudeSDKClient
- `_stream()` / `_astream()`: Token-level streaming via `include_partial_messages=True`
- `bind_tools()`: Converts LangChain tools to SDK MCP tools, returns RunnableBinding
- `_llm_type`: Returns `"claude-agent-sdk"`

### Sync/Async

The SDK is async-first. Sync methods (`_generate`, `_stream`) wrap their async counterparts using `asyncio.run()` (or event loop detection for nested loops).

## Message Conversion

The SDK takes a string prompt, not structured messages. Conversion strategy:

- `SystemMessage` → Extracted, passed as `ClaudeAgentOptions.system_prompt` (takes precedence over `self.system_prompt`)
- `HumanMessage` → `Human: {content}`
- `AIMessage` → `Assistant: {content}`
- `ToolMessage` → `Tool Result ({name}): {content}`

## SDK Interaction

### No Tools (query path)

1. Extract SystemMessage, convert remaining messages to prompt string
2. Create `ClaudeAgentOptions` with model, system_prompt, max_turns, etc.
3. Iterate over `query()` results, collect text from `AssistantMessage` blocks
4. Extract usage from `ResultMessage`
5. Return `ChatResult` with `AIMessage` containing text and `usage_metadata`

### With Tools (ClaudeSDKClient path)

1. Convert LangChain tools to SDK MCP tools via `convert_langchain_tool_to_sdk()`
2. Create `create_sdk_mcp_server()` with converted tools
3. Configure `ClaudeAgentOptions` with MCP server and tool names
4. Use `ClaudeSDKClient` context manager to execute
5. Collect results and return `ChatResult`

The SDK executes tools autonomously - no LangGraph ToolNode needed.

## Tool Conversion

LangChain `BaseTool` → SDK `@tool` function:

1. Extract name, description, args schema from LangChain tool
2. Create async wrapper that calls `lc_tool.ainvoke(args)`
3. Wrap with `@tool(name, description, schema)` decorator
4. Return SDK-compatible tool function

Tool naming follows SDK convention: `mcp__langchain-tools__{tool_name}`

## Streaming

Uses `include_partial_messages=True` on `ClaudeAgentOptions`:

1. `StreamEvent` messages with `content_block_delta` / `text_delta` → yield `ChatGenerationChunk`
2. Call `run_manager.on_llm_new_token()` for each chunk (enables LangGraph auto-streaming)
3. Final `ResultMessage` → yield usage metadata chunk

## Cost Tracking

Maps SDK `ResultMessage.usage` to LangChain `UsageMetadata`:

- `input_tokens` → `input_tokens`
- `output_tokens` → `output_tokens`
- `cache_read_input_tokens` → `input_token_details.cache_read`
- `cache_creation_input_tokens` → `input_token_details.cache_creation`
- `total_cost_usd` available on `ResultMessage`

## Credential Checking

`ChatClaudeAgent.check_credentials()` class method:

1. Check `ANTHROPIC_API_KEY` environment variable
2. Probe SDK via `query()` with `max_turns=0` to verify auth
3. Return `(bool, str)` tuple with status and message

Supports API key auth, Claude Code subscription, Bedrock, Vertex AI, Azure Foundry.

## Dependencies

- `langchain-core >= 0.3.0`
- `claude-agent-sdk`
- `fastcore`
- `pydantic >= 2.0`

Dev dependencies: `ruff`, `pytest`, `pytest-asyncio`

## Usage Examples

### Basic

```python
from langchain_claude_agent import ChatClaudeAgent

llm = ChatClaudeAgent(model="sonnet")
response = llm.invoke("What is the capital of France?")
```

### Streaming

```python
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### With Tools

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Paris?")
```

### With SDK Built-in Tools

```python
llm = ChatClaudeAgent(
    model="sonnet",
    allowed_tools=["Read", "Bash", "Grep"],
    cwd="/path/to/project",
)
response = llm.invoke("Find all TODO comments in the codebase")
```

### In LangGraph

```python
from langgraph.graph import StateGraph, MessagesState

graph = StateGraph(MessagesState)
graph.add_node("llm", lambda state: {"messages": [llm.invoke(state["messages"])]})
graph.set_entry_point("llm")
graph.set_finish_point("llm")
app = graph.compile()
```
