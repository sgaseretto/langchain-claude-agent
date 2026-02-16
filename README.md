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

## In LangGraph

```python
from langgraph.graph import StateGraph, MessagesState

llm = ChatClaudeAgent(model="sonnet")

graph = StateGraph(MessagesState)
graph.add_node("llm", lambda state: {"messages": [llm.invoke(state["messages"])]})
graph.set_entry_point("llm")
graph.set_finish_point("llm")
app = graph.compile()
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

## How It Works

`ChatClaudeAgent` extends LangChain's `BaseChatModel` with a hybrid architecture:

- **Basic calls** (`invoke`, `stream`): Uses the SDK's `query()` function for simple text generation
- **Tool calls** (`bind_tools().invoke()`): Uses `ClaudeSDKClient` with MCP servers for autonomous tool execution

The SDK handles tool execution autonomously -- when tools are bound, Claude decides when and how to use them, executes them, and returns the final result. This means LangGraph's `ToolNode` is not needed; the model handles the entire tool loop internally.
