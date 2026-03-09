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
from langchain_claude_agent import ChatClaudeAgSDK

llm = ChatClaudeAgSDK(model="sonnet")
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
llm = ChatClaudeAgSDK(
    model="sonnet",
    allowed_tools=["Read", "Bash", "Grep"],
    cwd="/path/to/project",
)
response = llm.invoke("Find all TODO comments in the codebase")
```

## Structured Output

Constrain the model to return data matching a Pydantic model or JSON schema:

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Tell me about Marie Curie")
print(result.name, result.age)
```

## Extended Thinking

Enable Claude's extended thinking to see its reasoning process:

```python
thinking_llm = ChatClaudeAgSDK(
    model="sonnet",
    thinking={"type": "enabled", "budget_tokens": 10000},
)
response = thinking_llm.invoke("What is 15 * 37?")
print(response.content)

# Access thinking blocks
if "thinking" in response.additional_kwargs:
    for block in response.additional_kwargs["thinking"]:
        print(block["thinking"])
```

## Effort Level

Control how much effort the model puts into responses:

```python
quick_llm = ChatClaudeAgSDK(model="sonnet", effort="low")
thorough_llm = ChatClaudeAgSDK(model="sonnet", effort="high")
```

## In LangGraph

```python
from langgraph.graph import StateGraph, MessagesState

llm = ChatClaudeAgSDK(model="sonnet")

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
| `thinking` | `None` | Extended thinking config |
| `effort` | `None` | Effort level (`"low"`, `"medium"`, `"high"`, `"max"`) |

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

`ChatClaudeAgSDK` extends LangChain's `BaseChatModel` using the SDK's `query()` function for all generation paths:

- **Basic calls** (`invoke`, `stream`): Calls `query()` with the converted messages
- **Tool calls** (`bind_tools().invoke()`): Converts LangChain tools to an in-process MCP server and passes it via `ClaudeAgentOptions.mcp_servers` to `query()`
- **Structured output** (`with_structured_output()`): Uses the SDK's native `output_format` option for constrained JSON generation
- **Extended thinking**: Passes `thinking` config to `ClaudeAgentOptions`; thinking blocks are returned in `additional_kwargs["thinking"]`

The SDK handles tool execution autonomously -- when tools are bound, Claude decides when and how to use them, executes them, and returns the final result. This means LangGraph's `ToolNode` is not needed; the model handles the entire tool loop internally.

## Notebook

See [`notebooks/features.ipynb`](notebooks/features.ipynb) for a comprehensive feature demo covering all capabilities.
