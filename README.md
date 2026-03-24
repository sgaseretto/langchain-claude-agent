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

Bound LangChain tools are surfaced as standard LangChain tool calls:

```python
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

llm_with_tools = llm.bind_tools([get_weather])
ai_msg = llm_with_tools.invoke("What's the weather in Paris?")

tool_result = get_weather.invoke(ai_msg.tool_calls[0]["args"])
final = llm_with_tools.invoke(
    [
        ("human", "What's the weather in Paris?"),
        ai_msg,
        ToolMessage(
            content=tool_result,
            tool_call_id=ai_msg.tool_calls[0]["id"],
            name="get_weather",
        ),
    ]
)
print(final.content)
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
- **Tool calls** (`bind_tools().invoke()`): Converts LangChain tools to an in-process MCP server, maps SDK tool-use blocks back to LangChain `tool_calls`, and expects `ToolMessage` follow-up inputs
- **Structured output** (`with_structured_output()`): Uses the SDK's native `output_format` option for constrained JSON generation
- **Extended thinking**: Passes `thinking` config to `ClaudeAgentOptions`; thinking blocks are returned in `additional_kwargs["thinking"]`

SDK built-in tools remain autonomous. Bound LangChain tools behave like a normal chat-model provider so they can participate in standard LangChain and LangGraph tool loops.

## Notebooks

Install the notebook dependencies first:

```bash
uv sync --extra dev
```

The repository includes a small numbered notebook suite:

- [`notebooks/00_features.ipynb`](notebooks/00_features.ipynb): Core chat-model features, tool calling, structured output, and multimodal input
- [`notebooks/01_react_agent_weather.ipynb`](notebooks/01_react_agent_weather.ipynb): A LangChain `create_agent(...)` weather agent
- [`notebooks/02_react_agent_weather_structured_output.ipynb`](notebooks/02_react_agent_weather_structured_output.ipynb): The same weather agent with structured output
- [`notebooks/03_react_agent_weather_mcp.ipynb`](notebooks/03_react_agent_weather_mcp.ipynb): A weather agent powered by MCP tools

[`notebooks/features.ipynb`](notebooks/features.ipynb) remains as a compact feature demo and mirrors `00_features.ipynb`.
