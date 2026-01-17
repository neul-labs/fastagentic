# PydanticAI Adapter

The PydanticAI adapter wraps [PydanticAI](https://ai.pydantic.dev/) agents for deployment through FastAgentic. Use PydanticAI's excellent type-safe agent building, deploy with FastAgentic's production runtime.

## TL;DR

Wrap any PydanticAI `Agent` and get REST + MCP + streaming + durability + governance.

## Why PydanticAI + FastAgentic?

PydanticAI excels at type-safe agent orchestration. FastAgentic adds what PydanticAI intentionally doesn't provide:

| Capability | PydanticAI | FastAgentic |
|------------|------------|-------------|
| Type-safe agents | Built-in | Inherited |
| Structured outputs | Built-in | Inherited |
| Model-agnostic | 15+ providers | Inherited |
| Dependency injection | Built-in | Inherited |
| REST API | Manual | Automatic |
| MCP Protocol | Client only | Server + Client |
| Multi-transport streaming | N/A | SSE/WS/MCP |
| Durable checkpoints | Via Temporal | Built-in |
| Auth & Policy | Application code | Middleware |
| Cost guardrails | Logfire monitoring | Enforcement |

**Build with PydanticAI. Deploy with FastAgentic.**

## Before FastAgentic

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
from sse_starlette import EventSourceResponse
from pydantic_ai import Agent
from pydantic import BaseModel
import jwt
import redis
import json

app = FastAPI()
security = HTTPBearer()
redis_client = redis.Redis()

class ChatInput(BaseModel):
    message: str

class ChatOutput(BaseModel):
    response: str
    tokens_used: int

agent = Agent(
    'openai:gpt-4',
    result_type=ChatOutput,
    system_prompt="You are a helpful assistant."
)

async def verify_token(credentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, options={"verify_signature": False})
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/chat", response_model=ChatOutput)
async def chat(input: ChatInput, user = Depends(verify_token)):
    try:
        result = await agent.run(input.message)
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(input: ChatInput, user = Depends(verify_token)):
    async def generate():
        try:
            async with agent.run_stream(input.message) as stream:
                async for chunk in stream.stream():
                    yield {"event": "token", "data": json.dumps({"content": chunk})}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
    return EventSourceResponse(generate())

# Missing:
# - MCP server (100+ lines)
# - Durable checkpoints
# - Resume from failure
# - Cost tracking
# - Rate limiting
# - Audit logging
# - OpenTelemetry traces
# - WebSocket support
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic_ai import Agent
from pydantic import BaseModel

class ChatInput(BaseModel):
    message: str

class ChatOutput(BaseModel):
    response: str
    tokens_used: int

agent = Agent(
    'openai:gpt-4',
    result_type=ChatOutput,
    system_prompt="You are a helpful assistant."
)

app = App(
    title="Chat Agent",
    version="1.0.0",
    oidc_issuer="https://auth.company.com",
    telemetry=True,
    durable_store="redis://localhost:6379",
)

@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    input_model=ChatInput,
    output_model=ChatOutput,
    stream=True,
    durable=True,
    scopes=["chat:run"],
)
async def chat(input: ChatInput) -> ChatOutput:
    pass
```

**Lines of code: 50+ → 25. Features: 3 → 12.**

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /chat` | Run agent synchronously |
| `POST /chat/stream` | Run agent with SSE streaming |
| `GET /chat/{run_id}` | Get run status and result |
| `GET /chat/{run_id}/events` | Replay event stream |
| `POST /chat/{run_id}/resume` | Resume from checkpoint |
| `/mcp/schema` | MCP tool registration |

### PydanticAI-Specific Features

**Tool Integration**

PydanticAI tools are automatically exposed as MCP tools:

```python
@agent.tool
async def search(ctx, query: str) -> list[str]:
    """Search the knowledge base."""
    return await ctx.deps.search_service.search(query)
```

Becomes MCP tool:
```json
{
  "name": "search",
  "description": "Search the knowledge base.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"}
    },
    "required": ["query"]
  }
}
```

**Structured Outputs**

PydanticAI's `result_type` becomes both REST and MCP response schema:

```python
class AnalysisResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    summary: str

agent = Agent('openai:gpt-4', result_type=AnalysisResult)
```

**Streaming Tokens**

Token-level streaming works across all transports:

```python
@agent_endpoint(path="/chat", runnable=PydanticAIAdapter(agent), stream=True)
async def chat(input: ChatInput) -> ChatOutput:
    pass
```

Events emitted:
- `token`: Each output token as generated
- `tool_call`: When agent calls a tool
- `tool_result`: Tool execution result
- `validation_error`: If output validation fails
- `run_complete`: Final validated result

**Dependencies**

PydanticAI dependencies work seamlessly:

```python
from dataclasses import dataclass

@dataclass
class MyDeps:
    search_service: SearchService
    user_id: str

agent = Agent('openai:gpt-4', deps_type=MyDeps)

@agent.tool
async def search(ctx: RunContext[MyDeps], query: str) -> str:
    # Access deps via ctx.deps
    return await ctx.deps.search_service.search(query)
```

Configure deps in FastAgentic:

```python
@agent_endpoint(
    path="/search",
    runnable=PydanticAIAdapter(agent, deps_factory=create_deps),
    ...
)
async def search(query: str) -> SearchResult:
    pass

def create_deps(request: Request) -> MyDeps:
    return MyDeps(
        search_service=request.app.state.search_service,
        user_id=request.state.user.id,
    )
```

## Configuration Options

### PydanticAIAdapter Constructor

```python
PydanticAIAdapter(
    agent: Agent,
    deps: Any = None,
    model: str | None = None,
    checkpoint_on_tool: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` | required | PydanticAI agent instance |
| `deps` | `Any` | `None` | Dependencies to pass to the agent |
| `model` | `str` | `None` | Optional model override |
| `checkpoint_on_tool` | `bool` | `True` | Create checkpoints after tool calls |

### Builder Methods

```python
# Configure dependencies
adapter = PydanticAIAdapter(agent).with_deps(my_deps)

# Override model
adapter = PydanticAIAdapter(agent).with_model("openai:gpt-4o")

# Configure checkpointing
adapter = PydanticAIAdapter(agent).with_checkpoints(on_tool=True)
```

## Event Mapping

| PydanticAI Event | FastAgentic Event | Payload |
|------------------|-------------------|---------|
| Token stream | `token` | `{content}` |
| Tool call start | `tool_call` | `{name, input}` |
| Tool result | `tool_result` | `{name, output}` |
| Checkpoint saved | `checkpoint` | `{step}` |
| Run complete | `done` | `{result}` |

## Checkpoint State

The adapter automatically creates checkpoints containing:

```python
{
    "state": {"completed": bool},
    "messages": [  # Full conversation history
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
    "tool_calls": [  # All tool invocations
        {"name": "search", "input": {...}, "output": "..."},
    ],
    "context": {"result": {...}},
}
```

**Checkpoint Triggers:**
- After each tool call (if `checkpoint_on_tool=True`)
- On run completion

**Resume Behavior:**
When resuming, the adapter:
1. Restores `message_history` from checkpoint
2. Sets `_is_resumed = True` on the run context
3. Emits a `NODE_START` event with `name: "__resume__"`

Resume from any checkpoint:

```bash
curl -X POST https://api.example.com/chat/run-123/resume
```

### Streaming with Tools

Use `stream_with_tools()` for detailed tool event streaming:

```python
adapter = PydanticAIAdapter(agent, checkpoint_on_tool=True)

# Events emitted:
# token → token → tool_call → tool_result → checkpoint → token → done
```

## Migration Guide

### Step 1: Install

```bash
pip install fastagentic[pydanticai]
```

### Step 2: Keep Your Agent

Your PydanticAI agent code stays exactly the same:

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4',
    deps_type=MyDeps,
    result_type=MyResult,
    system_prompt="..."
)

@agent.tool
async def my_tool(ctx, arg: str) -> str:
    ...
```

### Step 3: Wrap with Adapter

```python
from fastagentic.adapters.pydanticai import PydanticAIAdapter

adapter = PydanticAIAdapter(agent)
```

### Step 4: Create Endpoint

```python
from fastagentic import App, agent_endpoint

app = App(title="My Agent", ...)

@agent_endpoint(path="/agent", runnable=adapter, stream=True, durable=True)
async def run_agent(input: MyInput) -> MyResult:
    pass
```

### Step 5: Run

```bash
fastagentic run
```

## Common Patterns

### Chat with Memory

```python
from pydantic_ai import Agent
from pydantic_ai.messages import Message

agent = Agent('openai:gpt-4', system_prompt="...")

@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    stream=True,
    durable=True,  # Persists conversation history
)
async def chat(message: str) -> str:
    pass
```

### Multi-Model Agents

```python
from pydantic_ai import Agent

# Different models for different purposes
fast_agent = Agent('openai:gpt-4o-mini', ...)
smart_agent = Agent('anthropic:claude-3-opus', ...)

@agent_endpoint(path="/quick", runnable=PydanticAIAdapter(fast_agent))
async def quick_response(query: str) -> str:
    pass

@agent_endpoint(path="/detailed", runnable=PydanticAIAdapter(smart_agent))
async def detailed_analysis(query: str) -> AnalysisResult:
    pass
```

### With Cost Limits

```python
@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    cost_limit=0.10,  # $0.10 max per request
    scopes=["chat:run"],
)
async def chat(message: str) -> str:
    pass
```

## Troubleshooting

### Validation Errors

If you see repeated `validation_error` events:

1. Check your `result_type` matches expected output
2. Increase `max_retries` if model is close but not quite right
3. Consider using `Literal` types to constrain outputs
4. Add examples to your system prompt

### Dependency Injection

If deps aren't available:

1. Ensure `deps_factory` is provided to adapter
2. Check factory receives `Request` object
3. Verify deps are created before agent runs

### Streaming Not Working

1. Check `stream=True` on `@agent_endpoint`
2. Verify `stream_tokens=True` on adapter (default)
3. Ensure client accepts `text/event-stream`

## Next Steps

- [Adapters Overview](index.md) - Compare with other adapters
- [Operations Guide](../operations/index.md) - Deploy to production
- [LangGraph Adapter](langgraph.md) - For stateful workflows
