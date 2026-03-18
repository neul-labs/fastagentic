# 5-Minute Quickstart

Get a FastAgentic agent running in under 5 minutes.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key (or Anthropic)

## Step 1: Create Project (30 seconds)

```bash
# Create directory
mkdir my-agent && cd my-agent

# Initialize with uv
uv init
uv add "fastagentic[pydanticai]"
```

## Step 2: Create Agent (1 minute)

Create `app.py`:

```python
from fastagentic import App, agent_endpoint, tool
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic import BaseModel
from pydantic_ai import Agent

# Create your agent
agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)

# Add a tool
@agent.tool
async def get_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

# Create FastAgentic app
app = App(title="My Agent", version="1.0.0")

# Define request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Expose as endpoint
@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    input_model=ChatRequest,
    output_model=ChatResponse,
    stream=True,
)
async def chat(request: ChatRequest) -> ChatResponse:
    pass
```

## Step 3: Set API Key (10 seconds)

```bash
export OPENAI_API_KEY="sk-your-key"
```

## Step 4: Run (30 seconds)

```bash
uv run fastagentic run
```

Output:
```
Starting FastAgentic server...
  App: app:app
  URL: http://127.0.0.1:8000
```

## Step 5: Test (1 minute)

### Option A: Agent CLI (Recommended)

```bash
uv run fastagentic agent chat
```

```
FastAgentic Agent CLI
> What time is it?

The current time is 14:32:15.

> /quit
```

### Option B: curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}'
```

### Option C: OpenAPI Docs

Open http://localhost:8000/docs in your browser.

## What You Get

| Interface | URL |
|-----------|-----|
| REST API | `POST /chat` |
| OpenAPI Docs | `/docs` |
| MCP Schema | `/mcp/schema` |
| Health Check | `/health` |

## Next Steps

### Add more tools

```python
@agent.tool
async def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"
```

### Enable streaming

Already enabled! Use SSE client or Agent CLI.

### Add authentication

```python
app = App(
    title="My Agent",
    oidc_issuer="https://auth.example.com",
)
```

### Enable persistence

```bash
# Start Redis
docker run -d -p 6379:6379 redis

# Update app.py
app = App(
    title="My Agent",
    durable_store="redis://localhost:6379",
)
```

### Use with Claude Desktop (MCP)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-agent": {
      "command": "fastagentic",
      "args": ["mcp", "serve", "app:app"],
      "cwd": "/path/to/my-agent"
    }
  }
}
```

## Complete Example

```python
"""Complete FastAgentic agent in one file."""

from datetime import datetime
from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic import BaseModel
from pydantic_ai import Agent

# Agent with tools
agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant with access to tools.",
)

@agent.tool
async def get_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@agent.tool
async def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"

# FastAgentic app
app = App(title="Quick Agent", version="1.0.0")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    input_model=ChatRequest,
    output_model=ChatResponse,
    stream=True,
    mcp_tool="chat",
)
async def chat(request: ChatRequest) -> ChatResponse:
    pass

@resource(name="info", uri="info")
async def info() -> dict:
    return {"name": "Quick Agent", "version": "1.0.0"}

# Run: uv run fastagentic run
```

## Troubleshooting

### "Module not found"

```bash
uv sync  # Reinstall dependencies
```

### "API key not set"

```bash
export OPENAI_API_KEY="sk-..."
# Or for Anthropic:
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Port already in use

```bash
uv run fastagentic run --port 8001
```

## Learn More

- [Getting Started Guide](getting-started.md) - Full setup guide
- [Examples](../examples/) - Complete working examples
- [Adapters](adapters/index.md) - All supported frameworks
- [CLI Reference](cli-reference.md) - All commands
