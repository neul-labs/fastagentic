# PydanticAI Chat Agent - Claude Code Guide

This is a FastAgentic example using PydanticAI for a simple chat agent.

## Project Structure

```
pydanticai-chat/
├── CLAUDE.md          # This file - instructions for Claude Code
├── app.py             # Main FastAgentic application
├── agent.py           # PydanticAI agent definition
├── models.py          # Pydantic models for input/output
├── tools.py           # Agent tools
├── pyproject.toml     # Dependencies
├── .env.example       # Environment variables template
└── README.md          # User documentation
```

## Key Commands

```bash
# Install dependencies
uv sync

# Run the server
uv run fastagentic run

# Test with CLI
uv run fastagentic agent chat --endpoint /chat

# Run tests
uv run pytest tests/ -v
```

## Architecture

- **FastAgentic App** (`app.py`): Wraps the PydanticAI agent with REST/MCP/A2A interfaces
- **PydanticAI Agent** (`agent.py`): The actual AI agent with system prompt and tools
- **Tools** (`tools.py`): Functions the agent can call
- **Models** (`models.py`): Request/response schemas

## When Modifying

1. **Add new tools**: Edit `tools.py` and register in `agent.py`
2. **Change system prompt**: Edit `SYSTEM_PROMPT` in `agent.py`
3. **Add new endpoints**: Use `@agent_endpoint` in `app.py`
4. **Change models**: Edit `models.py` and update endpoint signatures

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models
- `ANTHROPIC_API_KEY`: Required for Claude models
- `FASTAGENTIC_ENV`: Environment (dev/staging/prod)
- `REDIS_URL`: Optional durable store

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Test specific endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## Common Tasks

### Add a new tool

```python
# In tools.py
async def my_new_tool(query: str) -> str:
    """Tool description for the agent."""
    return f"Result for {query}"

# In agent.py
from tools import my_new_tool
agent = Agent(..., tools=[my_new_tool])
```

### Enable streaming

```python
@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    stream=True,  # Enable SSE streaming
)
```

### Add authentication

```python
app = App(
    title="Chat Agent",
    oidc_issuer="https://auth.example.com",
)

@agent_endpoint(
    path="/chat",
    scopes=["chat:read", "chat:write"],
)
```
