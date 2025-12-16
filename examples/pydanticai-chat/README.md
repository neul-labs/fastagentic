# PydanticAI Chat Agent

A simple chat agent built with PydanticAI and deployed with FastAgentic.

## Features

- Chat with an AI assistant
- Tool calling (time, calculator)
- Streaming responses via SSE
- REST, MCP, and A2A interfaces

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the server
uv run fastagentic run

# 4. Test with CLI
uv run fastagentic agent chat
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with the agent |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/mcp/schema` | GET | MCP schema |

## Testing

```bash
# Using curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}'

# Using the Agent CLI
fastagentic agent chat --endpoint /chat

# Streaming response
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message": "Tell me a joke", "stream": true}'
```

## MCP Integration

Add to Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "pydanticai-chat": {
      "command": "fastagentic",
      "args": ["mcp", "serve", "app:app"]
    }
  }
}
```

## Project Structure

```
pydanticai-chat/
├── app.py          # FastAgentic application
├── agent.py        # PydanticAI agent definition
├── models.py       # Request/response models
├── pyproject.toml  # Dependencies
├── .env.example    # Environment template
├── CLAUDE.md       # Claude Code instructions
└── README.md       # This file
```

## Customization

### Change the model

Edit `agent.py`:

```python
chat_agent = Agent(
    model="anthropic:claude-3-haiku-20240307",  # Use Claude
    system_prompt=SYSTEM_PROMPT,
)
```

### Add a new tool

```python
# In agent.py
@chat_agent.tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    # Implement search logic
    return f"Results for: {query}"
```

### Enable persistence

```python
# In app.py
app = App(
    title="PydanticAI Chat Agent",
    durable_store="redis://localhost:6379",
)
```

## Learn More

- [FastAgentic Documentation](https://fastagentic.dev)
- [PydanticAI Documentation](https://ai.pydantic.dev)
- [Adapters Guide](../../docs/adapters/pydanticai.md)
