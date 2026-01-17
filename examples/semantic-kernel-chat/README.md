# Semantic Kernel Chat

A chat agent with plugins built with Microsoft Semantic Kernel and deployed with FastAgentic.

## Features

- Microsoft Semantic Kernel with plugin system
- Native function calling with `@kernel_function`
- Streaming responses via SSE
- Checkpointing for durability
- Azure OpenAI compatible
- REST, MCP, and A2A interfaces

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (or Azure OpenAI config)

# 3. Run the server
uv run fastagentic run

# 4. Test with CLI
uv run fastagentic agent chat --endpoint /chat
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with the assistant |
| `/chat/stream` | POST | Chat with streaming response |
| `/chat/{run_id}` | GET | Get run status |
| `/chat/{run_id}/resume` | POST | Resume from checkpoint |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/mcp/schema` | GET | MCP schema |

## Testing

```bash
# Simple chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}'

# Use calculator plugin
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate 15 * 23"}'

# Streaming response
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message": "Tell me a joke"}'

# Using the Agent CLI
uv run fastagentic agent chat --endpoint /chat
```

## Plugins

### Built-in Utility Plugin

The example includes a `UtilityPlugin` with:

| Function | Description |
|----------|-------------|
| `get_time` | Get current date and time |
| `calculate` | Perform arithmetic calculations |
| `format_text` | Format text (upper, lower, title, reverse) |

### Adding Custom Plugins

```python
class MyPlugin:
    @kernel_function(name="my_function", description="Does something useful")
    def my_function(self, input: str) -> str:
        return f"Processed: {input}"

kernel.add_plugin(MyPlugin(), "MyPlugin")
```

## Project Structure

```
semantic-kernel-chat/
├── app.py              # FastAgentic application with Semantic Kernel
├── pyproject.toml      # Dependencies
├── .env.example        # Environment template
├── CLAUDE.md           # Claude Code instructions
└── README.md           # This file
```

## Azure OpenAI Configuration

To use Azure OpenAI instead of OpenAI:

```python
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel.add_service(
    AzureChatCompletion(
        service_id="chat",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
)
```

## Learn More

- [FastAgentic Documentation](https://github.com/neul-labs/fastagentic)
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Semantic Kernel Adapter Guide](../../docs/adapters/semantic_kernel.md)
