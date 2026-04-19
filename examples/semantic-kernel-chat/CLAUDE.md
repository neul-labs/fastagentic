# Semantic Kernel Chat - Claude Code Guide

This is a FastAgentic example using Microsoft Semantic Kernel for chat with plugins.

## Project Structure

```
semantic-kernel-chat/
├── CLAUDE.md           # This file - instructions for Claude Code
├── app.py              # Main FastAgentic application with Semantic Kernel
├── pyproject.toml      # Dependencies
├── .env.example        # Environment variables template
└── README.md           # User documentation
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

- **FastAgentic App** (`app.py`): Wraps Semantic Kernel with REST/MCP interfaces
- **Semantic Kernel**: Microsoft's AI orchestration framework
- **Plugins**: Native function plugins with `@kernel_function`
- **Services**: OpenAI or Azure OpenAI chat completion

## Key Components

### Semantic Kernel Setup

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o-mini"))
```

### Plugin Definition

```python
class UtilityPlugin:
    @kernel_function(name="get_time", description="Get the current time")
    def get_time(self) -> str:
        return datetime.now().isoformat()

kernel.add_plugin(UtilityPlugin(), "Utility")
```

### SemanticKernelAdapter

```python
@agent_endpoint(
    path="/chat",
    runnable=SemanticKernelAdapter(kernel),
    stream=True,
    durable=True,
)
```

## When Modifying

1. **Add plugins**: Create class with `@kernel_function` methods, register with `kernel.add_plugin()`
2. **Change model**: Modify the `OpenAIChatCompletion` service configuration
3. **Use Azure**: Switch to `AzureChatCompletion` service
4. **Add agents**: Use `ChatCompletionAgent` with the adapter

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models
- `AZURE_OPENAI_ENDPOINT`: For Azure OpenAI
- `AZURE_OPENAI_API_KEY`: For Azure OpenAI
- `FASTAGENTIC_ENV`: Environment (dev/staging/prod)
- `REDIS_URL`: Optional durable store

## Testing

```bash
# Run tests
uv run pytest

# Test endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## Common Tasks

### Add a new plugin function

```python
class MyPlugin:
    @kernel_function(name="search", description="Search the web")
    def search(self, query: str) -> str:
        # Implementation
        return f"Results for: {query}"

kernel.add_plugin(MyPlugin(), "Search")
```

### Use ChatCompletionAgent

```python
from semantic_kernel.agents import ChatCompletionAgent

agent = ChatCompletionAgent(
    kernel=kernel,
    name="Assistant",
    instructions="You are a helpful assistant.",
)

adapter = SemanticKernelAdapter(kernel, agent=agent)
```

### Enable function calling

```python
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

settings = OpenAIChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto()
)

adapter = SemanticKernelAdapter(kernel, settings=settings)
```
