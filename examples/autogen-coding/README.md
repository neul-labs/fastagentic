# AutoGen Coding Assistant

A multi-agent coding assistant built with Microsoft AutoGen and deployed with FastAgentic.

## Features

- AutoGen multi-agent conversation (AssistantAgent + UserProxyAgent)
- Code generation and explanation
- Per-turn checkpointing for durability
- Code execution capability (optional Docker isolation)
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
uv run fastagentic agent chat --endpoint /code
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/code` | POST | Get coding help |
| `/code/stream` | POST | Get help with streaming |
| `/code/{run_id}` | GET | Get run status |
| `/code/{run_id}/resume` | POST | Resume from checkpoint |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/mcp/schema` | GET | MCP schema |

## Testing

```bash
# Ask a coding question
curl -X POST http://localhost:8000/code \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a Python function to check if a number is prime"}'

# Debug code
curl -X POST http://localhost:8000/code \
  -H "Content-Type: application/json" \
  -d '{"task": "Why does this code not work? def add(a, b) return a + b"}'

# Streaming response
curl -N -X POST http://localhost:8000/code/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"task": "Explain how async/await works in Python"}'

# Using the Agent CLI
uv run fastagentic agent chat --endpoint /code
```

## Agent Architecture

The example uses two AutoGen agents:

### AssistantAgent (coder)
- Generates and explains code
- Provides debugging help
- Suggests improvements

### UserProxyAgent (user)
- Handles code execution requests
- Manages conversation flow
- Enforces max_turns limit

## Code Execution

By default, code execution is enabled without Docker:

```python
user_proxy = UserProxyAgent(
    name="user",
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False,
    },
)
```

For isolated execution, enable Docker:

```python
code_execution_config={
    "work_dir": "workspace",
    "use_docker": True,
}
```

## Project Structure

```
autogen-coding/
├── app.py              # FastAgentic application with AutoGen
├── workspace/          # Code execution workspace (created at runtime)
├── pyproject.toml      # Dependencies
├── .env.example        # Environment template
├── CLAUDE.md           # Claude Code instructions
└── README.md           # This file
```

## Group Chat Extension

For multi-agent collaboration, use GroupChat:

```python
from autogen import GroupChat, GroupChatManager

researcher = AssistantAgent("researcher", llm_config=config)
coder = AssistantAgent("coder", llm_config=config)
reviewer = AssistantAgent("reviewer", llm_config=config)

group_chat = GroupChat(
    agents=[researcher, coder, reviewer],
    messages=[],
    max_round=10,
)

adapter = AutoGenAdapter(
    initiator=user_proxy,
    recipient=researcher,
    group_chat=group_chat,
)
```

## Learn More

- [FastAgentic Documentation](https://github.com/neul-labs/fastagentic)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AutoGen Adapter Guide](../../docs/adapters/autogen.md)
