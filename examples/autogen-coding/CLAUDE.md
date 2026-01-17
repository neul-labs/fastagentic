# AutoGen Coding Assistant - Claude Code Guide

This is a FastAgentic example using Microsoft AutoGen for multi-agent coding assistance.

## Project Structure

```
autogen-coding/
├── CLAUDE.md           # This file - instructions for Claude Code
├── app.py              # Main FastAgentic application with AutoGen
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
uv run fastagentic agent chat --endpoint /code

# Run tests
uv run pytest tests/ -v
```

## Architecture

- **FastAgentic App** (`app.py`): Wraps AutoGen agents with REST/MCP interfaces
- **AssistantAgent**: The coding assistant that generates and explains code
- **UserProxyAgent**: Manages conversation flow and code execution
- **Checkpointing**: Per-turn state saved for durability

## Key Components

### AutoGen Agent Setup

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(name="coder", llm_config=config)
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)
```

### AutoGenAdapter

```python
@agent_endpoint(
    path="/code",
    runnable=AutoGenAdapter(
        initiator=user_proxy,
        recipient=assistant,
        max_turns=5,
        checkpoint_turns=True,
    ),
    stream=True,
    durable=True,
)
```

## When Modifying

1. **Change agent behavior**: Modify `system_message` in AssistantAgent
2. **Add more agents**: Create additional AssistantAgents and use GroupChat
3. **Enable Docker**: Set `use_docker=True` in code_execution_config
4. **Adjust turns**: Modify `max_turns` in adapter or `max_consecutive_auto_reply` in user_proxy

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models
- `FASTAGENTIC_ENV`: Environment (dev/staging/prod)
- `REDIS_URL`: Optional durable store

## Testing

```bash
# Run tests
uv run pytest

# Test endpoint
curl -X POST http://localhost:8000/code \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a hello world program in Python"}'
```

## Common Tasks

### Add a specialized agent

```python
reviewer = AssistantAgent(
    name="reviewer",
    llm_config=config,
    system_message="You are a code reviewer. Review code for best practices and bugs.",
)
```

### Create group chat

```python
from autogen import GroupChat, GroupChatManager

group_chat = GroupChat(
    agents=[assistant, reviewer, user_proxy],
    messages=[],
    max_round=10,
)

manager = GroupChatManager(groupchat=group_chat, llm_config=config)

adapter = AutoGenAdapter(
    initiator=user_proxy,
    recipient=assistant,
    group_chat=group_chat,
)
```

### Enable function calling

```python
from autogen import register_function

def search_docs(query: str) -> str:
    return f"Documentation for: {query}"

register_function(
    search_docs,
    caller=assistant,
    executor=user_proxy,
    name="search_docs",
    description="Search documentation",
)
```

### Disable code execution

```python
user_proxy = UserProxyAgent(
    name="user",
    code_execution_config=False,  # Disable execution
)
```
