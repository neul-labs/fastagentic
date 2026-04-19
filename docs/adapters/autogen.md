# AutoGen Adapter

The AutoGen adapter wraps [Microsoft AutoGen](https://microsoft.github.io/autogen/) for deployment through FastAgentic. Deploy multi-agent conversations with full checkpointing and streaming support.

## TL;DR

Wrap any AutoGen agent pair or group chat and get REST + MCP + streaming + durability + per-turn checkpoints.

## Why AutoGen + FastAgentic?

AutoGen excels at multi-agent conversations and code execution. FastAgentic adds production deployment:

| Capability | AutoGen | FastAgentic |
|------------|---------|-------------|
| Multi-agent chat | Built-in | Inherited |
| Code execution | Built-in | Inherited |
| Group chats | Built-in | Inherited |
| Human-in-the-loop | Built-in | Inherited |
| REST API | Manual | Automatic |
| MCP Protocol | No | Yes |
| Checkpoints | No | Built-in |
| Token streaming | v0.4+ API | Automatic fallback |
| Auth & Policy | Application code | Middleware |

**Orchestrate with AutoGen. Deploy with FastAgentic.**

## Before FastAgentic

```python
from fastapi import FastAPI
from autogen import AssistantAgent, UserProxyAgent

app = FastAPI()

assistant = AssistantAgent("assistant", llm_config={"model": "gpt-4"})
user_proxy = UserProxyAgent("user", code_execution_config={"use_docker": False})

@app.post("/chat")
async def chat(message: str):
    result = await user_proxy.a_initiate_chat(assistant, message=message, max_turns=5)
    return {"response": result.chat_history[-1]["content"]}

# No streaming, no checkpoints, no auth, limited observability...
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.autogen import AutoGenAdapter
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={"model": "gpt-4"})
user_proxy = UserProxyAgent("user", code_execution_config={"use_docker": False})

app = App(
    title="AutoGen Chat",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost",
)

@agent_endpoint(
    path="/chat",
    runnable=AutoGenAdapter(initiator=user_proxy, recipient=assistant),
    stream=True,
    durable=True,
)
async def chat(message: str) -> str:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /chat` | Run conversation synchronously |
| `POST /chat/stream` | Run with per-turn streaming |
| `GET /chat/{run_id}` | Get run status and result |
| `POST /chat/{run_id}/resume` | Resume from checkpoint |

### AutoGen-Specific Features

**Two-Agent Conversations**

```python
adapter = AutoGenAdapter(
    initiator=user_proxy,
    recipient=assistant,
    max_turns=10,
)
```

**Group Chats**

```python
from autogen import GroupChat

group_chat = GroupChat(
    agents=[researcher, coder, reviewer],
    messages=[],
    max_round=10,
)

adapter = AutoGenAdapter(
    initiator=user_proxy,
    recipient=researcher,  # First agent
    group_chat=group_chat,
)
```

**Native Streaming (AutoGen v0.4+)**

When available, uses native streaming APIs:

```python
# Automatic detection - uses on_messages_stream() if available
adapter = AutoGenAdapter(initiator=user_proxy, recipient=assistant)

# Native streaming events:
# token → token → checkpoint → message → checkpoint → done
```

## Configuration Options

### AutoGenAdapter Constructor

```python
AutoGenAdapter(
    initiator: Agent,
    recipient: Agent,
    group_chat: GroupChat | None = None,
    max_turns: int | None = None,
    clear_history: bool = False,
    silent: bool = True,
    checkpoint_turns: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initiator` | `Agent` | required | Agent that initiates conversation |
| `recipient` | `Agent` | required | Agent that receives messages |
| `group_chat` | `GroupChat` | `None` | Group chat for multi-agent scenarios |
| `max_turns` | `int` | `None` | Maximum conversation turns |
| `clear_history` | `bool` | `False` | Clear history between runs |
| `silent` | `bool` | `True` | Suppress console output |
| `checkpoint_turns` | `bool` | `True` | Checkpoint after each turn |

### Builder Methods

```python
# Set max turns
adapter = AutoGenAdapter(initiator, recipient).with_max_turns(10)

# Add group chat
adapter = AutoGenAdapter(initiator, recipient).with_group_chat(group_chat)

# Configure checkpointing
adapter = AutoGenAdapter(initiator, recipient).with_checkpoints(enabled=True)
```

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `node_start` | Resume from checkpoint | `{name: "__resume__", checkpoint}` |
| `message` | Agent message | `{content, agent, role}` |
| `token` | Output token (native streaming) | `{content}` |
| `tool_call` | Function call | `{name, input, agent}` |
| `checkpoint` | State saved | `{turn}` or `{step: "completed"}` |
| `done` | Conversation complete | `{result}` |

## Checkpoint State

The adapter automatically creates checkpoints containing:

```python
{
    "state": {
        "turn": int,           # Current turn number
        "completed": bool,     # Whether conversation finished
    },
    "messages": [              # Full message buffer
        {
            "sender": "assistant",
            "recipient": "user",
            "content": "...",
            "role": "assistant",
        },
    ],
    "context": {"result": {...}},  # Final result (on completion)
}
```

**Checkpoint Triggers:**
- After each conversation turn
- On conversation completion

**Resume Behavior:**
When resuming, the adapter:
1. Restores message buffer from checkpoint
2. Sets `_is_resumed = True` on the run context
3. Emits a `NODE_START` event with `name: "__resume__"`

## Streaming Modes

### Native Streaming (AutoGen v0.4+)

Uses native streaming APIs when available:

```python
# Two-agent: on_messages_stream()
async for response in recipient.on_messages_stream(messages):
    if hasattr(response, "content"):
        yield TOKEN_event(response.content)

# Group chat: run_stream()
async for item in manager.run_stream(task=message):
    yield TOKEN_event(item.content)
```

### Polling-Based Streaming (Fallback)

For older versions, uses message buffer polling:

```python
# Polls message buffer every 100ms
# Yields MESSAGE events for each new message
```

## Common Patterns

### Coding Assistant

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    "coder",
    llm_config={"model": "gpt-4"},
    system_message="You are a helpful coding assistant.",
)

user_proxy = UserProxyAgent(
    "user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": True},
)

@agent_endpoint(
    path="/code",
    runnable=AutoGenAdapter(initiator=user_proxy, recipient=assistant, max_turns=10),
    stream=True,
    durable=True,
)
async def code_assist(task: str) -> str:
    pass
```

### Research Team

```python
from autogen import GroupChat, GroupChatManager

researcher = AssistantAgent("researcher", llm_config=config)
analyst = AssistantAgent("analyst", llm_config=config)
writer = AssistantAgent("writer", llm_config=config)

group_chat = GroupChat(
    agents=[researcher, analyst, writer],
    messages=[],
    max_round=15,
)

user_proxy = UserProxyAgent("user", human_input_mode="NEVER")

@agent_endpoint(
    path="/research",
    runnable=AutoGenAdapter(
        initiator=user_proxy,
        recipient=researcher,
        group_chat=group_chat,
    ),
    stream=True,
    durable=True,
)
async def research(topic: str) -> str:
    pass
```

### With Human Feedback

```python
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="ALWAYS",  # Requires human input
    code_execution_config=False,
)

@agent_endpoint(
    path="/interactive",
    runnable=AutoGenAdapter(initiator=user_proxy, recipient=assistant),
    durable=True,  # Checkpoint allows resuming after human input
)
async def interactive_chat(message: str) -> str:
    pass

# When human input needed:
# 1. Checkpoint saved
# 2. Run pauses
# 3. Resume with: POST /interactive/{run_id}/resume
```

### Function Calling

```python
from autogen import register_function

def search(query: str) -> str:
    return web_search(query)

assistant = AssistantAgent("assistant", llm_config=config)
user_proxy = UserProxyAgent("user", human_input_mode="NEVER")

register_function(
    search,
    caller=assistant,
    executor=user_proxy,
    name="search",
    description="Search the web",
)

@agent_endpoint(
    path="/search",
    runnable=AutoGenAdapter(initiator=user_proxy, recipient=assistant),
    stream=True,
)
async def search_agent(query: str) -> str:
    pass

# Events: message → tool_call → tool_result → message → done
```

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [CrewAI Adapter](crewai.md) - For role-based agents
- [SemanticKernel Adapter](semantic_kernel.md) - For Microsoft ecosystem
