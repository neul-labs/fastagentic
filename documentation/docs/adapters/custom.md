# Building Custom Adapters

Create adapters for proprietary frameworks or unique requirements by implementing the `BaseAdapter` interface.

## The Runnable Interface

```python
from fastagentic.adapters.base import BaseAdapter, Event, EventType
from typing import AsyncIterator, Any

class MyAdapter(BaseAdapter):
    """Adapter for my custom agent framework."""

    def __init__(self, agent: MyAgent, **options):
        self.agent = agent
        self.options = options

    async def invoke(self, input: dict, config: dict) -> dict:
        """Synchronous execution - return final result."""
        result = await self.agent.run(input)
        return {"output": result}

    async def stream(self, input: dict, config: dict) -> AsyncIterator[Event]:
        """Streaming execution - yield events as they occur."""
        async for chunk in self.agent.stream(input):
            yield self.create_event(EventType.TOKEN, {"content": chunk})

        # Final result
        result = await self.agent.get_result()
        yield self.create_event(EventType.RUN_COMPLETE, {"result": result})

    async def checkpoint(self, state: dict) -> str:
        """Save state, return checkpoint ID."""
        checkpoint_id = str(uuid4())
        await self.store.save(checkpoint_id, state)
        return checkpoint_id

    async def resume(self, checkpoint_id: str) -> dict:
        """Restore from checkpoint, return state."""
        return await self.store.load(checkpoint_id)

    def get_schema(self) -> dict:
        """Return JSON schema for MCP tool registration."""
        return {
            "name": "my_agent",
            "description": self.agent.description,
            "inputSchema": self.agent.input_schema,
        }
```

## Event Protocol

### Event Types

```python
from enum import Enum

class EventType(Enum):
    TOKEN = "token"              # LLM output token
    NODE_START = "node_start"    # Workflow node begins
    NODE_END = "node_end"        # Workflow node completes
    TOOL_CALL = "tool_call"      # Tool invocation
    TOOL_RESULT = "tool_result"  # Tool returns
    CHECKPOINT = "checkpoint"    # State persisted
    COST = "cost"                # Usage metrics
    RUN_COMPLETE = "run_complete"# Execution finished
    ERROR = "error"              # Error occurred
```

### Event Structure

```python
@dataclass
class Event:
    type: EventType
    data: dict
    timestamp: float
    run_id: str | None = None
    metadata: dict | None = None
```

### Creating Events

Use the helper method:

```python
# Simple event
yield self.create_event(EventType.TOKEN, {"content": "Hello"})

# With metadata
yield self.create_event(
    EventType.TOOL_CALL,
    {"tool": "search", "args": {"query": "AI"}},
    metadata={"latency_ms": 150}
)
```

## Lifecycle Hooks

Override these methods for custom behavior:

```python
class MyAdapter(BaseAdapter):

    async def on_start(self, input: dict, config: dict) -> None:
        """Called before execution begins."""
        self.start_time = time.time()

    async def on_complete(self, result: dict, config: dict) -> None:
        """Called after successful completion."""
        duration = time.time() - self.start_time
        logger.info(f"Completed in {duration:.2f}s")

    async def on_error(self, error: Exception, config: dict) -> None:
        """Called when execution fails."""
        logger.error(f"Failed: {error}")

    async def on_checkpoint(self, checkpoint_id: str, state: dict) -> None:
        """Called after checkpoint is saved."""
        logger.debug(f"Checkpoint saved: {checkpoint_id}")
```

## Complete Example: AutoGen Adapter

```python
from fastagentic.adapters.base import BaseAdapter, Event, EventType
from autogen import AssistantAgent, UserProxyAgent
from typing import AsyncIterator
import asyncio

class AutoGenAdapter(BaseAdapter):
    """Adapter for Microsoft AutoGen agents."""

    def __init__(
        self,
        assistant: AssistantAgent,
        user_proxy: UserProxyAgent,
        max_rounds: int = 10,
    ):
        self.assistant = assistant
        self.user_proxy = user_proxy
        self.max_rounds = max_rounds
        self._events: list[Event] = []

    async def invoke(self, input: dict, config: dict) -> dict:
        """Run AutoGen conversation synchronously."""
        message = input.get("message", "")

        # Collect events during execution
        self._events = []

        # Run the conversation
        await self.user_proxy.a_initiate_chat(
            self.assistant,
            message=message,
            max_turns=self.max_rounds,
        )

        # Get final response
        last_message = self.assistant.last_message()
        return {"response": last_message["content"]}

    async def stream(self, input: dict, config: dict) -> AsyncIterator[Event]:
        """Stream AutoGen conversation events."""
        message = input.get("message", "")

        # Set up message callback
        async def on_message(sender, recipient, message):
            yield self.create_event(
                EventType.TOKEN,
                {
                    "content": message.get("content", ""),
                    "sender": sender.name,
                    "recipient": recipient.name,
                }
            )

        # Register callback
        self.assistant.register_reply(
            trigger=self.user_proxy,
            reply_func=on_message,
        )

        # Emit start event
        yield self.create_event(
            EventType.NODE_START,
            {"node": "conversation", "message": message}
        )

        # Run conversation
        await self.user_proxy.a_initiate_chat(
            self.assistant,
            message=message,
            max_turns=self.max_rounds,
        )

        # Emit completion
        last_message = self.assistant.last_message()
        yield self.create_event(
            EventType.RUN_COMPLETE,
            {"result": last_message["content"]}
        )

    async def checkpoint(self, state: dict) -> str:
        """Save conversation state."""
        checkpoint_id = str(uuid4())
        checkpoint_data = {
            "assistant_messages": self.assistant.chat_messages,
            "user_proxy_messages": self.user_proxy.chat_messages,
            "state": state,
        }
        await self.store.save(checkpoint_id, checkpoint_data)
        return checkpoint_id

    async def resume(self, checkpoint_id: str) -> dict:
        """Restore conversation from checkpoint."""
        data = await self.store.load(checkpoint_id)
        self.assistant.chat_messages = data["assistant_messages"]
        self.user_proxy.chat_messages = data["user_proxy_messages"]
        return data["state"]

    def get_schema(self) -> dict:
        """MCP schema for AutoGen agent."""
        return {
            "name": "autogen_chat",
            "description": f"Chat with {self.assistant.name}",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send"
                    }
                },
                "required": ["message"]
            }
        }
```

## Usage

```python
from fastagentic import App, agent_endpoint
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={...})
user_proxy = UserProxyAgent("user", code_execution_config={...})

adapter = AutoGenAdapter(assistant, user_proxy, max_rounds=5)

app = App(title="AutoGen Chat", ...)

@agent_endpoint(
    path="/chat",
    runnable=adapter,
    stream=True,
    durable=True,
)
async def chat(message: str) -> str:
    pass
```

## Testing Custom Adapters

```python
import pytest
from fastagentic.testing import AdapterTestCase

class TestMyAdapter(AdapterTestCase):
    adapter_class = MyAdapter

    @pytest.fixture
    def adapter(self):
        return MyAdapter(mock_agent)

    async def test_invoke(self, adapter):
        result = await adapter.invoke({"input": "test"}, {})
        assert "output" in result

    async def test_stream_emits_events(self, adapter):
        events = [e async for e in adapter.stream({"input": "test"}, {})]
        assert any(e.type == EventType.TOKEN for e in events)
        assert events[-1].type == EventType.RUN_COMPLETE

    async def test_checkpoint_resume(self, adapter):
        # Run partially
        state = {"step": 1, "data": "partial"}
        checkpoint_id = await adapter.checkpoint(state)

        # Resume
        restored = await adapter.resume(checkpoint_id)
        assert restored == state
```

## Best Practices

1. **Always emit RUN_COMPLETE**: Ensure streaming ends with a completion event

2. **Include timestamps**: Use `self.create_event()` which adds timestamps automatically

3. **Handle errors gracefully**: Emit ERROR events and re-raise for proper handling

4. **Checkpoint strategically**: Save state at meaningful boundaries, not every token

5. **Test thoroughly**: Use `AdapterTestCase` base class for consistent testing

## Next Steps

- [Adapters Overview](index.md) - Compare with built-in adapters
- [PydanticAI Adapter](pydanticai.md) - Reference implementation
- [Operations Guide](../operations/index.md) - Deploy custom adapters
