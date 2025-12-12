# Mem0 Integration

[Mem0](https://mem0.ai) provides persistent memory for AI agents, enabling personalization across sessions. FastAgentic integrates via the `Mem0Provider`.

## Installation

```bash
pip install fastagentic[mem0]
```

## Quick Start

```python
from fastagentic import App
from fastagentic.integrations.mem0 import Mem0Provider

app = App(
    title="My Agent",
    memory=Mem0Provider(api_key="..."),
)
```

## Configuration

### Environment Variables

```bash
export MEM0_API_KEY="..."
```

```python
app = App(memory=Mem0Provider())  # Auto-reads from env
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `api_key` | Mem0 API key | `$MEM0_API_KEY` |
| `org_id` | Organization ID | From API key |
| `project_id` | Project ID | `default` |
| `user_id_field` | Field to extract user ID from context | `user.id` |

## Memory Provider Interface

FastAgentic's memory abstraction:

```python
from fastagentic.memory import MemoryProvider

class Mem0Provider(MemoryProvider):
    async def add(self, user_id: str, messages: list[dict], metadata: dict = None) -> str:
        """Add memories from conversation."""
        ...

    async def search(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        """Search user memories."""
        ...

    async def get_all(self, user_id: str) -> list[dict]:
        """Get all memories for user."""
        ...

    async def delete(self, user_id: str, memory_id: str) -> None:
        """Delete specific memory."""
        ...
```

## Using Memory in Agents

### Automatic Memory Injection

```python
from fastagentic import agent_endpoint
from fastagentic.memory import inject_memory

@agent_endpoint(
    path="/chat",
    runnable=...,
    memory=inject_memory(
        strategy="relevant",  # or "recent", "all"
        max_memories=10,
        min_relevance=0.7,
    ),
)
async def chat(message: str, ctx: AgentContext) -> str:
    # ctx.memories contains relevant memories
    # Automatically injected into agent context
    ...
```

### Manual Memory Access

```python
from fastagentic import agent_endpoint

@agent_endpoint(path="/chat", runnable=...)
async def chat(message: str, ctx: AgentContext) -> str:
    # Search memories
    memories = await ctx.memory.search(
        user_id=ctx.user.id,
        query=message,
        limit=5,
    )

    # Add to conversation context
    memory_context = "\n".join([m["memory"] for m in memories])

    # After response, store new memories
    await ctx.memory.add(
        user_id=ctx.user.id,
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ],
    )

    return response
```

## Memory Strategies

### Relevant (Semantic Search)

```python
inject_memory(
    strategy="relevant",
    max_memories=10,
    min_relevance=0.7,  # Cosine similarity threshold
)
```

Best for: Contextual recall based on current query.

### Recent

```python
inject_memory(
    strategy="recent",
    max_memories=20,
    time_window_hours=24,
)
```

Best for: Short-term context continuity.

### All

```python
inject_memory(
    strategy="all",
    max_memories=50,
)
```

Best for: Small memory sets, comprehensive context.

## Memory Categories

Organize memories by category:

```python
# Store with category
await ctx.memory.add(
    user_id=ctx.user.id,
    messages=[...],
    metadata={"category": "preferences"},
)

# Search by category
memories = await ctx.memory.search(
    user_id=ctx.user.id,
    query="...",
    filters={"category": "preferences"},
)
```

## Session vs Long-Term Memory

```python
from fastagentic.memory import Mem0Provider, RedisProvider

app = App(
    # Long-term memory (Mem0)
    memory=Mem0Provider(api_key="..."),

    # Session memory (Redis) - cleared after session
    session_memory=RedisProvider(
        url="redis://localhost:6379",
        ttl_seconds=3600,  # 1 hour
    ),
)
```

## Self-Hosted Mem0

```python
from mem0 import Memory

# Use self-hosted mem0 with custom config
memory_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
        }
    },
}

app = App(
    memory=Mem0Provider(config=memory_config),
)
```

## Privacy and Data Handling

### User Data Isolation

Memories are automatically isolated by `user_id`:

```python
# User A's memories never leak to User B
memories = await ctx.memory.search(
    user_id="user_a",  # Strictly scoped
    query="...",
)
```

### Memory Deletion (GDPR)

```python
from fastagentic import App

app = App(memory=Mem0Provider(...))

# Delete all memories for user
@app.post("/users/{user_id}/forget")
async def forget_user(user_id: str, ctx: AppContext):
    memories = await ctx.memory.get_all(user_id)
    for m in memories:
        await ctx.memory.delete(user_id, m["id"])
    return {"deleted": len(memories)}
```

### Memory Encryption

```python
Mem0Provider(
    api_key="...",
    encryption_key="...",  # Encrypt at rest
)
```

## Metrics

```
fastagentic_memory_operations_total{provider="mem0", operation="search"} 5234
fastagentic_memory_operations_total{provider="mem0", operation="add"} 1256
fastagentic_memory_latency_ms{provider="mem0", operation="search", quantile="p99"} 145
```

## Alternative: Zep

For session-focused memory with auto-summarization:

```python
from fastagentic.integrations.zep import ZepProvider

app = App(
    memory=ZepProvider(
        api_key="...",
        # Zep automatically summarizes long conversations
        auto_summarize=True,
    ),
)
```

## Alternative: Redis (Simple)

For simple key-value memory without semantic search:

```python
from fastagentic.memory import RedisProvider

app = App(
    memory=RedisProvider(
        url="redis://localhost:6379",
        prefix="memory:",
    ),
)
```

## Troubleshooting

### Memories not being found

- Check `min_relevance` threshold
- Verify user_id is consistent
- Check if memories were added successfully

### High latency

- Reduce `max_memories`
- Use category filters to narrow search
- Consider caching frequent queries

## Next Steps

- [Mem0 Docs](https://docs.mem0.ai)
- [Zep Integration](zep.md)
- [Hooks Architecture](../hooks.md)
