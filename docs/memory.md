# Memory Providers

FastAgentic provides a unified memory abstraction for agent state persistence. Memory providers handle short-term context, long-term recall, and session management.

## Philosophy

| Type | FastAgentic Builds | Integrate Instead |
|------|-------------------|-------------------|
| **Session memory** | Redis (simple) | - |
| **Checkpoints** | Redis/Postgres | - |
| **Long-term memory** | Abstract interface | Mem0, Zep |
| **Semantic search** | None | Mem0, Zep, custom |

FastAgentic provides simple built-in options and interfaces to specialized memory systems.

---

## Memory Types

### Short-Term Memory (Session)

Holds context within a single conversation/session:

```python
from fastagentic import App
from fastagentic.memory import RedisSessionMemory

app = App(
    session_memory=RedisSessionMemory(
        url="redis://localhost:6379",
        ttl_seconds=3600,  # 1 hour
    ),
)
```

### Long-Term Memory (Cross-Session)

Persists user context across sessions:

```python
from fastagentic import App
from fastagentic.integrations.mem0 import Mem0Provider

app = App(
    memory=Mem0Provider(api_key="..."),
)
```

### Checkpoint Memory (Durable Runs)

Stores agent execution state for resume:

```python
from fastagentic import App

app = App(
    durable_store="redis://localhost:6379",  # or postgres://
)
```

---

## Provider Interface

All memory providers implement this interface:

```python
from abc import ABC, abstractmethod
from typing import Any

class MemoryProvider(ABC):
    """Abstract base for memory providers."""

    @abstractmethod
    async def add(
        self,
        user_id: str,
        content: str | list[dict],
        metadata: dict = None,
    ) -> str:
        """
        Add memory content.

        Args:
            user_id: User identifier
            content: Text or messages to memorize
            metadata: Optional metadata (category, tags, etc.)

        Returns:
            Memory ID
        """
        ...

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        filters: dict = None,
    ) -> list[dict]:
        """
        Search memories.

        Args:
            user_id: User identifier
            query: Search query
            limit: Max results
            filters: Optional filters (category, date range, etc.)

        Returns:
            List of memory objects with score
        """
        ...

    @abstractmethod
    async def get(self, user_id: str, memory_id: str) -> dict | None:
        """Get specific memory by ID."""
        ...

    @abstractmethod
    async def get_all(self, user_id: str, limit: int = 100) -> list[dict]:
        """Get all memories for user."""
        ...

    @abstractmethod
    async def update(
        self,
        user_id: str,
        memory_id: str,
        content: str = None,
        metadata: dict = None,
    ) -> None:
        """Update existing memory."""
        ...

    @abstractmethod
    async def delete(self, user_id: str, memory_id: str) -> None:
        """Delete specific memory."""
        ...

    @abstractmethod
    async def delete_all(self, user_id: str) -> int:
        """Delete all memories for user. Returns count deleted."""
        ...
```

---

## Built-in Providers

### RedisProvider

Simple key-value memory without semantic search:

```python
from fastagentic.memory import RedisProvider

memory = RedisProvider(
    url="redis://localhost:6379",
    prefix="memory:",
    ttl_seconds=None,  # No expiration
)
```

**Capabilities:**
- Fast read/write
- Simple key-value storage
- No semantic search
- Good for: Session state, simple recall

**Methods:**

```python
# Add memory (stored as JSON)
memory_id = await memory.add(
    user_id="user_123",
    content="User prefers dark mode",
    metadata={"category": "preferences"},
)

# Get by key pattern
memories = await memory.get_all(user_id="user_123")

# No semantic search - exact match only
# search() returns all memories (no ranking)
```

### PostgresProvider

Relational storage with optional pgvector:

```python
from fastagentic.memory import PostgresProvider

memory = PostgresProvider(
    url="postgres://user:pass@localhost/db",
    table="agent_memories",
    # Optional: Enable vector search with pgvector
    vector_enabled=True,
    embedding_model="text-embedding-3-small",
)
```

**Capabilities:**
- Relational queries
- Full-text search
- Vector search (with pgvector)
- Good for: Enterprise, complex queries

---

## Integrated Providers

### Mem0Provider

Specialized AI memory with automatic extraction:

```python
from fastagentic.integrations.mem0 import Mem0Provider

memory = Mem0Provider(
    api_key="...",
    # Optional: self-hosted config
    config={
        "vector_store": {"provider": "qdrant", ...},
        "llm": {"provider": "openai", ...},
    },
)
```

**Capabilities:**
- Automatic memory extraction from conversations
- Semantic search
- Memory consolidation
- Entity recognition
- Good for: Personalization, long-term user context

See [Mem0 Integration](integrations/mem0.md) for details.

### ZepProvider

Session memory with auto-summarization:

```python
from fastagentic.integrations.zep import ZepProvider

memory = ZepProvider(
    api_key="...",
    # Or self-hosted
    url="http://localhost:8000",
)
```

**Capabilities:**
- Conversation history with summarization
- Semantic search
- Entity extraction
- Temporal awareness
- Good for: Chat applications, session context

---

## Using Memory in Agents

### Automatic Injection

Inject relevant memories into agent context:

```python
from fastagentic import agent_endpoint
from fastagentic.memory import inject_memory

@agent_endpoint(
    path="/chat",
    runnable=...,
    memory=inject_memory(
        provider=Mem0Provider(...),
        strategy="relevant",
        max_memories=10,
    ),
)
async def chat(message: str, ctx: AgentContext) -> str:
    # Memories automatically in ctx.memories
    # and injected into agent's context
    ...
```

### Injection Strategies

```python
# Semantic similarity to current query
inject_memory(strategy="relevant", max_memories=10, min_relevance=0.7)

# Most recent memories
inject_memory(strategy="recent", max_memories=20)

# All memories (use with caution)
inject_memory(strategy="all", max_memories=50)

# Custom strategy
inject_memory(
    strategy="custom",
    selector=lambda ctx, memories: filter_by_category(memories, "important"),
)
```

### Manual Access

```python
@agent_endpoint(path="/chat", runnable=...)
async def chat(message: str, ctx: AgentContext) -> str:
    # Access memory provider directly
    memory = ctx.memory

    # Search for relevant context
    memories = await memory.search(
        user_id=ctx.user.id,
        query=message,
        limit=5,
    )

    # Build context
    memory_context = "\n".join([
        f"- {m['content']}" for m in memories
    ])

    # Run agent with context
    response = await agent.run(
        message,
        context={"memories": memory_context},
    )

    # Store new memories from conversation
    await memory.add(
        user_id=ctx.user.id,
        content=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ],
    )

    return response
```

---

## Memory Hooks

Intercept memory operations:

```python
from fastagentic.hooks import hook, HookContext

@hook("on_memory_add")
async def before_memory_add(ctx: HookContext):
    # Filter sensitive content
    if contains_pii(ctx.memory_content):
        ctx.memory_content = redact_pii(ctx.memory_content)

@hook("on_memory_search")
async def after_memory_search(ctx: HookContext):
    # Log memory access
    await audit_log.record(
        user=ctx.user.id,
        action="memory_search",
        query=ctx.memory_query,
        results_count=len(ctx.memory_results),
    )
```

---

## Session Memory

Separate from long-term memory, session memory handles within-conversation context:

```python
from fastagentic import App
from fastagentic.memory import RedisSessionMemory

app = App(
    # Long-term memory
    memory=Mem0Provider(...),

    # Session memory (separate)
    session_memory=RedisSessionMemory(
        url="redis://localhost:6379",
        ttl_seconds=3600,  # 1 hour
        max_messages=50,   # Keep last 50 messages
    ),
)
```

### Session Memory Interface

```python
class SessionMemory(ABC):
    """Session-scoped conversation memory."""

    @abstractmethod
    async def get_messages(self, session_id: str) -> list[dict]:
        """Get conversation history."""
        ...

    @abstractmethod
    async def add_message(self, session_id: str, message: dict) -> None:
        """Add message to history."""
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear session history."""
        ...

    @abstractmethod
    async def summarize(self, session_id: str) -> str:
        """Get summary of conversation."""
        ...
```

---

## Checkpoint Memory

For durable runs, checkpoint memory stores execution state:

```python
from fastagentic import App

app = App(
    durable_store="redis://localhost:6379",
)

@agent_endpoint(
    path="/long-task",
    runnable=...,
    durable=True,  # Enables checkpointing
)
async def long_task(input: Input) -> Output:
    ...
```

Checkpoints are managed automatically by the durable run system. See [Architecture](architecture.md) for details.

---

## Privacy and Compliance

### User Data Isolation

```python
# Memories are always scoped by user_id
await memory.search(user_id="user_a", query="...")  # Only user_a's memories
await memory.search(user_id="user_b", query="...")  # Only user_b's memories
```

### GDPR Right to Deletion

```python
@app.delete("/users/{user_id}/data")
async def delete_user_data(user_id: str, ctx: AppContext):
    # Delete all memories
    deleted_count = await ctx.memory.delete_all(user_id)

    # Delete session data
    await ctx.session_memory.clear(user_id)

    # Delete checkpoints
    await ctx.checkpoint_store.delete_user(user_id)

    return {"deleted_memories": deleted_count}
```

### Encryption

```python
from fastagentic.memory import EncryptedProvider

memory = EncryptedProvider(
    provider=Mem0Provider(...),
    encryption_key=os.getenv("MEMORY_ENCRYPTION_KEY"),
)
```

---

## Metrics

```
# Memory operations
fastagentic_memory_operations_total{provider="mem0", operation="add"} 1234
fastagentic_memory_operations_total{provider="mem0", operation="search"} 5678
fastagentic_memory_latency_ms{provider="mem0", operation="search", quantile="p99"} 145

# Memory size
fastagentic_memory_count{provider="mem0"} 50234
fastagentic_memory_bytes{provider="redis"} 1048576
```

---

## Choosing a Provider

| Use Case | Recommended Provider |
|----------|---------------------|
| Simple session state | `RedisProvider` |
| Conversation history | `ZepProvider` |
| Long-term personalization | `Mem0Provider` |
| Enterprise/compliance | `PostgresProvider` |
| Semantic search required | `Mem0Provider`, `ZepProvider` |
| Self-hosted requirement | Any (all support self-host) |

---

## Custom Provider

Implement your own:

```python
from fastagentic.memory import MemoryProvider

class MyCustomProvider(MemoryProvider):
    def __init__(self, connection_string: str):
        self.db = connect(connection_string)

    async def add(self, user_id: str, content: str, metadata: dict = None) -> str:
        memory_id = generate_id()
        await self.db.insert({
            "id": memory_id,
            "user_id": user_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        })
        return memory_id

    async def search(self, user_id: str, query: str, limit: int = 10, filters: dict = None) -> list[dict]:
        # Implement your search logic
        ...

    # Implement other methods...
```

---

## Next Steps

- [Mem0 Integration](integrations/mem0.md)
- [Zep Integration](integrations/zep.md)
- [Hooks Architecture](hooks.md)
- [Durability](architecture.md#durable-run-management)
