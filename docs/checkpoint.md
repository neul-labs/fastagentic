# Distributed Checkpointing

FastAgentic provides distributed checkpointing for state persistence and recovery across failures and restarts.

## Overview

- **Checkpoints** - Snapshot agent state at any point
- **Checkpoint Manager** - Create and restore checkpoints
- **Multiple Stores** - In-memory, file, Redis, S3 backends
- **TTL Support** - Automatic cleanup of expired checkpoints

## Quick Start

```python
from fastagentic import (
    CheckpointManager,
    CheckpointConfig,
    InMemoryCheckpointStore,
)

# Create store and manager
store = InMemoryCheckpointStore()
manager = CheckpointManager(
    store,
    config=CheckpointConfig(
        auto_checkpoint=True,
        max_checkpoints_per_run=10,
    ),
)

# Create a checkpoint
checkpoint = await manager.create(
    run_id="run-123",
    state={"step": 1, "data": {"key": "value"}},
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
    step_name="processing",
)

# Restore latest checkpoint
checkpoint = await manager.restore("run-123")
if checkpoint:
    state = checkpoint.state
    messages = checkpoint.messages

# Or restore specific checkpoint
checkpoint = await manager.restore_from(checkpoint_id)
```

## Checkpoint Structure

```python
from fastagentic import Checkpoint, CheckpointMetadata

# Checkpoint contains:
checkpoint = Checkpoint(
    metadata=CheckpointMetadata(
        run_id="run-123",
        checkpoint_id="ckpt-abc123",
        sequence=5,
        status=CheckpointStatus.ACTIVE,
        step_name="processing",
        tags=["important"],
        ttl_seconds=3600,
    ),
    state={"step": 5, "progress": 0.5},
    messages=[...],      # Conversation history
    tool_calls=[...],    # Tool calls made
    context={"user_id": "..."}, # Additional context
)

# Access properties
print(checkpoint.checkpoint_id)
print(checkpoint.run_id)
print(checkpoint.metadata.sequence)
print(checkpoint.metadata.is_expired)
```

### Checkpoint Status

```python
class CheckpointStatus(str, Enum):
    ACTIVE = "active"        # Run in progress
    COMPLETED = "completed"  # Run finished successfully
    FAILED = "failed"        # Run failed
    EXPIRED = "expired"      # TTL exceeded
```

## Checkpoint Manager

### Creating Checkpoints

```python
from fastagentic import CheckpointManager, CheckpointConfig

manager = CheckpointManager(store, config=CheckpointConfig(
    auto_checkpoint=True,
    checkpoint_interval=5.0,      # Min seconds between auto-checkpoints
    max_checkpoints_per_run=100,
    default_ttl_seconds=86400,    # 24 hours
))

# Basic checkpoint
checkpoint = await manager.create(
    run_id="run-123",
    state={"step": 1},
)

# Full checkpoint
checkpoint = await manager.create(
    run_id="run-123",
    state={"step": 2, "data": {...}},
    messages=[{"role": "user", "content": "..."}],
    tool_calls=[{"name": "search", "args": {...}}],
    context={"user_id": "user-456"},
    step_name="analysis",
    tags=["retry", "important"],
    ttl_seconds=3600,
    force=True,  # Bypass interval check
)

# With parent (for branching)
child = await manager.create(
    run_id="run-123",
    state={"branch": "A"},
    parent_id=checkpoint.checkpoint_id,
)
```

### Restoring Checkpoints

```python
# Restore latest for a run
checkpoint = await manager.restore("run-123")

# Restore specific checkpoint
checkpoint = await manager.restore_from("ckpt-abc123")

# List all checkpoints for a run
checkpoints = await manager.list_checkpoints("run-123", limit=50)
for meta in checkpoints:
    print(f"{meta.sequence}: {meta.step_name} ({meta.status})")
```

### Managing Checkpoints

```python
# Delete a checkpoint
await manager.delete(checkpoint_id)

# Delete all checkpoints for a run
count = await manager.delete_run("run-123")

# Mark run as completed
await manager.mark_completed("run-123")

# Mark run as failed
await manager.mark_failed("run-123", error="Something went wrong")
```

## Storage Backends

### In-Memory Store

For development and testing:

```python
from fastagentic import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
manager = CheckpointManager(store)
```

### File Store

For single-node deployments:

```python
from fastagentic import FileCheckpointStore

store = FileCheckpointStore("/var/lib/fastagentic/checkpoints")
manager = CheckpointManager(store)
```

Directory structure:
```
/var/lib/fastagentic/checkpoints/
  run-123/
    ckpt-abc123.json
    ckpt-def456.json
  run-456/
    ckpt-ghi789.json
```

### Redis Store

For distributed deployments:

```python
from fastagentic.checkpoint import RedisCheckpointStore
import redis.asyncio as redis

client = redis.Redis(host="localhost", port=6379)
store = RedisCheckpointStore(
    client,
    prefix="fastagentic:checkpoint:",
    ttl_seconds=86400,  # 24 hours
)
manager = CheckpointManager(store)
```

Redis keys:
```
fastagentic:checkpoint:ckpt:{checkpoint_id} -> checkpoint JSON
fastagentic:checkpoint:run:{run_id} -> sorted set of checkpoint IDs
```

### S3 Store

For durable cloud storage:

```python
from fastagentic.checkpoint import S3CheckpointStore
import aioboto3

session = aioboto3.Session()
async with session.client("s3") as s3:
    store = S3CheckpointStore(
        s3,
        bucket="my-checkpoints-bucket",
        prefix="checkpoints/",
    )
    manager = CheckpointManager(store)
```

S3 structure:
```
s3://my-checkpoints-bucket/
  checkpoints/
    run-123/
      ckpt-abc123.json
      _manifest.json
    run-456/
      ckpt-def789.json
      _manifest.json
```

## Configuration

```python
from fastagentic import CheckpointConfig

config = CheckpointConfig(
    auto_checkpoint=True,           # Auto-checkpoint after each step
    checkpoint_interval=0.0,        # Min seconds between checkpoints
    max_checkpoints_per_run=100,    # Prune old checkpoints
    default_ttl_seconds=None,       # Default TTL (None = no expiry)
    cleanup_interval=300.0,         # Cleanup expired every 5 minutes
)
```

## TTL and Cleanup

```python
# Create checkpoint with TTL
checkpoint = await manager.create(
    run_id="run-123",
    state={...},
    ttl_seconds=3600,  # Expires in 1 hour
)

# Check if expired
if checkpoint.metadata.is_expired:
    print("Checkpoint has expired")

# Manual cleanup
count = await store.cleanup_expired()
print(f"Cleaned up {count} expired checkpoints")

# Automatic cleanup (when manager started)
await manager.start()  # Starts cleanup loop
# ... run application ...
await manager.stop()
```

## Serialization

Checkpoints serialize to JSON:

```python
# To JSON
json_str = checkpoint.to_json()

# From JSON
checkpoint = Checkpoint.from_json(json_str)

# To dict
data = checkpoint.to_dict()

# From dict
checkpoint = Checkpoint.from_dict(data)
```

## Integration with Agent Runs

```python
from fastagentic import App, agent_endpoint
from fastagentic.checkpoint import CheckpointManager, FileCheckpointStore

store = FileCheckpointStore("./checkpoints")
checkpoints = CheckpointManager(store)

@agent_endpoint(path="/run")
async def run_agent(request: Request):
    run_id = request.run_id

    # Try to resume from checkpoint
    checkpoint = await checkpoints.restore(run_id)
    if checkpoint:
        state = checkpoint.state
        messages = checkpoint.messages
        start_step = state.get("step", 0)
    else:
        state = {}
        messages = []
        start_step = 0

    try:
        for step in range(start_step, 10):
            # Do work
            result = await process_step(step, state)
            state["step"] = step + 1
            messages.append(result.message)

            # Checkpoint progress
            await checkpoints.create(
                run_id=run_id,
                state=state,
                messages=messages,
                step_name=f"step_{step}",
            )

        # Mark complete
        await checkpoints.mark_completed(run_id)
        return {"status": "complete", "state": state}

    except Exception as e:
        await checkpoints.mark_failed(run_id, error=str(e))
        raise
```

## Best Practices

1. **Checkpoint frequently** - Minimize work lost on failure
2. **Use TTLs** - Prevent unlimited storage growth
3. **Prune old checkpoints** - Set `max_checkpoints_per_run`
4. **Choose appropriate store** - Redis/S3 for distributed, file for single-node
5. **Include context** - Store user_id, session info for debugging
6. **Mark completion** - Always mark runs as completed/failed
7. **Test recovery** - Regularly test restore functionality
8. **Monitor storage** - Alert on storage growth or cleanup failures
