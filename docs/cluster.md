# Cluster Orchestration

FastAgentic provides built-in cluster orchestration for distributing agent workloads across multiple workers.

## Overview

- **Workers** - Execute tasks with configurable concurrency
- **Worker Pools** - Manage groups of workers with auto-scaling
- **Task Queue** - Priority queue with retry support
- **Coordinator** - Orchestrate distributed workloads

## Quick Start

```python
from fastagentic import (
    Coordinator,
    CoordinatorConfig,
    TaskPriority,
)

# Create coordinator
coordinator = Coordinator(
    config=CoordinatorConfig(
        min_workers=2,
        max_workers=10,
        auto_scale=True,
    ),
)

# Register task handlers
@coordinator.handler("process_document")
async def process_document(data: dict) -> dict:
    # Process the document
    return {"processed": True, "pages": data["page_count"]}

@coordinator.handler("generate_summary")
async def generate_summary(data: dict) -> dict:
    # Generate summary
    return {"summary": "..."}

# Start the coordinator
await coordinator.start()

# Submit tasks
result = await coordinator.submit(
    "process_document",
    {"document_id": "doc-123", "page_count": 10},
)

# Or submit async and wait later
task = await coordinator.submit_async(
    "generate_summary",
    {"text": "..."},
    priority=TaskPriority.HIGH,
)
result = await coordinator.wait_for(task.id)

# Stop gracefully
await coordinator.stop()
```

## Workers

Individual task executors:

```python
from fastagentic import Worker, WorkerConfig

# Create worker with config
worker = Worker(
    id="worker-1",
    config=WorkerConfig(
        max_concurrent_tasks=4,
        heartbeat_interval=5.0,
        task_timeout=300.0,
        tags=["gpu", "fast"],
    ),
)

# Register handlers
@worker.handler("compute")
async def compute(data: dict) -> dict:
    return {"result": data["x"] * 2}

# Start worker
await worker.start()

# Execute task directly
result = await worker.execute_task(
    task_id="task-123",
    task_type="compute",
    data={"x": 21},
)

# Stop worker
await worker.stop(graceful=True)
```

### Worker Status

```python
class WorkerStatus(str, Enum):
    IDLE = "idle"          # Ready for tasks
    BUSY = "busy"          # Processing tasks
    DRAINING = "draining"  # Finishing current, no new tasks
    STOPPED = "stopped"    # Not running
    ERROR = "error"        # Error state
```

### Worker Info

```python
info = worker.info
print(f"ID: {info.id}")
print(f"Status: {info.status}")
print(f"Current tasks: {info.current_tasks}")
print(f"Total completed: {info.total_completed}")
print(f"Total failed: {info.total_failed}")
print(f"Uptime: {info.uptime_seconds}s")
print(f"Available: {info.is_available}")
```

## Worker Pools

Manage groups of workers:

```python
from fastagentic import WorkerPool, WorkerConfig

pool = WorkerPool(
    min_workers=2,
    max_workers=10,
    config=WorkerConfig(max_concurrent_tasks=4),
)

# Register handlers for all workers
pool.register_handler("process", process_handler)
pool.register_handler("analyze", analyze_handler)

# Start pool (creates min_workers)
await pool.start()

# Submit task to pool
result = await pool.submit("process", {"data": "..."})

# Scale pool
await pool.scale(5)  # Scale to 5 workers

# Get stats
stats = pool.get_stats()
print(f"Workers: {stats['worker_count']}")
print(f"Current tasks: {stats['current_tasks']}")
print(f"Completed: {stats['total_completed']}")

# Stop pool
await pool.stop(graceful=True)
```

## Task Queue

Priority-based task queue with retry support:

```python
from fastagentic import TaskQueue, TaskPriority

queue = TaskQueue(max_queue_size=10000)

# Enqueue tasks with priority
task = await queue.enqueue(
    task_type="analyze",
    data={"document": "..."},
    priority=TaskPriority.HIGH,
    timeout=60.0,
    retries=3,
    metadata={"user_id": "user-123"},
)

# Dequeue for processing
task = await queue.dequeue(worker_id="worker-1", timeout=5.0)

# Complete task
result = await queue.complete(task.id, result={"analysis": "..."})

# Or fail task (will retry if retries remaining)
result = await queue.fail(task.id, error="Processing failed")

# Cancel task
result = await queue.cancel(task.id)

# Wait for task completion
result = await queue.wait_for_task(task.id, timeout=30.0)
```

### Task Priority

```python
class TaskPriority(int, Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20
```

### Task Status

```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
```

## Coordinator

High-level orchestration with auto-scaling:

```python
from fastagentic import Coordinator, CoordinatorConfig

coordinator = Coordinator(
    config=CoordinatorConfig(
        min_workers=2,
        max_workers=20,
        scale_up_threshold=10,    # Queue size to trigger scale up
        scale_down_threshold=2,   # Queue size to trigger scale down
        health_check_interval=10.0,
        worker_timeout=30.0,
        auto_scale=True,
    ),
)

# Register handlers
coordinator.register_handler("task_type", handler_function)

# Or use decorator
@coordinator.handler("process")
async def process(data: dict) -> dict:
    return {"result": "..."}

# Start coordinator
await coordinator.start()

# Submit and wait
result = await coordinator.submit("process", {"input": "..."})

# Submit async
task = await coordinator.submit_async(
    "process",
    {"input": "..."},
    priority=TaskPriority.HIGH,
    timeout=60.0,
    retries=2,
)

# Wait for result
result = await coordinator.wait_for(task.id)

# Get task status
task = await coordinator.get_task(task.id)
print(f"Status: {task.status}")

# Cancel task
result = await coordinator.cancel_task(task.id)

# Manual scaling
await coordinator.scale(5)

# Get workers
workers = await coordinator.get_workers()

# Get stats
stats = coordinator.get_stats()
print(f"Running: {stats['running']}")
print(f"Queue size: {stats['queue']['queue_size']}")
print(f"Worker count: {stats['workers']['worker_count']}")

# Stop coordinator
await coordinator.stop(graceful=True)
```

### Auto-Scaling

The coordinator automatically scales workers based on queue depth:

```python
config = CoordinatorConfig(
    min_workers=2,           # Always keep at least 2 workers
    max_workers=20,          # Never exceed 20 workers
    scale_up_threshold=10,   # Add worker when queue > 10
    scale_down_threshold=2,  # Remove worker when queue < 2
    auto_scale=True,
)
```

### Health Checks

Workers are monitored via heartbeats:

```python
config = CoordinatorConfig(
    health_check_interval=10.0,  # Check every 10 seconds
    worker_timeout=30.0,         # Consider dead after 30s no heartbeat
)
```

Dead workers are automatically removed and replaced.

## Worker Registry

Track workers across the cluster:

```python
from fastagentic.cluster import InMemoryWorkerRegistry

registry = InMemoryWorkerRegistry()

# Register worker
await registry.register(worker.info)

# Heartbeat
await registry.heartbeat(worker.id)

# Get workers
workers = await registry.get_workers(
    status=WorkerStatus.IDLE,
    tags=["gpu"],
)

# Unregister
await registry.unregister(worker.id)
```

## Integration with App

```python
from fastagentic import App
from fastagentic.cluster import Coordinator, CoordinatorConfig

app = App(title="Distributed Agent")

coordinator = Coordinator(
    config=CoordinatorConfig(min_workers=4),
)

@coordinator.handler("agent_task")
async def agent_task(data: dict) -> dict:
    # Run agent logic
    return await run_agent(data)

@app.on_startup
async def startup():
    await coordinator.start()

@app.on_shutdown
async def shutdown():
    await coordinator.stop()

@app.agent_endpoint(path="/run")
async def run_endpoint(request: Request) -> dict:
    result = await coordinator.submit("agent_task", request.data)
    return result
```

## Best Practices

1. **Set appropriate timeouts** - Tasks should have realistic timeouts
2. **Use retries wisely** - Only retry idempotent operations
3. **Monitor queue depth** - Alert when queue grows too large
4. **Graceful shutdown** - Always stop with `graceful=True` in production
5. **Tag workers** - Use tags for task routing (GPU, memory, etc.)
6. **Health checks** - Monitor worker health and restart unhealthy workers
7. **Capacity planning** - Set min/max workers based on expected load
