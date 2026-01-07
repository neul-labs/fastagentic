"""Task queue for distributed FastAgentic."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Task priority levels."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class Task:
    """A distributed task.

    Attributes:
        id: Unique task ID
        type: Task type (maps to handler)
        data: Task input data
        priority: Task priority
        status: Current status
        result: Task result (when completed)
        error: Error message (when failed)
        created_at: Creation timestamp
        started_at: When execution started
        completed_at: When execution completed
        timeout: Task timeout in seconds
        retries: Number of retries allowed
        retry_count: Current retry count
        worker_id: ID of worker executing this task
        metadata: Additional metadata
    """

    type: str
    data: dict[str, Any]
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout: float = 300.0
    retries: int = 0
    retry_count: int = 0
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.retries


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    duration_seconds: float | None = None
    worker_id: str | None = None


class TaskStore(Protocol):
    """Protocol for task storage."""

    async def save(self, task: Task) -> None:
        """Save a task."""
        ...

    async def get(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        ...

    async def update(self, task: Task) -> None:
        """Update a task."""
        ...

    async def delete(self, task_id: str) -> None:
        """Delete a task."""
        ...

    async def list_by_status(
        self,
        status: TaskStatus,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks by status."""
        ...


class InMemoryTaskStore:
    """In-memory task store for development/testing."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    async def save(self, task: Task) -> None:
        self._tasks[task.id] = task

    async def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def update(self, task: Task) -> None:
        self._tasks[task.id] = task

    async def delete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    async def list_by_status(
        self,
        status: TaskStatus,
        limit: int = 100,
    ) -> list[Task]:
        results = []
        for task in self._tasks.values():
            if task.status == status:
                results.append(task)
                if len(results) >= limit:
                    break
        return results


class TaskQueue:
    """Distributed task queue.

    Example:
        queue = TaskQueue()

        # Enqueue tasks
        task = await queue.enqueue(
            task_type="process_document",
            data={"document_id": "doc-123"},
            priority=TaskPriority.HIGH,
        )

        # Dequeue for processing
        task = await queue.dequeue(worker_id="worker-1")

        # Complete the task
        await queue.complete(task.id, result={"processed": True})

        # Or fail the task
        await queue.fail(task.id, error="Processing failed")
    """

    def __init__(
        self,
        store: TaskStore | None = None,
        max_queue_size: int = 10000,
    ) -> None:
        """Initialize task queue.

        Args:
            store: Task storage backend
            max_queue_size: Maximum queue size
        """
        self._store = store or InMemoryTaskStore()
        self._max_queue_size = max_queue_size
        self._queue: asyncio.PriorityQueue[tuple[int, float, str]] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._pending_count = 0

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    async def enqueue(
        self,
        task_type: str,
        data: dict[str, Any],
        *,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 300.0,
        retries: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Add a task to the queue.

        Args:
            task_type: Type of task
            data: Task input data
            priority: Task priority
            timeout: Task timeout
            retries: Number of retries allowed
            metadata: Additional metadata

        Returns:
            The created task
        """
        task = Task(
            type=task_type,
            data=data,
            priority=priority,
            status=TaskStatus.QUEUED,
            timeout=timeout,
            retries=retries,
            metadata=metadata or {},
        )

        await self._store.save(task)

        # Add to priority queue (lower priority value = higher priority)
        # Use negative priority so higher priority tasks come first
        await self._queue.put((-priority.value, task.created_at, task.id))
        self._pending_count += 1

        return task

    async def dequeue(
        self,
        worker_id: str,
        timeout: float | None = None,
    ) -> Task | None:
        """Get the next task from the queue.

        Args:
            worker_id: ID of worker requesting task
            timeout: How long to wait for a task

        Returns:
            Next task or None if timeout
        """
        try:
            if timeout is not None:
                _, _, task_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
            else:
                _, _, task_id = await self._queue.get()

            task = await self._store.get(task_id)
            if task is None:
                return None

            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.worker_id = worker_id
            await self._store.update(task)

            self._pending_count -= 1
            return task

        except asyncio.TimeoutError:
            return None

    async def complete(
        self,
        task_id: str,
        result: Any = None,
    ) -> TaskResult:
        """Mark a task as completed.

        Args:
            task_id: Task ID
            result: Task result

        Returns:
            TaskResult
        """
        task = await self._store.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = time.time()
        await self._store.update(task)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            duration_seconds=task.duration_seconds,
            worker_id=task.worker_id,
        )

    async def fail(
        self,
        task_id: str,
        error: str,
        *,
        retry: bool = True,
    ) -> TaskResult:
        """Mark a task as failed.

        Args:
            task_id: Task ID
            error: Error message
            retry: Whether to retry if possible

        Returns:
            TaskResult
        """
        task = await self._store.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        # Check if we should retry
        if retry and task.can_retry:
            task.retry_count += 1
            task.status = TaskStatus.QUEUED
            task.error = error
            task.started_at = None
            task.worker_id = None
            await self._store.update(task)

            # Re-queue
            await self._queue.put((-task.priority.value, time.time(), task.id))
            self._pending_count += 1

            return TaskResult(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                error=f"Retry {task.retry_count}/{task.retries}: {error}",
            )

        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = time.time()
        await self._store.update(task)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=error,
            duration_seconds=task.duration_seconds,
            worker_id=task.worker_id,
        )

    async def cancel(self, task_id: str) -> TaskResult:
        """Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            TaskResult
        """
        task = await self._store.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task.is_complete:
            return TaskResult(
                task_id=task_id,
                status=task.status,
                error="Task already complete",
            )

        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        await self._store.update(task)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
        )

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None
        """
        return await self._store.get(task_id)

    async def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
        poll_interval: float = 0.5,
    ) -> TaskResult:
        """Wait for a task to complete.

        Args:
            task_id: Task ID
            timeout: Maximum wait time
            poll_interval: Polling interval

        Returns:
            TaskResult

        Raises:
            TimeoutError: If timeout exceeded
        """
        deadline = time.time() + timeout if timeout else None

        while True:
            task = await self._store.get(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            if task.is_complete:
                return TaskResult(
                    task_id=task_id,
                    status=task.status,
                    result=task.result,
                    error=task.error,
                    duration_seconds=task.duration_seconds,
                    worker_id=task.worker_id,
                )

            if deadline and time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for task: {task_id}")

            await asyncio.sleep(poll_interval)

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": self.size,
            "pending_count": self._pending_count,
            "max_size": self._max_queue_size,
        }
