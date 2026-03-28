"""Worker management for distributed FastAgentic."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class WorkerStatus(str, Enum):
    """Worker status states."""

    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"  # No new tasks, finishing current
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerConfig:
    """Configuration for a worker.

    Attributes:
        max_concurrent_tasks: Maximum tasks to run concurrently
        heartbeat_interval: Seconds between heartbeats
        task_timeout: Default task timeout in seconds
        graceful_shutdown_timeout: Seconds to wait during shutdown
        tags: Worker tags for task routing
    """

    max_concurrent_tasks: int = 1
    heartbeat_interval: float = 5.0
    task_timeout: float = 300.0
    graceful_shutdown_timeout: float = 30.0
    tags: list[str] = field(default_factory=list)


@dataclass
class WorkerInfo:
    """Information about a worker.

    Attributes:
        id: Unique worker ID
        status: Current status
        current_tasks: Number of tasks currently running
        total_completed: Total tasks completed
        total_failed: Total tasks failed
        last_heartbeat: Last heartbeat timestamp
        started_at: When the worker started
        config: Worker configuration
        metadata: Additional metadata
    """

    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_tasks: int = 0
    total_completed: int = 0
    total_failed: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.time)
    config: WorkerConfig = field(default_factory=WorkerConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if worker can accept tasks."""
        if self.status not in (WorkerStatus.IDLE, WorkerStatus.BUSY):
            return False
        return self.current_tasks < self.config.max_concurrent_tasks

    @property
    def uptime_seconds(self) -> float:
        """Get worker uptime in seconds."""
        return time.time() - self.started_at


# Task handler type
TaskHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class Worker:
    """A worker that processes tasks.

    Example:
        worker = Worker(
            id="worker-1",
            config=WorkerConfig(max_concurrent_tasks=4),
        )

        # Register task handlers
        @worker.handler("process_document")
        async def process_document(data: dict) -> dict:
            # Process the document
            return {"processed": True}

        # Start the worker
        await worker.start()
    """

    def __init__(
        self,
        id: str | None = None,
        config: WorkerConfig | None = None,
    ) -> None:
        """Initialize worker.

        Args:
            id: Worker ID (auto-generated if None)
            config: Worker configuration
        """
        self.id = id or f"worker-{uuid.uuid4().hex[:8]}"
        self.config = config or WorkerConfig()
        self._handlers: dict[str, TaskHandler] = {}
        self._info = WorkerInfo(id=self.id, config=self.config)
        self._running = False
        self._task_semaphore: asyncio.Semaphore | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._on_task_complete: list[Callable[[str, Any], Awaitable[None]]] = []
        self._on_task_error: list[Callable[[str, Exception], Awaitable[None]]] = []

    @property
    def info(self) -> WorkerInfo:
        """Get worker info."""
        return self._info

    @property
    def status(self) -> WorkerStatus:
        """Get worker status."""
        return self._info.status

    def handler(self, task_type: str) -> Callable[[TaskHandler], TaskHandler]:
        """Decorator to register a task handler.

        Args:
            task_type: Type of task this handler processes

        Returns:
            Decorator function
        """

        def decorator(func: TaskHandler) -> TaskHandler:
            self._handlers[task_type] = func
            return func

        return decorator

    def register_handler(self, task_type: str, handler: TaskHandler) -> None:
        """Register a task handler.

        Args:
            task_type: Type of task
            handler: Handler function
        """
        self._handlers[task_type] = handler

    def on_task_complete(
        self,
        callback: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        """Register callback for task completion.

        Args:
            callback: Async function(task_id, result)
        """
        self._on_task_complete.append(callback)

    def on_task_error(
        self,
        callback: Callable[[str, Exception], Awaitable[None]],
    ) -> None:
        """Register callback for task errors.

        Args:
            callback: Async function(task_id, error)
        """
        self._on_task_error.append(callback)

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            return

        self._running = True
        self._task_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._info.status = WorkerStatus.IDLE
        self._info.started_at = time.time()

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self, graceful: bool = True) -> None:
        """Stop the worker.

        Args:
            graceful: Wait for current tasks to complete
        """
        if not self._running:
            return

        self._info.status = WorkerStatus.DRAINING if graceful else WorkerStatus.STOPPED
        self._running = False

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if graceful:
            # Wait for current tasks (with timeout)
            deadline = time.time() + self.config.graceful_shutdown_timeout
            while self._info.current_tasks > 0 and time.time() < deadline:
                await asyncio.sleep(0.1)

        self._info.status = WorkerStatus.STOPPED

    async def execute_task(
        self,
        task_id: str,
        task_type: str,
        data: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Execute a task.

        Args:
            task_id: Task identifier
            task_type: Type of task
            data: Task data
            timeout: Task timeout (uses config default if None)

        Returns:
            Task result

        Raises:
            ValueError: If no handler for task type
            asyncio.TimeoutError: If task times out
        """
        handler = self._handlers.get(task_type)
        if handler is None:
            raise ValueError(f"No handler for task type: {task_type}")

        if not self._running:
            raise RuntimeError("Worker is not running")

        timeout = timeout or self.config.task_timeout

        async with self._task_semaphore:  # type: ignore
            self._info.current_tasks += 1
            self._update_status()

            try:
                result = await asyncio.wait_for(handler(data), timeout=timeout)
                self._info.total_completed += 1

                # Notify callbacks
                for callback in self._on_task_complete:
                    try:
                        await callback(task_id, result)
                    except Exception:
                        pass

                return result

            except Exception as e:
                self._info.total_failed += 1

                # Notify callbacks
                for callback in self._on_task_error:
                    try:
                        await callback(task_id, e)
                    except Exception:
                        pass

                raise

            finally:
                self._info.current_tasks -= 1
                self._update_status()

    def _update_status(self) -> None:
        """Update worker status based on current tasks."""
        if self._info.status == WorkerStatus.DRAINING:
            return
        if self._info.status == WorkerStatus.STOPPED:
            return

        if self._info.current_tasks > 0:
            self._info.status = WorkerStatus.BUSY
        else:
            self._info.status = WorkerStatus.IDLE

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            self._info.last_heartbeat = time.time()
            await asyncio.sleep(self.config.heartbeat_interval)


class WorkerRegistry(Protocol):
    """Protocol for worker registry."""

    async def register(self, info: WorkerInfo) -> None:
        """Register a worker."""
        ...

    async def unregister(self, worker_id: str) -> None:
        """Unregister a worker."""
        ...

    async def heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat."""
        ...

    async def get_workers(
        self,
        status: WorkerStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[WorkerInfo]:
        """Get workers matching criteria."""
        ...


class InMemoryWorkerRegistry:
    """In-memory worker registry for development/testing."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}

    async def register(self, info: WorkerInfo) -> None:
        self._workers[info.id] = info

    async def unregister(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    async def heartbeat(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].last_heartbeat = time.time()

    async def get_workers(
        self,
        status: WorkerStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[WorkerInfo]:
        results = []
        for info in self._workers.values():
            if status and info.status != status:
                continue
            if tags:
                worker_tags = set(info.config.tags)
                if not worker_tags.intersection(tags):
                    continue
            results.append(info)
        return results


class WorkerPool:
    """Pool of workers for task execution.

    Example:
        pool = WorkerPool(min_workers=2, max_workers=10)

        # Register handlers for the pool
        pool.register_handler("process", process_handler)

        # Start the pool
        await pool.start()

        # Submit tasks
        result = await pool.submit("process", {"data": "..."})

        # Scale the pool
        await pool.scale(5)

        # Stop the pool
        await pool.stop()
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        config: WorkerConfig | None = None,
        registry: WorkerRegistry | None = None,
    ) -> None:
        """Initialize worker pool.

        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            config: Default worker configuration
            registry: Worker registry for coordination
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.config = config or WorkerConfig()
        self._registry = registry or InMemoryWorkerRegistry()
        self._workers: list[Worker] = []
        self._handlers: dict[str, TaskHandler] = {}
        self._running = False

    @property
    def worker_count(self) -> int:
        """Get current number of workers."""
        return len(self._workers)

    def register_handler(self, task_type: str, handler: TaskHandler) -> None:
        """Register a task handler for all workers.

        Args:
            task_type: Type of task
            handler: Handler function
        """
        self._handlers[task_type] = handler
        # Register on existing workers
        for worker in self._workers:
            worker.register_handler(task_type, handler)

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._running = True

        # Create minimum workers
        for _ in range(self.min_workers):
            await self._create_worker()

    async def stop(self, graceful: bool = True) -> None:
        """Stop all workers.

        Args:
            graceful: Wait for current tasks to complete
        """
        if not self._running:
            return

        self._running = False

        # Stop all workers concurrently
        await asyncio.gather(
            *[w.stop(graceful=graceful) for w in self._workers],
            return_exceptions=True,
        )

        # Unregister from registry
        for worker in self._workers:
            await self._registry.unregister(worker.id)

        self._workers.clear()

    async def scale(self, target: int) -> None:
        """Scale to target number of workers.

        Args:
            target: Target worker count
        """
        target = max(self.min_workers, min(target, self.max_workers))

        current = len(self._workers)

        if target > current:
            # Scale up
            for _ in range(target - current):
                await self._create_worker()

        elif target < current:
            # Scale down (stop excess workers)
            to_stop = self._workers[target:]
            self._workers = self._workers[:target]

            await asyncio.gather(
                *[w.stop(graceful=True) for w in to_stop],
                return_exceptions=True,
            )

            for worker in to_stop:
                await self._registry.unregister(worker.id)

    async def submit(
        self,
        task_type: str,
        data: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Submit a task to the pool.

        Args:
            task_type: Type of task
            data: Task data
            timeout: Task timeout

        Returns:
            Task result
        """
        if not self._running:
            raise RuntimeError("Worker pool is not running")

        # Find available worker
        worker = self._find_available_worker()
        if worker is None:
            # All workers busy - wait for one
            worker = await self._wait_for_worker()

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        return await worker.execute_task(task_id, task_type, data, timeout)

    def _find_available_worker(self) -> Worker | None:
        """Find an available worker."""
        for worker in self._workers:
            if worker.info.is_available:
                return worker
        return None

    async def _wait_for_worker(self, timeout: float = 30.0) -> Worker:
        """Wait for an available worker."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            worker = self._find_available_worker()
            if worker:
                return worker
            await asyncio.sleep(0.1)
        raise TimeoutError("No available workers")

    async def _create_worker(self) -> Worker:
        """Create and start a new worker."""
        worker = Worker(config=self.config)

        # Register all handlers
        for task_type, handler in self._handlers.items():
            worker.register_handler(task_type, handler)

        await worker.start()
        await self._registry.register(worker.info)

        self._workers.append(worker)
        return worker

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        total_completed = sum(w.info.total_completed for w in self._workers)
        total_failed = sum(w.info.total_failed for w in self._workers)
        current_tasks = sum(w.info.current_tasks for w in self._workers)

        return {
            "worker_count": len(self._workers),
            "current_tasks": current_tasks,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "workers": [
                {
                    "id": w.id,
                    "status": w.status.value,
                    "current_tasks": w.info.current_tasks,
                    "completed": w.info.total_completed,
                }
                for w in self._workers
            ],
        }
