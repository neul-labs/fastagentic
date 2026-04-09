"""Cluster coordinator for distributed FastAgentic."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastagentic.cluster.task import Task, TaskPriority, TaskQueue, TaskResult
from fastagentic.cluster.worker import (
    InMemoryWorkerRegistry,
    WorkerConfig,
    WorkerInfo,
    WorkerPool,
    WorkerRegistry,
)


@dataclass
class CoordinatorConfig:
    """Configuration for the cluster coordinator.

    Attributes:
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
        scale_up_threshold: Queue size to trigger scale up
        scale_down_threshold: Queue size to trigger scale down
        health_check_interval: Seconds between health checks
        worker_timeout: Seconds before considering worker dead
        auto_scale: Whether to automatically scale workers
    """

    min_workers: int = 1
    max_workers: int = 10
    scale_up_threshold: int = 10
    scale_down_threshold: int = 2
    health_check_interval: float = 10.0
    worker_timeout: float = 30.0
    auto_scale: bool = True


class Coordinator:
    """Cluster coordinator for managing distributed workloads.

    The coordinator manages:
    - Worker pool scaling
    - Task distribution
    - Health monitoring
    - Load balancing

    Example:
        coordinator = Coordinator(
            config=CoordinatorConfig(
                min_workers=2,
                max_workers=20,
                auto_scale=True,
            ),
        )

        # Register task handlers
        @coordinator.handler("process")
        async def process(data: dict) -> dict:
            return {"processed": True}

        # Start the coordinator
        await coordinator.start()

        # Submit tasks
        result = await coordinator.submit("process", {"input": "data"})

        # Or submit and wait
        task = await coordinator.submit_async("process", {"input": "data"})
        result = await coordinator.wait_for(task.id)

        # Stop the coordinator
        await coordinator.stop()
    """

    def __init__(
        self,
        config: CoordinatorConfig | None = None,
        worker_config: WorkerConfig | None = None,
        registry: WorkerRegistry | None = None,
    ) -> None:
        """Initialize coordinator.

        Args:
            config: Coordinator configuration
            worker_config: Default worker configuration
            registry: Worker registry for coordination
        """
        self.config = config or CoordinatorConfig()
        self.worker_config = worker_config or WorkerConfig()
        self._registry = registry or InMemoryWorkerRegistry()

        self._pool = WorkerPool(
            min_workers=self.config.min_workers,
            max_workers=self.config.max_workers,
            config=self.worker_config,
            registry=self._registry,
        )
        self._queue = TaskQueue()
        self._running = False
        self._health_check_task: asyncio.Task | None = None
        self._dispatcher_task: asyncio.Task | None = None
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {}

    @property
    def is_running(self) -> bool:
        """Check if coordinator is running."""
        return self._running

    def handler(
        self,
        task_type: str,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a task handler.

        Args:
            task_type: Type of task

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._handlers[task_type] = func
            self._pool.register_handler(task_type, func)
            return func

        return decorator

    def register_handler(
        self,
        task_type: str,
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Register a task handler.

        Args:
            task_type: Type of task
            handler: Handler function
        """
        self._handlers[task_type] = handler
        self._pool.register_handler(task_type, handler)

    async def start(self) -> None:
        """Start the coordinator."""
        if self._running:
            return

        self._running = True

        # Start worker pool
        await self._pool.start()

        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())

    async def stop(self, graceful: bool = True) -> None:
        """Stop the coordinator.

        Args:
            graceful: Wait for current tasks to complete
        """
        if not self._running:
            return

        self._running = False

        # Stop background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        # Stop worker pool
        await self._pool.stop(graceful=graceful)

    async def submit(
        self,
        task_type: str,
        data: dict[str, Any],
        *,
        _priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 300.0,
    ) -> Any:
        """Submit a task and wait for result.

        Args:
            task_type: Type of task
            data: Task input data
            priority: Task priority
            timeout: Task timeout

        Returns:
            Task result
        """
        if not self._running:
            raise RuntimeError("Coordinator is not running")

        # For immediate execution, use pool directly
        return await self._pool.submit(task_type, data, timeout=timeout)

    async def submit_async(
        self,
        task_type: str,
        data: dict[str, Any],
        *,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 300.0,
        retries: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Submit a task without waiting.

        Args:
            task_type: Type of task
            data: Task input data
            priority: Task priority
            timeout: Task timeout
            retries: Number of retries
            metadata: Additional metadata

        Returns:
            The created task
        """
        if not self._running:
            raise RuntimeError("Coordinator is not running")

        return await self._queue.enqueue(
            task_type=task_type,
            data=data,
            priority=priority,
            timeout=timeout,
            retries=retries,
            metadata=metadata,
        )

    async def wait_for(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> TaskResult:
        """Wait for a task to complete.

        Args:
            task_id: Task ID
            timeout: Maximum wait time

        Returns:
            TaskResult
        """
        return await self._queue.wait_for_task(task_id, timeout=timeout)

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None
        """
        return await self._queue.get_task(task_id)

    async def cancel_task(self, task_id: str) -> TaskResult:
        """Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            TaskResult
        """
        return await self._queue.cancel(task_id)

    async def scale(self, target: int) -> None:
        """Scale to target number of workers.

        Args:
            target: Target worker count
        """
        await self._pool.scale(target)

    async def get_workers(self) -> list[WorkerInfo]:
        """Get all workers."""
        return await self._registry.get_workers()

    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        pool_stats = self._pool.get_stats()
        queue_stats = self._queue.get_stats()

        return {
            "running": self._running,
            "workers": pool_stats,
            "queue": queue_stats,
            "config": {
                "min_workers": self.config.min_workers,
                "max_workers": self.config.max_workers,
                "auto_scale": self.config.auto_scale,
            },
        }

    async def _health_check_loop(self) -> None:
        """Periodic health check and scaling."""
        while self._running:
            try:
                await self._health_check()
            except Exception:
                pass  # Don't crash on health check errors

            await asyncio.sleep(self.config.health_check_interval)

    async def _health_check(self) -> None:
        """Perform health check and auto-scaling."""
        workers = await self._registry.get_workers()
        now = time.time()

        # Check for dead workers
        for worker in workers:
            if now - worker.last_heartbeat > self.config.worker_timeout:
                # Worker is dead - remove it
                await self._registry.unregister(worker.id)

        # Auto-scaling
        if not self.config.auto_scale:
            return

        queue_size = self._queue.size
        current_workers = self._pool.worker_count

        if queue_size > self.config.scale_up_threshold:
            # Scale up
            target = min(
                current_workers + 1,
                self.config.max_workers,
            )
            if target > current_workers:
                await self._pool.scale(target)

        elif queue_size < self.config.scale_down_threshold:
            # Scale down
            target = max(
                current_workers - 1,
                self.config.min_workers,
            )
            if target < current_workers:
                await self._pool.scale(target)

    async def _dispatcher_loop(self) -> None:
        """Dispatch queued tasks to workers."""
        while self._running:
            try:
                # Try to get a task from the queue
                task = await self._queue.dequeue(
                    worker_id="dispatcher",
                    timeout=1.0,
                )

                if task:
                    # Execute in worker pool
                    asyncio.create_task(self._execute_task(task))

            except Exception:
                pass  # Don't crash on dispatch errors

    async def _execute_task(self, task: Task) -> None:
        """Execute a task in the worker pool."""
        try:
            result = await self._pool.submit(
                task.type,
                task.data,
                timeout=task.timeout,
            )
            await self._queue.complete(task.id, result=result)

        except Exception as e:
            await self._queue.fail(task.id, error=str(e))
