"""Tests for cluster orchestration module."""

import asyncio

import pytest

from fastagentic.cluster import (
    Coordinator,
    CoordinatorConfig,
    Task,
    TaskPriority,
    TaskQueue,
    TaskStatus,
    Worker,
    WorkerConfig,
    WorkerPool,
    WorkerStatus,
)

# ============================================================================
# Worker Tests
# ============================================================================


class TestWorker:
    """Tests for Worker class."""

    @pytest.mark.asyncio
    async def test_create_worker(self):
        """Test creating a worker."""
        worker = Worker(id="test-worker")
        assert worker.id == "test-worker"
        assert worker.status == WorkerStatus.IDLE

    @pytest.mark.asyncio
    async def test_worker_auto_id(self):
        """Test worker auto-generates ID."""
        worker = Worker()
        assert worker.id.startswith("worker-")

    @pytest.mark.asyncio
    async def test_worker_start_stop(self):
        """Test starting and stopping worker."""
        worker = Worker()
        await worker.start()
        assert worker.status == WorkerStatus.IDLE

        await worker.stop()
        assert worker.status == WorkerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_register_handler_decorator(self):
        """Test registering handler via decorator."""
        worker = Worker()

        @worker.handler("process")
        async def process_handler(data):
            return {"result": data["value"] * 2}

        assert "process" in worker._handlers

    @pytest.mark.asyncio
    async def test_register_handler_method(self):
        """Test registering handler via method."""
        worker = Worker()

        async def process_handler(data):
            return {"result": data["value"] * 2}

        worker.register_handler("process", process_handler)
        assert "process" in worker._handlers

    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test executing a task."""
        worker = Worker()

        @worker.handler("double")
        async def double_handler(data):
            return data["value"] * 2

        await worker.start()
        result = await worker.execute_task("task-1", "double", {"value": 5})
        assert result == 10
        assert worker.info.total_completed == 1

        await worker.stop()

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self):
        """Test executing unknown task type raises error."""
        worker = Worker()
        await worker.start()

        with pytest.raises(ValueError, match="No handler for task type"):
            await worker.execute_task("task-1", "unknown", {})

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_task_failure(self):
        """Test task failure tracking."""
        worker = Worker()

        @worker.handler("fail")
        async def fail_handler(data):
            raise ValueError("Task failed")

        await worker.start()

        with pytest.raises(ValueError):
            await worker.execute_task("task-1", "fail", {})

        assert worker.info.total_failed == 1
        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_config(self):
        """Test worker with custom config."""
        config = WorkerConfig(
            max_concurrent_tasks=4,
            heartbeat_interval=2.0,
            task_timeout=60.0,
            tags=["gpu", "fast"],
        )
        worker = Worker(config=config)

        assert worker.config.max_concurrent_tasks == 4
        assert worker.config.heartbeat_interval == 2.0
        assert "gpu" in worker.config.tags

    @pytest.mark.asyncio
    async def test_worker_availability(self):
        """Test worker availability check."""
        config = WorkerConfig(max_concurrent_tasks=1)
        worker = Worker(config=config)
        await worker.start()

        assert worker.info.is_available

        # Simulate busy worker
        worker._info.current_tasks = 1
        assert not worker.info.is_available

        await worker.stop()


class TestWorkerConfig:
    """Tests for WorkerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WorkerConfig()
        assert config.max_concurrent_tasks == 1
        assert config.heartbeat_interval == 5.0
        assert config.task_timeout == 300.0
        assert config.graceful_shutdown_timeout == 30.0
        assert config.tags == []

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkerConfig(
            max_concurrent_tasks=8,
            tags=["ml", "cpu"],
        )
        assert config.max_concurrent_tasks == 8
        assert config.tags == ["ml", "cpu"]


# ============================================================================
# Task Tests
# ============================================================================


class TestTask:
    """Tests for Task class."""

    def test_create_task(self):
        """Test creating a task."""
        task = Task(type="process", data={"key": "value"})
        assert task.type == "process"
        assert task.data == {"key": "value"}
        assert task.id.startswith("task-")
        assert task.status == TaskStatus.PENDING

    def test_task_priority(self):
        """Test task priority levels."""
        assert TaskPriority.LOW < TaskPriority.NORMAL
        assert TaskPriority.NORMAL < TaskPriority.HIGH
        assert TaskPriority.HIGH < TaskPriority.CRITICAL

    def test_task_is_complete(self):
        """Test task completion check."""
        task = Task(type="test", data={})
        assert not task.is_complete

        task.status = TaskStatus.COMPLETED
        assert task.is_complete

        task.status = TaskStatus.FAILED
        assert task.is_complete

        task.status = TaskStatus.CANCELLED
        assert task.is_complete

    def test_task_can_retry(self):
        """Test retry check."""
        task = Task(type="test", data={}, retries=3)
        assert task.can_retry  # 0 < 3

        task.retry_count = 3
        assert not task.can_retry  # 3 >= 3


class TestTaskQueue:
    """Tests for TaskQueue class."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        """Test basic enqueue and dequeue."""
        queue = TaskQueue()

        task = await queue.enqueue(
            task_type="process",
            data={"value": 1},
        )
        assert task.status == TaskStatus.QUEUED
        assert queue.size == 1

        dequeued = await queue.dequeue("worker-1")
        assert dequeued.id == task.id
        assert dequeued.status == TaskStatus.RUNNING
        assert queue.size == 0

    @pytest.mark.asyncio
    async def test_priority_queue(self):
        """Test priority ordering."""
        queue = TaskQueue()

        low = await queue.enqueue("task", {}, priority=TaskPriority.LOW)
        normal = await queue.enqueue("task", {}, priority=TaskPriority.NORMAL)
        high = await queue.enqueue("task", {}, priority=TaskPriority.HIGH)

        # Should dequeue in priority order
        first = await queue.dequeue("worker-1")
        assert first.id == high.id

        second = await queue.dequeue("worker-1")
        assert second.id == normal.id

        third = await queue.dequeue("worker-1")
        assert third.id == low.id

    @pytest.mark.asyncio
    async def test_complete_task(self):
        """Test completing a task."""
        queue = TaskQueue()
        task = await queue.enqueue("process", {"x": 1})
        await queue.dequeue("worker-1")

        result = await queue.complete(task.id, result={"success": True})
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"success": True}

    @pytest.mark.asyncio
    async def test_fail_task(self):
        """Test failing a task."""
        queue = TaskQueue()
        task = await queue.enqueue("process", {})
        await queue.dequeue("worker-1")

        result = await queue.fail(task.id, error="Something went wrong")
        assert result.status == TaskStatus.FAILED
        assert result.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_fail_with_retry(self):
        """Test failing a task with retry."""
        queue = TaskQueue()
        task = await queue.enqueue("process", {}, retries=2)
        await queue.dequeue("worker-1")

        result = await queue.fail(task.id, error="Temporary failure")
        assert result.status == TaskStatus.QUEUED  # Re-queued for retry
        assert queue.size == 1

        # Get the task again
        updated = await queue.get_task(task.id)
        assert updated.retry_count == 1

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test cancelling a task."""
        queue = TaskQueue()
        task = await queue.enqueue("process", {})

        result = await queue.cancel(task.id)
        assert result.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_wait_for_task(self):
        """Test waiting for task completion."""
        queue = TaskQueue()
        task = await queue.enqueue("process", {})
        await queue.dequeue("worker-1")

        # Complete in background
        async def complete_later():
            await asyncio.sleep(0.1)
            await queue.complete(task.id, result="done")

        asyncio.create_task(complete_later())

        result = await queue.wait_for_task(task.id, timeout=1.0)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "done"

    @pytest.mark.asyncio
    async def test_queue_stats(self):
        """Test queue statistics."""
        queue = TaskQueue()
        await queue.enqueue("task", {})
        await queue.enqueue("task", {})

        stats = queue.get_stats()
        assert stats["queue_size"] == 2
        assert stats["pending_count"] == 2


# ============================================================================
# WorkerPool Tests
# ============================================================================


class TestWorkerPool:
    """Tests for WorkerPool class."""

    @pytest.mark.asyncio
    async def test_create_pool(self):
        """Test creating a worker pool."""
        pool = WorkerPool(min_workers=2, max_workers=5)
        assert pool.min_workers == 2
        assert pool.max_workers == 5
        assert pool.worker_count == 0

    @pytest.mark.asyncio
    async def test_start_pool(self):
        """Test starting a pool creates minimum workers."""
        pool = WorkerPool(min_workers=2, max_workers=5)
        await pool.start()

        assert pool.worker_count == 2

        await pool.stop()

    @pytest.mark.asyncio
    async def test_register_handler(self):
        """Test registering handler on pool."""
        pool = WorkerPool(min_workers=1)

        async def handler(data):
            return data

        pool.register_handler("test", handler)
        assert "test" in pool._handlers

        await pool.start()
        assert "test" in pool._workers[0]._handlers
        await pool.stop()

    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test submitting task to pool."""
        pool = WorkerPool(min_workers=1)

        async def double(data):
            return data["value"] * 2

        pool.register_handler("double", double)
        await pool.start()

        result = await pool.submit("double", {"value": 5})
        assert result == 10

        await pool.stop()

    @pytest.mark.asyncio
    async def test_scale_up(self):
        """Test scaling up workers."""
        pool = WorkerPool(min_workers=1, max_workers=5)
        await pool.start()
        assert pool.worker_count == 1

        await pool.scale(3)
        assert pool.worker_count == 3

        await pool.stop()

    @pytest.mark.asyncio
    async def test_scale_down(self):
        """Test scaling down workers."""
        pool = WorkerPool(min_workers=1, max_workers=5)
        await pool.start()
        await pool.scale(4)
        assert pool.worker_count == 4

        await pool.scale(2)
        assert pool.worker_count == 2

        await pool.stop()

    @pytest.mark.asyncio
    async def test_scale_respects_limits(self):
        """Test scaling respects min/max limits."""
        pool = WorkerPool(min_workers=2, max_workers=5)
        await pool.start()

        await pool.scale(1)  # Below min
        assert pool.worker_count == 2

        await pool.scale(10)  # Above max
        assert pool.worker_count == 5

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_stats(self):
        """Test pool statistics."""
        pool = WorkerPool(min_workers=2)

        async def handler(data):
            return data

        pool.register_handler("test", handler)
        await pool.start()

        await pool.submit("test", {})

        stats = pool.get_stats()
        assert stats["worker_count"] == 2
        assert stats["total_completed"] == 1

        await pool.stop()


# ============================================================================
# Coordinator Tests
# ============================================================================


class TestCoordinator:
    """Tests for Coordinator class."""

    @pytest.mark.asyncio
    async def test_create_coordinator(self):
        """Test creating a coordinator."""
        coordinator = Coordinator()
        assert not coordinator.is_running

    @pytest.mark.asyncio
    async def test_start_stop_coordinator(self):
        """Test starting and stopping coordinator."""
        coordinator = Coordinator(config=CoordinatorConfig(min_workers=1, auto_scale=False))
        await coordinator.start()
        assert coordinator.is_running

        await coordinator.stop()
        assert not coordinator.is_running

    @pytest.mark.asyncio
    async def test_register_handler_decorator(self):
        """Test registering handler via decorator."""
        coordinator = Coordinator()

        @coordinator.handler("process")
        async def process(data):
            return {"processed": True}

        assert "process" in coordinator._handlers

    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test submitting a task."""
        coordinator = Coordinator(config=CoordinatorConfig(min_workers=1, auto_scale=False))

        @coordinator.handler("double")
        async def double(data):
            return data["value"] * 2

        await coordinator.start()
        result = await coordinator.submit("double", {"value": 7})
        assert result == 14

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_submit_async_and_wait(self):
        """Test async submit and wait."""
        coordinator = Coordinator(config=CoordinatorConfig(min_workers=1, auto_scale=False))

        @coordinator.handler("process")
        async def process(data):
            return {"result": data["x"] + data["y"]}

        await coordinator.start()

        task = await coordinator.submit_async(
            "process",
            {"x": 3, "y": 4},
        )
        assert task.status == TaskStatus.QUEUED

        # Wait for result
        result = await coordinator.wait_for(task.id, timeout=5.0)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"result": 7}

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinator_stats(self):
        """Test coordinator statistics."""
        coordinator = Coordinator(config=CoordinatorConfig(min_workers=2, auto_scale=False))
        await coordinator.start()

        stats = coordinator.get_stats()
        assert stats["running"] is True
        assert stats["workers"]["worker_count"] == 2
        assert stats["config"]["min_workers"] == 2

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinator_scale(self):
        """Test manual scaling."""
        coordinator = Coordinator(
            config=CoordinatorConfig(min_workers=1, max_workers=5, auto_scale=False)
        )
        await coordinator.start()

        await coordinator.scale(3)
        stats = coordinator.get_stats()
        assert stats["workers"]["worker_count"] == 3

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinator_get_workers(self):
        """Test getting worker list."""
        coordinator = Coordinator(config=CoordinatorConfig(min_workers=2, auto_scale=False))
        await coordinator.start()

        workers = await coordinator.get_workers()
        assert len(workers) == 2

        await coordinator.stop()


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CoordinatorConfig()
        assert config.min_workers == 1
        assert config.max_workers == 10
        assert config.auto_scale is True
        assert config.health_check_interval == 10.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CoordinatorConfig(
            min_workers=5,
            max_workers=50,
            auto_scale=False,
            scale_up_threshold=20,
        )
        assert config.min_workers == 5
        assert config.max_workers == 50
        assert config.scale_up_threshold == 20
