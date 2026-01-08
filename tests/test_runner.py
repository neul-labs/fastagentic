"""Tests for the runner module."""

from __future__ import annotations

import pytest

from fastagentic.checkpoint import CheckpointManager, InMemoryCheckpointStore
from fastagentic.runner import (
    ExecutionGraph,
    StepResult,
    StepStatus,
    StepTracker,
    run,
    run_opaque,
)


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_create_step_result(self):
        """Test creating a step result."""
        result = StepResult(
            name="search",
            status=StepStatus.COMPLETED,
            output={"data": "test"},
            tokens=1500,
            duration_ms=3200,
        )

        assert result.name == "search"
        assert result.status == StepStatus.COMPLETED
        assert result.output == {"data": "test"}
        assert result.tokens == 1500
        assert result.duration_ms == 3200

    def test_step_result_to_dict(self):
        """Test converting step result to dict."""
        result = StepResult(
            name="analyze",
            status=StepStatus.IN_PROGRESS,
            tokens=500,
        )

        data = result.to_dict()

        assert data["name"] == "analyze"
        assert data["status"] == "in_progress"
        assert data["tokens"] == 500

    def test_step_result_from_dict(self):
        """Test creating step result from dict."""
        data = {
            "name": "synthesize",
            "status": "completed",
            "output": "summary",
            "tokens": 2000,
            "duration_ms": 5000,
            "error": None,
        }

        result = StepResult.from_dict(data)

        assert result.name == "synthesize"
        assert result.status == StepStatus.COMPLETED
        assert result.output == "summary"
        assert result.tokens == 2000


class TestExecutionGraph:
    """Tests for ExecutionGraph dataclass."""

    def test_create_execution_graph(self):
        """Test creating an execution graph."""
        graph = ExecutionGraph(run_id="run-123")

        assert graph.run_id == "run-123"
        assert graph.steps == []
        assert graph.current_step is None

    def test_total_tokens(self):
        """Test total token calculation."""
        graph = ExecutionGraph(
            run_id="run-123",
            steps=[
                StepResult(name="s1", status=StepStatus.COMPLETED, tokens=1000),
                StepResult(name="s2", status=StepStatus.COMPLETED, tokens=2000),
                StepResult(name="s3", status=StepStatus.COMPLETED, tokens=500),
            ],
        )

        assert graph.total_tokens == 3500

    def test_total_duration(self):
        """Test total duration calculation."""
        graph = ExecutionGraph(
            run_id="run-123",
            steps=[
                StepResult(name="s1", status=StepStatus.COMPLETED, duration_ms=1000),
                StepResult(name="s2", status=StepStatus.COMPLETED, duration_ms=2000),
            ],
        )

        assert graph.total_duration_ms == 3000

    def test_is_complete(self):
        """Test completion check."""
        # Not complete - has pending
        graph1 = ExecutionGraph(
            run_id="run-1",
            steps=[
                StepResult(name="s1", status=StepStatus.COMPLETED),
                StepResult(name="s2", status=StepStatus.PENDING),
            ],
        )
        assert not graph1.is_complete

        # Complete - all done
        graph2 = ExecutionGraph(
            run_id="run-2",
            steps=[
                StepResult(name="s1", status=StepStatus.COMPLETED),
                StepResult(name="s2", status=StepStatus.SKIPPED),
            ],
        )
        assert graph2.is_complete

    def test_to_rich_table(self):
        """Test Rich table generation."""
        graph = ExecutionGraph(
            run_id="run-123",
            steps=[
                StepResult(name="search", status=StepStatus.COMPLETED, tokens=1500, duration_ms=3200),
                StepResult(name="analyze", status=StepStatus.IN_PROGRESS, tokens=500),
            ],
        )

        table = graph.to_rich_table()

        assert table.title == "Run: run-123"
        assert len(table.columns) == 4  # Step, Status, Tokens, Time

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        graph = ExecutionGraph(
            run_id="run-123",
            steps=[
                StepResult(name="s1", status=StepStatus.COMPLETED, tokens=1000),
            ],
            current_step="s2",
        )

        data = graph.to_dict()
        restored = ExecutionGraph.from_dict(data)

        assert restored.run_id == graph.run_id
        assert len(restored.steps) == 1
        assert restored.steps[0].name == "s1"
        assert restored.current_step == "s2"


class TestStepTracker:
    """Tests for StepTracker class."""

    @pytest.fixture
    def manager(self):
        """Create a checkpoint manager with in-memory store."""
        store = InMemoryCheckpointStore()
        return CheckpointManager(store)

    @pytest.mark.asyncio
    async def test_create_tracker(self, manager):
        """Test creating a step tracker."""
        tracker = StepTracker("run-123", manager)

        assert tracker.run_id == "run-123"
        assert tracker.graph.run_id == "run-123"

    @pytest.mark.asyncio
    async def test_step_context_manager(self, manager):
        """Test using step as context manager."""
        tracker = StepTracker("run-123", manager)

        async with tracker.step("search") as step:
            step.set_output({"results": [1, 2, 3]})
            step.add_tokens(1500)

        assert len(tracker.graph.steps) == 1
        assert tracker.graph.steps[0].name == "search"
        assert tracker.graph.steps[0].status == StepStatus.COMPLETED
        assert tracker.graph.steps[0].tokens == 1500
        assert tracker.get_output("search") == {"results": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_multiple_steps(self, manager):
        """Test multiple steps in sequence."""
        tracker = StepTracker("run-123", manager)

        async with tracker.step("step1") as step:
            step.set_output("result1")
            step.add_tokens(100)

        async with tracker.step("step2") as step:
            prev = step.get_previous_output("step1")
            step.set_output(f"processed {prev}")
            step.add_tokens(200)

        assert len(tracker.graph.steps) == 2
        assert tracker.get_output("step2") == "processed result1"
        assert tracker.graph.total_tokens == 300

    @pytest.mark.asyncio
    async def test_step_failure(self, manager):
        """Test step failure handling."""
        tracker = StepTracker("run-123", manager)

        with pytest.raises(ValueError, match="test error"):
            async with tracker.step("failing") as step:
                step.add_tokens(50)
                raise ValueError("test error")

        assert len(tracker.graph.steps) == 1
        assert tracker.graph.steps[0].status == StepStatus.FAILED
        assert tracker.graph.steps[0].error == "test error"

    @pytest.mark.asyncio
    async def test_step_callback(self, manager):
        """Test step status callback."""
        events = []

        def on_step(name: str, result: StepResult):
            events.append((name, result.status))

        tracker = StepTracker("run-123", manager, on_step=on_step)

        async with tracker.step("test"):
            pass

        # Should have in_progress and completed events
        assert len(events) == 2
        assert events[0] == ("test", StepStatus.IN_PROGRESS)
        assert events[1] == ("test", StepStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, manager):
        """Test restoring from checkpoint."""
        # First run - complete one step
        tracker1 = StepTracker("run-123", manager)

        async with tracker1.step("step1") as step:
            step.set_output("result1")
            step.add_tokens(100)

        # Second tracker - restore from checkpoint
        tracker2 = StepTracker("run-123", manager)
        restored = await tracker2.restore_from_checkpoint()

        assert restored is True
        assert tracker2.is_step_completed("step1")
        assert tracker2.get_output("step1") == "result1"

    @pytest.mark.asyncio
    async def test_skip_completed_step(self, manager):
        """Test skipping already completed steps on resume."""
        # First run
        tracker1 = StepTracker("run-123", manager)

        async with tracker1.step("step1") as step:
            step.set_output("result1")
            step.add_tokens(100)

        # Second run - restore and try same step
        tracker2 = StepTracker("run-123", manager)
        await tracker2.restore_from_checkpoint()

        skipped_events = []

        def on_step(name: str, result: StepResult):
            if result.status == StepStatus.SKIPPED:
                skipped_events.append(name)

        tracker2._on_step = on_step

        # This step should be skipped
        async with tracker2.step("step1") as step:
            # This code shouldn't run for skipped steps
            step.set_output("new_result")

        assert "step1" in skipped_events


class TestRunFunction:
    """Tests for the run() function."""

    @pytest.mark.asyncio
    async def test_run_basic_callable(self):
        """Test running a basic async callable."""

        async def my_agent(input: str, tracker: StepTracker):
            async with tracker.step("process"):
                pass
            return f"processed: {input}"

        result = await run(my_agent, "test input")

        assert result == "processed: test input"

    @pytest.mark.asyncio
    async def test_run_with_class(self):
        """Test running an agent class."""

        class MyAgent:
            async def run(self, input: str, tracker: StepTracker):
                async with tracker.step("execute"):
                    pass
                return input.upper()

        agent = MyAgent()
        result = await run(agent, "hello")

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_run_with_callback(self):
        """Test running with step callback."""
        events = []

        async def my_agent(input: str, tracker: StepTracker):
            async with tracker.step("step1"):
                pass
            async with tracker.step("step2"):
                pass
            return "done"

        def on_step(name: str, result: StepResult):
            events.append((name, result.status.value))

        await run(my_agent, "test", on_step=on_step)

        assert len(events) == 4  # 2 steps x 2 events each (in_progress, completed)

    @pytest.mark.asyncio
    async def test_run_custom_run_id(self):
        """Test running with custom run ID."""
        captured_run_id = None

        async def my_agent(input: str, tracker: StepTracker):
            nonlocal captured_run_id
            captured_run_id = tracker.run_id
            return "done"

        await run(my_agent, "test", run_id="my-custom-run")

        assert captured_run_id == "my-custom-run"

    @pytest.mark.asyncio
    async def test_run_with_custom_store(self):
        """Test running with custom checkpoint store."""
        store = InMemoryCheckpointStore()

        async def my_agent(input: str, tracker: StepTracker):
            async with tracker.step("test"):
                pass
            return "done"

        await run(my_agent, "test", run_id="run-123", store=store)

        # Verify checkpoint was saved
        checkpoint = await store.load_latest("run-123")
        assert checkpoint is not None


class TestRunOpaque:
    """Tests for the run_opaque() function - zero code changes wrapping."""

    @pytest.mark.asyncio
    async def test_run_opaque_async_function(self):
        """Test run_opaque with async function."""

        async def my_agent(query: str) -> dict:
            return {"result": f"processed {query}"}

        store = InMemoryCheckpointStore()
        result = await run_opaque(
            my_agent,
            query="test query",
            run_id="opaque-001",
            store=store,
        )

        assert result == {"result": "processed test query"}

    @pytest.mark.asyncio
    async def test_run_opaque_sync_function(self):
        """Test run_opaque with sync function."""

        def my_sync_agent(query: str) -> dict:
            return {"result": f"sync processed {query}"}

        store = InMemoryCheckpointStore()
        result = await run_opaque(
            my_sync_agent,
            query="sync test",
            run_id="opaque-sync-001",
            store=store,
        )

        assert result == {"result": "sync processed sync test"}

    @pytest.mark.asyncio
    async def test_run_opaque_caching(self):
        """Test that run_opaque returns cached result on second call."""
        call_count = 0

        async def my_agent(query: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": f"call {call_count}"}

        store = InMemoryCheckpointStore()

        # First call
        result1 = await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-cache-001",
            store=store,
        )
        assert result1 == {"result": "call 1"}
        assert call_count == 1

        # Second call - should return cached result
        result2 = await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-cache-001",
            store=store,
        )
        assert result2 == {"result": "call 1"}  # Same as first result
        assert call_count == 1  # Agent not called again

    @pytest.mark.asyncio
    async def test_run_opaque_different_run_ids(self):
        """Test that different run_ids don't share cache."""
        call_count = 0

        async def my_agent(query: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": f"call {call_count}"}

        store = InMemoryCheckpointStore()

        # First run_id
        result1 = await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-A",
            store=store,
        )
        assert call_count == 1

        # Different run_id - should execute again
        result2 = await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-B",
            store=store,
        )
        assert call_count == 2
        assert result1["result"] != result2["result"]

    @pytest.mark.asyncio
    async def test_run_opaque_progress_callback(self):
        """Test progress callback is called."""
        progress_messages = []

        def on_progress(msg: str) -> None:
            progress_messages.append(msg)

        async def my_agent(query: str) -> dict:
            return {"result": query}

        store = InMemoryCheckpointStore()

        await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-progress-001",
            store=store,
            on_progress=on_progress,
        )

        assert len(progress_messages) == 2
        assert "Running" in progress_messages[0]
        assert "Complete" in progress_messages[1]

    @pytest.mark.asyncio
    async def test_run_opaque_cached_progress_callback(self):
        """Test progress callback shows cached message on resume."""
        progress_messages = []

        def on_progress(msg: str) -> None:
            progress_messages.append(msg)

        async def my_agent(query: str) -> dict:
            return {"result": query}

        store = InMemoryCheckpointStore()

        # First call
        await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-cached-progress",
            store=store,
        )

        # Second call with progress callback
        await run_opaque(
            my_agent,
            query="test",
            run_id="opaque-cached-progress",
            store=store,
            on_progress=on_progress,
        )

        assert len(progress_messages) == 1
        assert "cached" in progress_messages[0].lower()
        assert "skipping" in progress_messages[0].lower()

    @pytest.mark.asyncio
    async def test_run_opaque_error_handling(self):
        """Test that errors are checkpointed but not cached as success."""

        async def failing_agent(query: str) -> dict:
            raise ValueError("Agent failed!")

        store = InMemoryCheckpointStore()

        with pytest.raises(ValueError, match="Agent failed!"):
            await run_opaque(
                failing_agent,
                query="test",
                run_id="opaque-error-001",
                store=store,
            )

        # Verify checkpoint was saved with failure
        checkpoint = await store.load_latest("opaque-error-001")
        assert checkpoint is not None
        assert checkpoint.state.get("completed") is False
        assert "error" in checkpoint.state

    @pytest.mark.asyncio
    async def test_run_opaque_retry_after_error(self):
        """Test that agent is re-run after a failed attempt."""
        attempts = 0

        async def flaky_agent(query: str) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("First attempt failed")
            return {"result": "success"}

        store = InMemoryCheckpointStore()

        # First attempt - fails
        with pytest.raises(ValueError):
            await run_opaque(
                flaky_agent,
                query="test",
                run_id="opaque-retry-001",
                store=store,
            )

        assert attempts == 1

        # Second attempt - should run agent again (not use failed cache)
        result = await run_opaque(
            flaky_agent,
            query="test",
            run_id="opaque-retry-001",
            store=store,
        )

        assert attempts == 2
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_run_opaque_auto_run_id(self):
        """Test that run_id is auto-generated if not provided."""

        async def my_agent(query: str) -> dict:
            return {"result": query}

        store = InMemoryCheckpointStore()

        result = await run_opaque(
            my_agent,
            query="test",
            store=store,
        )

        assert result == {"result": "test"}

    @pytest.mark.asyncio
    async def test_run_opaque_positional_args(self):
        """Test run_opaque with positional arguments."""

        async def my_agent(query: str, depth: int) -> dict:
            return {"query": query, "depth": depth}

        store = InMemoryCheckpointStore()

        result = await run_opaque(
            my_agent,
            "test query",
            3,
            run_id="opaque-positional-001",
            store=store,
        )

        assert result == {"query": "test query", "depth": 3}

    @pytest.mark.asyncio
    async def test_run_opaque_mixed_args(self):
        """Test run_opaque with both positional and keyword arguments."""

        async def my_agent(query: str, depth: int = 1, verbose: bool = False) -> dict:
            return {"query": query, "depth": depth, "verbose": verbose}

        store = InMemoryCheckpointStore()

        result = await run_opaque(
            my_agent,
            "test query",
            run_id="opaque-mixed-001",
            store=store,
            depth=5,
            verbose=True,
        )

        assert result == {"query": "test query", "depth": 5, "verbose": True}
