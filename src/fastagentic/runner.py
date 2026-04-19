"""Runner module for production-safe agent execution.

Provides a simple one-liner API to wrap any agent with checkpointing,
observability, and resume capability.

Example:
    from fastagentic import run

    # Simple execution
    result = await run(my_agent, "Research quantum computing")

    # Resume after crash
    result = await run(my_agent, "Research quantum computing",
                       resume=True, run_id="run-abc123")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from rich.table import Table

from fastagentic.checkpoint import (
    CheckpointManager,
    CheckpointStore,
    FileCheckpointStore,
)

T = TypeVar("T")


class StepStatus(str, Enum):
    """Status of a step in the execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result from a single step execution.

    Attributes:
        name: Step name
        status: Current status
        output: Step output data (if completed)
        tokens: Tokens used in this step
        duration_ms: Execution time in milliseconds
        error: Error message (if failed)
    """

    name: str
    status: StepStatus
    output: Any = None
    tokens: int = 0
    duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for checkpointing."""
        return {
            "name": self.name,
            "status": self.status.value,
            "output": self.output,
            "tokens": self.tokens,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepResult:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            status=StepStatus(data["status"]),
            output=data.get("output"),
            tokens=data.get("tokens", 0),
            duration_ms=data.get("duration_ms", 0),
            error=data.get("error"),
        )


@dataclass
class ExecutionGraph:
    """Visualization data for an execution run.

    Provides methods to render the execution state as a Rich table
    for CLI display.

    Attributes:
        run_id: Unique run identifier
        steps: List of step results
        current_step: Name of currently executing step
        started_at: Run start timestamp
    """

    run_id: str
    steps: list[StepResult] = field(default_factory=list)
    current_step: str | None = None
    started_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all steps."""
        return sum(s.tokens for s in self.steps)

    @property
    def total_duration_ms(self) -> int:
        """Total duration in milliseconds."""
        return sum(s.duration_ms for s in self.steps)

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete or failed."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)
            for s in self.steps
        )

    def to_rich_table(self) -> Table:
        """Render as Rich table for CLI display."""
        table = Table(title=f"Run: {self.run_id}", show_header=True)
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Tokens", justify="right")
        table.add_column("Time", justify="right")

        status_icons = {
            StepStatus.PENDING: "[dim]○ pending[/dim]",
            StepStatus.IN_PROGRESS: "[yellow]● running[/yellow]",
            StepStatus.COMPLETED: "[green]✓ completed[/green]",
            StepStatus.FAILED: "[red]✗ failed[/red]",
            StepStatus.SKIPPED: "[blue]↷ skipped[/blue]",
        }

        for i, step in enumerate(self.steps, 1):
            tokens_str = f"{step.tokens:,}" if step.tokens else "-"
            time_str = f"{step.duration_ms / 1000:.1f}s" if step.duration_ms else "-"
            table.add_row(
                f"{i}. {step.name}",
                status_icons.get(step.status, step.status.value),
                tokens_str,
                time_str,
            )

        # Add total row
        table.add_section()
        table.add_row(
            "Total",
            "",
            f"{self.total_tokens:,}",
            f"{self.total_duration_ms / 1000:.1f}s",
            style="bold",
        )

        return table

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for checkpointing."""
        return {
            "run_id": self.run_id,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionGraph:
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            steps=[StepResult.from_dict(s) for s in data.get("steps", [])],
            current_step=data.get("current_step"),
            started_at=data.get("started_at", time.time()),
        )


class StepTracker:
    """Track steps within a run for checkpointing.

    Provides a context manager interface for defining steps
    and automatically handles checkpointing.

    Example:
        tracker = StepTracker(run_id, manager)

        async with tracker.step("search") as step:
            result = await search(query)
            step.set_output(result)
            step.add_tokens(1500)

        async with tracker.step("analyze") as step:
            analysis = await analyze(tracker.get_output("search"))
            step.set_output(analysis)
    """

    def __init__(
        self,
        run_id: str,
        manager: CheckpointManager,
        *,
        on_step: Callable[[str, StepResult], None] | None = None,
    ) -> None:
        """Initialize step tracker.

        Args:
            run_id: Unique run identifier
            manager: Checkpoint manager for persistence
            on_step: Optional callback when step status changes
        """
        self.run_id = run_id
        self._manager = manager
        self._on_step = on_step
        self._graph = ExecutionGraph(run_id=run_id)
        self._outputs: dict[str, Any] = {}
        self._current_step: _StepContext | None = None

    @property
    def graph(self) -> ExecutionGraph:
        """Get the execution graph."""
        return self._graph

    def get_output(self, step_name: str) -> Any:
        """Get output from a previous step."""
        return self._outputs.get(step_name)

    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has already been completed."""
        for step in self._graph.steps:
            if step.name == step_name and step.status == StepStatus.COMPLETED:
                return True
        return False

    async def restore_from_checkpoint(self) -> bool:
        """Restore state from the latest checkpoint.

        Returns:
            True if restored from checkpoint, False if starting fresh
        """
        checkpoint = await self._manager.restore(self.run_id)
        if checkpoint and checkpoint.state:
            # Restore graph
            if "graph" in checkpoint.state:
                self._graph = ExecutionGraph.from_dict(checkpoint.state["graph"])

            # Restore outputs
            if "outputs" in checkpoint.state:
                self._outputs = checkpoint.state["outputs"]

            return True
        return False

    async def _save_checkpoint(self) -> None:
        """Save current state to checkpoint."""
        state = {
            "graph": self._graph.to_dict(),
            "outputs": self._outputs,
        }
        await self._manager.create(
            run_id=self.run_id,
            state=state,
            step_name=self._graph.current_step or "",
            force=True,
        )

    @asynccontextmanager
    async def step(self, name: str) -> AsyncIterator[_StepContext]:
        """Context manager for a step.

        Automatically checkpoints on completion or failure.

        Args:
            name: Step name

        Yields:
            Step context for setting output and tokens
        """
        # Check if step already completed (resume case)
        if self.is_step_completed(name):
            # Create a skipped step result
            skipped = StepResult(
                name=name,
                status=StepStatus.SKIPPED,
                output=self._outputs.get(name),
            )
            if self._on_step:
                self._on_step(name, skipped)
            yield _StepContext(name, self)
            return

        # Create step result
        step_result = StepResult(name=name, status=StepStatus.IN_PROGRESS)
        self._graph.steps.append(step_result)
        self._graph.current_step = name

        # Notify callback
        if self._on_step:
            self._on_step(name, step_result)

        # Create context
        ctx = _StepContext(name, self)
        start_time = time.time()

        try:
            yield ctx

            # Mark completed
            step_result.status = StepStatus.COMPLETED
            step_result.output = ctx._output
            step_result.tokens = ctx._tokens
            step_result.duration_ms = int((time.time() - start_time) * 1000)

            # Store output
            if ctx._output is not None:
                self._outputs[name] = ctx._output

            # Save checkpoint
            await self._save_checkpoint()

            # Notify callback
            if self._on_step:
                self._on_step(name, step_result)

        except Exception as e:
            # Mark failed
            step_result.status = StepStatus.FAILED
            step_result.error = str(e)
            step_result.duration_ms = int((time.time() - start_time) * 1000)

            # Save checkpoint
            await self._save_checkpoint()

            # Notify callback
            if self._on_step:
                self._on_step(name, step_result)

            raise

        finally:
            self._graph.current_step = None


class _StepContext:
    """Context for a step execution."""

    def __init__(self, name: str, tracker: StepTracker) -> None:
        self.name = name
        self._tracker = tracker
        self._output: Any = None
        self._tokens: int = 0

    def set_output(self, output: Any) -> None:
        """Set the step output."""
        self._output = output

    def add_tokens(self, tokens: int) -> None:
        """Add to the token count for this step."""
        self._tokens += tokens

    def get_previous_output(self, step_name: str) -> Any:
        """Get output from a previous step."""
        return self._tracker.get_output(step_name)


async def run(
    agent: Callable[..., Any],
    input: Any,
    *,
    resume: bool = False,
    run_id: str | None = None,
    store: CheckpointStore | None = None,
    store_path: str = ".checkpoints",
    on_step: Callable[[str, StepResult], None] | None = None,
) -> Any:
    """Run an agent with automatic checkpointing.

    The magic one-liner for production-safe execution.

    Args:
        agent: The agent callable to run. Can be:
            - A coroutine function that takes (input, tracker)
            - A class with async run(input, tracker) method
        input: Input to pass to the agent
        resume: Whether to resume from last checkpoint
        run_id: Run identifier (auto-generated if not provided)
        store: Checkpoint store (FileCheckpointStore used if not provided)
        store_path: Path for FileCheckpointStore if store not provided
        on_step: Callback when step status changes

    Returns:
        Agent result

    Example:
        from fastagentic import run

        result = await run(my_agent, "Research quantum computing")

        # After crash, resume:
        result = await run(my_agent, "Research quantum computing",
                          resume=True, run_id="run-abc123")
    """
    # Generate run_id if not provided
    if run_id is None:
        run_id = f"run-{uuid.uuid4().hex[:8]}"

    # Create store if not provided
    if store is None:
        store = FileCheckpointStore(store_path)

    # Create manager
    manager = CheckpointManager(store)

    # Create tracker
    tracker = StepTracker(run_id, manager, on_step=on_step)

    # Restore from checkpoint if resuming
    if resume:
        restored = await tracker.restore_from_checkpoint()
        if not restored:
            # No checkpoint found, but resume was requested
            pass  # Continue with fresh run

    # Run the agent
    # Handle both callable and class with run method
    run_method = getattr(agent, "run", None)
    if run_method is not None and callable(run_method):
        result = await run_method(input, tracker)
    elif asyncio.iscoroutinefunction(agent):
        result = await agent(input, tracker)
    else:
        # Try calling it anyway
        result = await agent(input, tracker)

    # Mark run as completed
    await manager.mark_completed(run_id)

    return result


async def run_opaque(
    agent: Callable[..., Any],
    *args: Any,
    run_id: str | None = None,
    store: CheckpointStore | None = None,
    store_path: str = ".checkpoints",
    on_progress: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Run any agent with automatic checkpointing - zero code changes required.

    This wraps an existing agent function without requiring any modifications.
    The entire agent execution is treated as a single unit:
    - If already completed, returns cached result (skips execution)
    - If not completed, runs agent and caches result

    Args:
        agent: Any callable (sync or async function)
        *args: Positional arguments to pass to the agent
        run_id: Run identifier (auto-generated if not provided)
        store: Checkpoint store (FileCheckpointStore used if not provided)
        store_path: Path for FileCheckpointStore if store not provided
        on_progress: Optional callback for progress updates
        **kwargs: Keyword arguments to pass to the agent

    Returns:
        Agent result (cached if already completed)

    Example:
        from local_deep_research import quick_summary
        from fastagentic import run_opaque

        # Wrap ANY existing agent - zero changes required
        result = await run_opaque(
            quick_summary,
            query="quantum computing applications",
            run_id="research-001",
        )

        # Resume - returns cached result if complete
        result = await run_opaque(
            quick_summary,
            query="quantum computing applications",
            run_id="research-001",  # Same run_id
        )
    """
    # Generate run_id if not provided
    if run_id is None:
        run_id = f"run-{uuid.uuid4().hex[:8]}"

    # Create store if not provided
    if store is None:
        store = FileCheckpointStore(store_path)

    # Create manager
    manager = CheckpointManager(store)

    # Check for existing completed checkpoint
    checkpoint = await manager.restore(run_id)
    if checkpoint and checkpoint.state.get("completed"):
        # Return cached result
        if on_progress:
            on_progress(f"Found cached result for {run_id} - skipping execution")
        return checkpoint.state.get("result")

    # Notify progress
    if on_progress:
        agent_name = getattr(agent, "__name__", str(agent))
        on_progress(f"Running {agent_name}...")

    # Run the agent
    start_time = time.time()
    try:
        # Handle both sync and async callables
        if asyncio.iscoroutinefunction(agent):
            result = await agent(*args, **kwargs)
        else:
            # Run sync function in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: agent(*args, **kwargs)
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Checkpoint the result
        await manager.create(
            run_id=run_id,
            state={
                "completed": True,
                "result": result,
                "duration_ms": duration_ms,
                "agent": getattr(agent, "__name__", str(agent)),
                "args": str(args)[:500],  # Truncate for storage
                "kwargs_keys": list(kwargs.keys()),
            },
            step_name="complete",
            force=True,
        )

        # Mark completed
        await manager.mark_completed(run_id)

        if on_progress:
            on_progress(f"Complete! Cached for resume. Duration: {duration_ms / 1000:.1f}s")

        return result

    except Exception as e:
        # Checkpoint the failure
        duration_ms = int((time.time() - start_time) * 1000)
        await manager.create(
            run_id=run_id,
            state={
                "completed": False,
                "error": str(e),
                "duration_ms": duration_ms,
            },
            step_name="failed",
            force=True,
        )
        await manager.mark_failed(run_id, str(e))
        raise
