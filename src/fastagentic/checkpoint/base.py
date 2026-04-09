"""Base checkpoint classes for FastAgentic."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class CheckpointStatus(str, Enum):
    """Checkpoint status."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint.

    Attributes:
        run_id: The run this checkpoint belongs to
        checkpoint_id: Unique checkpoint ID
        sequence: Sequence number within the run
        created_at: When checkpoint was created
        status: Checkpoint status
        step_name: Name of the step being checkpointed
        parent_id: Parent checkpoint ID (for branching)
        tags: Optional tags for filtering
        ttl_seconds: Time-to-live in seconds (None = no expiry)
    """

    run_id: str
    checkpoint_id: str = field(default_factory=lambda: f"ckpt-{uuid.uuid4().hex[:12]}")
    sequence: int = 0
    created_at: float = field(default_factory=time.time)
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    step_name: str = ""
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    ttl_seconds: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if checkpoint has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() > self.created_at + self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "status": self.status.value,
            "step_name": self.step_name,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            checkpoint_id=data["checkpoint_id"],
            sequence=data["sequence"],
            created_at=data["created_at"],
            status=CheckpointStatus(data["status"]),
            step_name=data.get("step_name", ""),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            ttl_seconds=data.get("ttl_seconds"),
        )


@dataclass
class Checkpoint:
    """A checkpoint containing state and metadata.

    Attributes:
        metadata: Checkpoint metadata
        state: The checkpointed state
        messages: Conversation history up to this point
        tool_calls: Tool calls made up to this point
        context: Additional context data
    """

    metadata: CheckpointMetadata
    state: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_id(self) -> str:
        """Get checkpoint ID."""
        return self.metadata.checkpoint_id

    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self.metadata.run_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            state=data.get("state", {}),
            messages=data.get("messages", []),
            tool_calls=data.get("tool_calls", []),
            context=data.get("context", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> Checkpoint:
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(data))


class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends."""

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        ...

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        ...

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        """Load the latest checkpoint for a run."""
        ...

    async def list_checkpoints(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a run."""
        ...

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        ...

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run. Returns count deleted."""
        ...

    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints. Returns count deleted."""
        ...


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint manager.

    Attributes:
        auto_checkpoint: Automatically checkpoint after each step
        checkpoint_interval: Minimum seconds between auto-checkpoints
        max_checkpoints_per_run: Maximum checkpoints to keep per run
        default_ttl_seconds: Default TTL for checkpoints
        cleanup_interval: Seconds between cleanup runs
    """

    auto_checkpoint: bool = True
    checkpoint_interval: float = 0.0
    max_checkpoints_per_run: int = 100
    default_ttl_seconds: float | None = None
    cleanup_interval: float = 300.0  # 5 minutes


class CheckpointManager:
    """Manager for creating and restoring checkpoints.

    Example:
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        # Create checkpoint
        checkpoint = await manager.create(
            run_id="run-123",
            state={"step": "processing"},
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Restore from checkpoint
        checkpoint = await manager.restore("run-123")
        if checkpoint:
            state = checkpoint.state
            messages = checkpoint.messages

        # Or restore from specific checkpoint
        checkpoint = await manager.restore_from(checkpoint_id)
    """

    def __init__(
        self,
        store: CheckpointStore,
        config: CheckpointConfig | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            store: Checkpoint storage backend
            config: Manager configuration
        """
        self._store = store
        self.config = config or CheckpointConfig()
        self._sequence_counters: dict[str, int] = {}
        self._last_checkpoint_time: dict[str, float] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the checkpoint manager."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the checkpoint manager."""
        if not self._running:
            return
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def create(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        messages: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        step_name: str = "",
        parent_id: str | None = None,
        tags: list[str] | None = None,
        ttl_seconds: float | None = None,
        force: bool = False,
    ) -> Checkpoint | None:
        """Create a checkpoint.

        Args:
            run_id: Run identifier
            state: State to checkpoint
            messages: Conversation messages
            tool_calls: Tool calls made
            context: Additional context
            step_name: Name of current step
            parent_id: Parent checkpoint ID
            tags: Tags for filtering
            ttl_seconds: TTL for this checkpoint
            force: Force checkpoint even if interval not met

        Returns:
            Created checkpoint or None if skipped
        """
        # Check interval
        if not force and self.config.checkpoint_interval > 0:
            last_time = self._last_checkpoint_time.get(run_id, 0)
            if time.time() - last_time < self.config.checkpoint_interval:
                return None

        # Get next sequence number
        sequence = self._sequence_counters.get(run_id, 0)
        self._sequence_counters[run_id] = sequence + 1

        # Create checkpoint
        metadata = CheckpointMetadata(
            run_id=run_id,
            sequence=sequence,
            step_name=step_name,
            parent_id=parent_id,
            tags=tags or [],
            ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
        )

        checkpoint = Checkpoint(
            metadata=metadata,
            state=state,
            messages=messages or [],
            tool_calls=tool_calls or [],
            context=context or {},
        )

        # Save
        await self._store.save(checkpoint)
        self._last_checkpoint_time[run_id] = time.time()

        # Prune old checkpoints
        await self._prune_old_checkpoints(run_id)

        return checkpoint

    async def restore(self, run_id: str) -> Checkpoint | None:
        """Restore from the latest checkpoint for a run.

        Args:
            run_id: Run identifier

        Returns:
            Latest checkpoint or None
        """
        checkpoint = await self._store.load_latest(run_id)
        if checkpoint and checkpoint.metadata.is_expired:
            return None
        return checkpoint

    async def restore_from(self, checkpoint_id: str) -> Checkpoint | None:
        """Restore from a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None
        """
        checkpoint = await self._store.load(checkpoint_id)
        if checkpoint and checkpoint.metadata.is_expired:
            return None
        return checkpoint

    async def list_checkpoints(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a run.

        Args:
            run_id: Run identifier
            limit: Maximum to return

        Returns:
            List of checkpoint metadata
        """
        return await self._store.list_checkpoints(run_id, limit=limit)

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID
        """
        await self._store.delete(checkpoint_id)

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: Run identifier

        Returns:
            Number deleted
        """
        count = await self._store.delete_run(run_id)
        self._sequence_counters.pop(run_id, None)
        self._last_checkpoint_time.pop(run_id, None)
        return count

    async def mark_completed(self, run_id: str) -> None:
        """Mark all checkpoints for a run as completed.

        Args:
            run_id: Run identifier
        """
        checkpoint = await self._store.load_latest(run_id)
        if checkpoint:
            checkpoint.metadata.status = CheckpointStatus.COMPLETED
            await self._store.save(checkpoint)

    async def mark_failed(self, run_id: str, error: str = "") -> None:
        """Mark all checkpoints for a run as failed.

        Args:
            run_id: Run identifier
            error: Error message
        """
        checkpoint = await self._store.load_latest(run_id)
        if checkpoint:
            checkpoint.metadata.status = CheckpointStatus.FAILED
            checkpoint.context["error"] = error
            await self._store.save(checkpoint)

    async def _prune_old_checkpoints(self, run_id: str) -> None:
        """Prune old checkpoints for a run."""
        if self.config.max_checkpoints_per_run <= 0:
            return

        checkpoints = await self._store.list_checkpoints(
            run_id,
            limit=self.config.max_checkpoints_per_run + 10,
        )

        if len(checkpoints) > self.config.max_checkpoints_per_run:
            # Delete oldest checkpoints
            to_delete = checkpoints[self.config.max_checkpoints_per_run :]
            for meta in to_delete:
                await self._store.delete(meta.checkpoint_id)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired checkpoints."""
        while self._running:
            with contextlib.suppress(Exception):
                await self._store.cleanup_expired()
            await asyncio.sleep(self.config.cleanup_interval)
