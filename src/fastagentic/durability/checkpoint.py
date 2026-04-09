"""Checkpoint management for durable runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastagentic.durability.store import DurableStore


@dataclass
class Checkpoint:
    """Represents a checkpoint in a durable run.

    Checkpoints capture the state of an agent run at a specific point,
    enabling resume and replay functionality.
    """

    checkpoint_id: str
    run_id: str
    node: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        run_id: str,
        state: dict[str, Any],
        node: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            run_id: The run identifier
            state: State to checkpoint
            node: Optional node name (for LangGraph)
            metadata: Optional metadata

        Returns:
            A new Checkpoint instance
        """
        return cls(
            checkpoint_id=str(uuid.uuid4()),
            run_id=run_id,
            node=node,
            state=state,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "node": self.node,
            "state": self.state,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        return cls(
            checkpoint_id=data["checkpoint_id"],
            run_id=data["run_id"],
            node=data.get("node"),
            state=data.get("state", {}),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


class CheckpointManager:
    """Manages checkpoints for durable runs.

    Provides high-level operations for creating, loading, and managing
    checkpoints using a configured durable store.

    Example:
        store = RedisDurableStore(url="redis://localhost:6379")
        manager = CheckpointManager(store)

        # Create checkpoint
        checkpoint = await manager.create_checkpoint(run_id, state)

        # Resume from checkpoint
        state = await manager.get_resume_state(run_id)
    """

    def __init__(self, store: DurableStore) -> None:
        """Initialize the checkpoint manager.

        Args:
            store: The durable store to use for persistence
        """
        self.store = store

    async def create_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        node: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create and save a new checkpoint.

        Args:
            run_id: The run identifier
            state: State to checkpoint
            node: Optional node name
            metadata: Optional metadata

        Returns:
            The created Checkpoint
        """
        checkpoint = Checkpoint.create(
            run_id=run_id,
            state=state,
            node=node,
            metadata=metadata,
        )

        await self.store.save_checkpoint(
            run_id=run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            data=checkpoint.to_dict(),
        )

        return checkpoint

    async def get_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
    ) -> Checkpoint | None:
        """Get a specific checkpoint.

        Args:
            run_id: The run identifier
            checkpoint_id: The checkpoint identifier

        Returns:
            Checkpoint or None if not found
        """
        data = await self.store.get_checkpoint(run_id, checkpoint_id)
        if data:
            return Checkpoint.from_dict(data.get("data", data))
        return None

    async def get_latest_checkpoint(
        self,
        run_id: str,
    ) -> Checkpoint | None:
        """Get the latest checkpoint for a run.

        Args:
            run_id: The run identifier

        Returns:
            Latest Checkpoint or None
        """
        data = await self.store.get_latest_checkpoint(run_id)
        if data:
            return Checkpoint.from_dict(data.get("data", data))
        return None

    async def get_resume_state(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        """Get the state to resume from.

        Convenience method that returns the state from the latest checkpoint.

        Args:
            run_id: The run identifier

        Returns:
            State dict or None if no checkpoint
        """
        checkpoint = await self.get_latest_checkpoint(run_id)
        if checkpoint:
            return checkpoint.state
        return None

    async def list_checkpoints(
        self,
        run_id: str,
    ) -> list[Checkpoint]:
        """List all checkpoints for a run.

        Args:
            run_id: The run identifier

        Returns:
            List of Checkpoint objects
        """
        checkpoint_data = await self.store.list_checkpoints(run_id)
        return [Checkpoint.from_dict(cp.get("data", cp)) for cp in checkpoint_data]

    async def delete_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
    ) -> None:
        """Delete a specific checkpoint.

        Args:
            run_id: The run identifier
            checkpoint_id: The checkpoint identifier
        """
        await self.store.delete_checkpoint(run_id, checkpoint_id)

    async def clear_checkpoints(self, run_id: str) -> int:
        """Clear all checkpoints for a run.

        Args:
            run_id: The run identifier

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = await self.list_checkpoints(run_id)
        for cp in checkpoints:
            await self.store.delete_checkpoint(run_id, cp.checkpoint_id)
        return len(checkpoints)

    async def set_run_status(
        self,
        run_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set the status of a run.

        Args:
            run_id: The run identifier
            status: Status string (pending, running, completed, failed, etc.)
            metadata: Optional metadata
        """
        await self.store.save_run_state(run_id, status, metadata)

    async def get_run_status(self, run_id: str) -> str | None:
        """Get the status of a run.

        Args:
            run_id: The run identifier

        Returns:
            Status string or None if not found
        """
        state = await self.store.get_run_state(run_id)
        if state:
            return state.get("status")
        return None
