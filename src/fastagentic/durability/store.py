"""Durable store implementations for checkpoint persistence."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class DurableStore(ABC):
    """Abstract base class for durable checkpoint storage.

    Durable stores persist checkpoint data for agent runs, enabling
    resume, replay, and inspection capabilities.
    """

    @abstractmethod
    async def save_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        data: dict[str, Any],
    ) -> None:
        """Save a checkpoint.

        Args:
            run_id: The run identifier
            checkpoint_id: Unique checkpoint identifier
            data: Checkpoint data to persist
        """
        ...

    @abstractmethod
    async def get_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        """Get a specific checkpoint.

        Args:
            run_id: The run identifier
            checkpoint_id: The checkpoint identifier

        Returns:
            Checkpoint data or None if not found
        """
        ...

    @abstractmethod
    async def get_latest_checkpoint(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        """Get the latest checkpoint for a run.

        Args:
            run_id: The run identifier

        Returns:
            Latest checkpoint data or None if no checkpoints
        """
        ...

    @abstractmethod
    async def list_checkpoints(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """List all checkpoints for a run.

        Args:
            run_id: The run identifier

        Returns:
            List of checkpoint metadata
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def delete_run(self, run_id: str) -> int:
        """Delete all data for a run.

        Args:
            run_id: The run identifier

        Returns:
            Number of items deleted
        """
        ...

    @abstractmethod
    async def save_run_state(
        self,
        run_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save run state/status.

        Args:
            run_id: The run identifier
            status: Run status (pending, running, completed, failed, etc.)
            metadata: Optional metadata
        """
        ...

    @abstractmethod
    async def get_run_state(self, run_id: str) -> dict[str, Any] | None:
        """Get run state.

        Args:
            run_id: The run identifier

        Returns:
            Run state or None if not found
        """
        ...


class RedisDurableStore(DurableStore):
    """Redis-based durable store.

    Stores checkpoints and run state in Redis with configurable TTL.

    Example:
        store = RedisDurableStore(url="redis://localhost:6379")
        await store.save_checkpoint(run_id, checkpoint_id, {"state": ...})
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "fastagentic:",
        ttl_seconds: int = 86400 * 7,  # 7 days
    ) -> None:
        """Initialize Redis durable store.

        Args:
            url: Redis connection URL
            prefix: Key prefix for all keys
            ttl_seconds: TTL for stored data
        """
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create the Redis client."""
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self.url)
        return self._client

    def _checkpoint_key(self, run_id: str, checkpoint_id: str) -> str:
        """Generate key for a checkpoint."""
        return f"{self.prefix}checkpoint:{run_id}:{checkpoint_id}"

    def _checkpoints_key(self, run_id: str) -> str:
        """Generate key for checkpoint list."""
        return f"{self.prefix}checkpoints:{run_id}"

    def _run_key(self, run_id: str) -> str:
        """Generate key for run state."""
        return f"{self.prefix}run:{run_id}"

    async def save_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        data: dict[str, Any],
    ) -> None:
        """Save a checkpoint to Redis."""
        client = await self._get_client()

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Save checkpoint data
        key = self._checkpoint_key(run_id, checkpoint_id)
        await client.setex(key, self.ttl_seconds, json.dumps(checkpoint_data))

        # Add to checkpoint list
        list_key = self._checkpoints_key(run_id)
        await client.rpush(list_key, checkpoint_id)
        await client.expire(list_key, self.ttl_seconds)

    async def get_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        """Get a checkpoint from Redis."""
        client = await self._get_client()
        key = self._checkpoint_key(run_id, checkpoint_id)
        data = await client.get(key)
        if data:
            return json.loads(data)
        return None

    async def get_latest_checkpoint(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        """Get the latest checkpoint for a run."""
        client = await self._get_client()
        list_key = self._checkpoints_key(run_id)

        # Get the last checkpoint ID
        checkpoint_id = await client.lindex(list_key, -1)
        if not checkpoint_id:
            return None

        if isinstance(checkpoint_id, bytes):
            checkpoint_id = checkpoint_id.decode()

        return await self.get_checkpoint(run_id, checkpoint_id)

    async def list_checkpoints(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """List all checkpoints for a run."""
        client = await self._get_client()
        list_key = self._checkpoints_key(run_id)

        checkpoint_ids = await client.lrange(list_key, 0, -1)
        checkpoints = []

        for cp_id in checkpoint_ids:
            if isinstance(cp_id, bytes):
                cp_id = cp_id.decode()
            checkpoint = await self.get_checkpoint(run_id, cp_id)
            if checkpoint:
                checkpoints.append(checkpoint)

        return checkpoints

    async def delete_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
    ) -> None:
        """Delete a checkpoint."""
        client = await self._get_client()

        # Delete checkpoint data
        key = self._checkpoint_key(run_id, checkpoint_id)
        await client.delete(key)

        # Remove from list
        list_key = self._checkpoints_key(run_id)
        await client.lrem(list_key, 0, checkpoint_id)

    async def delete_run(self, run_id: str) -> int:
        """Delete all data for a run."""
        client = await self._get_client()
        count = 0

        # Delete all checkpoints
        checkpoints = await self.list_checkpoints(run_id)
        for cp in checkpoints:
            key = self._checkpoint_key(run_id, cp["checkpoint_id"])
            await client.delete(key)
            count += 1

        # Delete checkpoint list
        list_key = self._checkpoints_key(run_id)
        await client.delete(list_key)

        # Delete run state
        run_key = self._run_key(run_id)
        await client.delete(run_key)
        count += 1

        return count

    async def save_run_state(
        self,
        run_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save run state to Redis."""
        client = await self._get_client()

        run_data = {
            "run_id": run_id,
            "status": status,
            "metadata": metadata or {},
            "updated_at": datetime.utcnow().isoformat(),
        }

        key = self._run_key(run_id)
        await client.setex(key, self.ttl_seconds, json.dumps(run_data))

    async def get_run_state(self, run_id: str) -> dict[str, Any] | None:
        """Get run state from Redis."""
        client = await self._get_client()
        key = self._run_key(run_id)
        data = await client.get(key)
        if data:
            return json.loads(data)
        return None

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
