"""Checkpoint storage backends for FastAgentic."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import aiofiles

from fastagentic.checkpoint.base import Checkpoint, CheckpointMetadata


class InMemoryCheckpointStore:
    """In-memory checkpoint store for development/testing."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}
        self._by_run: dict[str, list[str]] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        ckpt_id = checkpoint.checkpoint_id
        run_id = checkpoint.run_id

        self._checkpoints[ckpt_id] = checkpoint

        if run_id not in self._by_run:
            self._by_run[run_id] = []
        if ckpt_id not in self._by_run[run_id]:
            self._by_run[run_id].append(ckpt_id)

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        """Load the latest checkpoint for a run."""
        ckpt_ids = self._by_run.get(run_id, [])
        if not ckpt_ids:
            return None

        # Find latest by sequence number
        latest: Checkpoint | None = None
        max_seq = -1

        for ckpt_id in ckpt_ids:
            ckpt = self._checkpoints.get(ckpt_id)
            if ckpt and ckpt.metadata.sequence > max_seq:
                max_seq = ckpt.metadata.sequence
                latest = ckpt

        return latest

    async def list_checkpoints(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a run."""
        ckpt_ids = self._by_run.get(run_id, [])
        results = []

        for ckpt_id in ckpt_ids:
            ckpt = self._checkpoints.get(ckpt_id)
            if ckpt:
                results.append(ckpt.metadata)

        # Sort by sequence descending (newest first)
        results.sort(key=lambda m: m.sequence, reverse=True)
        return results[:limit]

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        ckpt = self._checkpoints.pop(checkpoint_id, None)
        if ckpt:
            run_ids = self._by_run.get(ckpt.run_id, [])
            if checkpoint_id in run_ids:
                run_ids.remove(checkpoint_id)

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run."""
        ckpt_ids = self._by_run.pop(run_id, [])
        for ckpt_id in ckpt_ids:
            self._checkpoints.pop(ckpt_id, None)
        return len(ckpt_ids)

    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints."""
        expired = []
        for ckpt_id, ckpt in self._checkpoints.items():
            if ckpt.metadata.is_expired:
                expired.append(ckpt_id)

        for ckpt_id in expired:
            await self.delete(ckpt_id)

        return len(expired)


class FileCheckpointStore:
    """File-based checkpoint store.

    Stores checkpoints as JSON files in a directory structure:
    base_path/
        run_id/
            checkpoint_id.json
            ...
    """

    def __init__(self, base_path: str | Path) -> None:
        """Initialize file checkpoint store.

        Args:
            base_path: Base directory for storing checkpoints
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        """Get directory for a run."""
        return self.base_path / run_id

    def _checkpoint_path(self, run_id: str, checkpoint_id: str) -> Path:
        """Get path for a checkpoint file."""
        return self._run_dir(run_id) / f"{checkpoint_id}.json"

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        run_dir = self._run_dir(checkpoint.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        path = self._checkpoint_path(
            checkpoint.run_id,
            checkpoint.checkpoint_id,
        )
        path.write_text(checkpoint.to_json())

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        # Search all run directories
        for run_dir in self.base_path.iterdir():
            if run_dir.is_dir():
                path = run_dir / f"{checkpoint_id}.json"
                if path.exists():
                    async with aiofiles.open(path) as f:
                        return Checkpoint.from_json(await f.read())
        return None

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        """Load the latest checkpoint for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return None

        latest: Checkpoint | None = None
        max_seq = -1

        for path in run_dir.glob("*.json"):
            try:
                async with aiofiles.open(path) as f:
                    ckpt = Checkpoint.from_json(await f.read())
                if ckpt.metadata.sequence > max_seq:
                    max_seq = ckpt.metadata.sequence
                    latest = ckpt
            except (json.JSONDecodeError, ValueError, OSError):
                continue

        return latest

    async def list_checkpoints(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return []

        results = []
        for path in run_dir.glob("*.json"):
            try:
                async with aiofiles.open(path) as f:
                    ckpt = Checkpoint.from_json(await f.read())
                results.append(ckpt.metadata)
            except (json.JSONDecodeError, ValueError, OSError):
                continue

        results.sort(key=lambda m: m.sequence, reverse=True)
        return results[:limit]

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        for run_dir in self.base_path.iterdir():
            if run_dir.is_dir():
                path = run_dir / f"{checkpoint_id}.json"
                if path.exists():
                    path.unlink()
                    return

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return 0

        count = 0
        for path in run_dir.glob("*.json"):
            path.unlink()
            count += 1

        # Remove empty directory
        with contextlib.suppress(OSError):
            run_dir.rmdir()

        return count

    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints."""
        expired = []

        for run_dir in self.base_path.iterdir():
            if run_dir.is_dir():
                for path in run_dir.glob("*.json"):
                    try:
                        async with aiofiles.open(path) as f:
                            ckpt = Checkpoint.from_json(await f.read())
                        if ckpt.metadata.is_expired:
                            expired.append(path)
                    except (json.JSONDecodeError, ValueError, OSError):
                        continue

        for path in expired:
            path.unlink()

        return len(expired)


class RedisCheckpointStore:
    """Redis-based checkpoint store for distributed deployments.

    Requires redis-py async client.

    Example:
        import redis.asyncio as redis
        client = redis.Redis(host="localhost", port=6379)
        store = RedisCheckpointStore(client)
    """

    def __init__(
        self,
        client: Any,  # redis.asyncio.Redis
        prefix: str = "fastagentic:checkpoint:",
        ttl_seconds: int | None = None,
    ) -> None:
        """Initialize Redis checkpoint store.

        Args:
            client: Redis async client
            prefix: Key prefix for all checkpoint data
            ttl_seconds: Default TTL for Redis keys
        """
        self._client = client
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _ckpt_key(self, checkpoint_id: str) -> str:
        """Get Redis key for a checkpoint."""
        return f"{self._prefix}ckpt:{checkpoint_id}"

    def _run_key(self, run_id: str) -> str:
        """Get Redis key for run checkpoint list."""
        return f"{self._prefix}run:{run_id}"

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        ckpt_key = self._ckpt_key(checkpoint.checkpoint_id)
        run_key = self._run_key(checkpoint.run_id)

        # Store checkpoint data
        data = checkpoint.to_json()

        # Determine TTL
        ttl = None
        if checkpoint.metadata.ttl_seconds:
            ttl = int(checkpoint.metadata.ttl_seconds)
        elif self._ttl:
            ttl = self._ttl

        if ttl:
            await self._client.setex(ckpt_key, ttl, data)
        else:
            await self._client.set(ckpt_key, data)

        # Add to run's checkpoint list (sorted set by sequence)
        await self._client.zadd(
            run_key,
            {checkpoint.checkpoint_id: checkpoint.metadata.sequence},
        )

        if ttl:
            # Set TTL on the run key too
            await self._client.expire(run_key, ttl)

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        data = await self._client.get(self._ckpt_key(checkpoint_id))
        if data is None:
            return None
        return Checkpoint.from_json(data)

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        """Load the latest checkpoint for a run."""
        run_key = self._run_key(run_id)

        # Get highest sequence checkpoint
        results = await self._client.zrevrange(run_key, 0, 0)
        if not results:
            return None

        checkpoint_id = results[0]
        if isinstance(checkpoint_id, bytes):
            checkpoint_id = checkpoint_id.decode()

        return await self.load(checkpoint_id)

    async def list_checkpoints(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a run."""
        run_key = self._run_key(run_id)

        # Get checkpoint IDs sorted by sequence descending
        ckpt_ids = await self._client.zrevrange(run_key, 0, limit - 1)

        results = []
        for ckpt_id in ckpt_ids:
            if isinstance(ckpt_id, bytes):
                ckpt_id = ckpt_id.decode()
            ckpt = await self.load(ckpt_id)
            if ckpt:
                results.append(ckpt.metadata)

        return results

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        ckpt = await self.load(checkpoint_id)
        if ckpt:
            await self._client.delete(self._ckpt_key(checkpoint_id))
            await self._client.zrem(
                self._run_key(ckpt.run_id),
                checkpoint_id,
            )

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run."""
        run_key = self._run_key(run_id)

        # Get all checkpoint IDs
        ckpt_ids = await self._client.zrange(run_key, 0, -1)

        count = 0
        for ckpt_id in ckpt_ids:
            if isinstance(ckpt_id, bytes):
                ckpt_id = ckpt_id.decode()
            await self._client.delete(self._ckpt_key(ckpt_id))
            count += 1

        await self._client.delete(run_key)
        return count

    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints.

        Note: Redis handles TTL-based expiry automatically.
        This method cleans up orphaned run keys.
        """
        # Redis handles expiry via TTL
        # This could scan for orphaned keys if needed
        return 0


class S3CheckpointStore:
    """S3-based checkpoint store for durable storage.

    Requires aioboto3 or similar async S3 client.

    Example:
        import aioboto3
        session = aioboto3.Session()
        async with session.client("s3") as s3:
            store = S3CheckpointStore(s3, bucket="my-checkpoints")
    """

    def __init__(
        self,
        client: Any,  # aioboto3 S3 client
        bucket: str,
        prefix: str = "checkpoints/",
    ) -> None:
        """Initialize S3 checkpoint store.

        Args:
            client: Async S3 client
            bucket: S3 bucket name
            prefix: Key prefix for checkpoint objects
        """
        self._client = client
        self._bucket = bucket
        self._prefix = prefix

    def _ckpt_key(self, run_id: str, checkpoint_id: str) -> str:
        """Get S3 key for a checkpoint."""
        return f"{self._prefix}{run_id}/{checkpoint_id}.json"

    def _manifest_key(self, run_id: str) -> str:
        """Get S3 key for run manifest."""
        return f"{self._prefix}{run_id}/_manifest.json"

    async def _load_manifest(self, run_id: str) -> dict[str, Any]:
        """Load run manifest."""
        try:
            response = await self._client.get_object(
                Bucket=self._bucket,
                Key=self._manifest_key(run_id),
            )
            data = await response["Body"].read()
            return json.loads(data)
        except (json.JSONDecodeError, KeyError):
            return {"checkpoints": []}

    async def _save_manifest(
        self,
        run_id: str,
        manifest: dict[str, Any],
    ) -> None:
        """Save run manifest."""
        await self._client.put_object(
            Bucket=self._bucket,
            Key=self._manifest_key(run_id),
            Body=json.dumps(manifest),
            ContentType="application/json",
        )

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        # Save checkpoint object
        await self._client.put_object(
            Bucket=self._bucket,
            Key=self._ckpt_key(checkpoint.run_id, checkpoint.checkpoint_id),
            Body=checkpoint.to_json(),
            ContentType="application/json",
            Metadata={
                "sequence": str(checkpoint.metadata.sequence),
                "created_at": str(checkpoint.metadata.created_at),
            },
        )

        # Update manifest
        manifest = await self._load_manifest(checkpoint.run_id)
        ckpt_entry = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "sequence": checkpoint.metadata.sequence,
            "created_at": checkpoint.metadata.created_at,
        }

        # Add or update
        found = False
        for i, entry in enumerate(manifest["checkpoints"]):
            if entry["checkpoint_id"] == checkpoint.checkpoint_id:
                manifest["checkpoints"][i] = ckpt_entry
                found = True
                break
        if not found:
            manifest["checkpoints"].append(ckpt_entry)

        # Sort by sequence
        manifest["checkpoints"].sort(key=lambda x: x["sequence"], reverse=True)

        await self._save_manifest(checkpoint.run_id, manifest)

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        # Need to find which run it belongs to
        # List prefixes to find the run
        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=self._prefix,
            Delimiter="/",
        ):
            for prefix in page.get("CommonPrefixes", []):
                run_prefix = prefix["Prefix"]
                run_id = run_prefix.rstrip("/").split("/")[-1]
                try:
                    response = await self._client.get_object(
                        Bucket=self._bucket,
                        Key=self._ckpt_key(run_id, checkpoint_id),
                    )
                    data = await response["Body"].read()
                    return Checkpoint.from_json(data)
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
        return None

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        """Load the latest checkpoint for a run."""
        manifest = await self._load_manifest(run_id)
        checkpoints = manifest.get("checkpoints", [])

        if not checkpoints:
            return None

        # First entry is latest (sorted by sequence desc)
        latest_id = checkpoints[0]["checkpoint_id"]

        try:
            response = await self._client.get_object(
                Bucket=self._bucket,
                Key=self._ckpt_key(run_id, latest_id),
            )
            data = await response["Body"].read()
            return Checkpoint.from_json(data)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    async def list_checkpoints(
        self,
        run_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a run."""
        manifest = await self._load_manifest(run_id)
        checkpoints = manifest.get("checkpoints", [])[:limit]

        results = []
        for entry in checkpoints:
            try:
                response = await self._client.get_object(
                    Bucket=self._bucket,
                    Key=self._ckpt_key(run_id, entry["checkpoint_id"]),
                )
                data = await response["Body"].read()
                ckpt = Checkpoint.from_json(data)
                results.append(ckpt.metadata)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        return results

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        # Find and delete
        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=self._prefix,
            Delimiter="/",
        ):
            for prefix in page.get("CommonPrefixes", []):
                run_prefix = prefix["Prefix"]
                run_id = run_prefix.rstrip("/").split("/")[-1]
                key = self._ckpt_key(run_id, checkpoint_id)
                try:
                    await self._client.delete_object(
                        Bucket=self._bucket,
                        Key=key,
                    )
                    # Update manifest
                    manifest = await self._load_manifest(run_id)
                    manifest["checkpoints"] = [
                        c for c in manifest["checkpoints"] if c["checkpoint_id"] != checkpoint_id
                    ]
                    await self._save_manifest(run_id, manifest)
                    return
                except (KeyError, json.JSONDecodeError):
                    continue

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run."""
        # List all objects with run prefix
        prefix = f"{self._prefix}{run_id}/"
        paginator = self._client.get_paginator("list_objects_v2")

        objects_to_delete = []
        async for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=prefix,
        ):
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})

        if objects_to_delete:
            await self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": objects_to_delete},
            )

        return len(objects_to_delete)

    async def cleanup_expired(self) -> int:
        """Clean up expired checkpoints."""
        deleted = 0

        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=self._prefix,
            Delimiter="/",
        ):
            for prefix in page.get("CommonPrefixes", []):
                run_prefix = prefix["Prefix"]
                run_id = run_prefix.rstrip("/").split("/")[-1]

                manifest = await self._load_manifest(run_id)
                expired_ids = []

                for entry in manifest.get("checkpoints", []):
                    try:
                        response = await self._client.get_object(
                            Bucket=self._bucket,
                            Key=self._ckpt_key(run_id, entry["checkpoint_id"]),
                        )
                        data = await response["Body"].read()
                        ckpt = Checkpoint.from_json(data)
                        if ckpt.metadata.is_expired:
                            expired_ids.append(entry["checkpoint_id"])
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

                for ckpt_id in expired_ids:
                    await self._client.delete_object(
                        Bucket=self._bucket,
                        Key=self._ckpt_key(run_id, ckpt_id),
                    )
                    deleted += 1

                # Update manifest
                if expired_ids:
                    manifest["checkpoints"] = [
                        c for c in manifest["checkpoints"] if c["checkpoint_id"] not in expired_ids
                    ]
                    await self._save_manifest(run_id, manifest)

        return deleted
