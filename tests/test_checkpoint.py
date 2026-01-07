"""Tests for checkpoint module."""

import tempfile
import time

import pytest

from fastagentic.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointManager,
    CheckpointMetadata,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from fastagentic.checkpoint.base import CheckpointStatus

# ============================================================================
# CheckpointMetadata Tests
# ============================================================================


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_create_metadata(self):
        """Test creating metadata."""
        meta = CheckpointMetadata(run_id="run-123")
        assert meta.run_id == "run-123"
        assert meta.checkpoint_id.startswith("ckpt-")
        assert meta.sequence == 0
        assert meta.status == CheckpointStatus.ACTIVE

    def test_metadata_with_ttl(self):
        """Test metadata with TTL."""
        meta = CheckpointMetadata(
            run_id="run-123",
            ttl_seconds=3600,
        )
        assert meta.ttl_seconds == 3600
        assert not meta.is_expired

    def test_metadata_expired(self):
        """Test expired metadata."""
        meta = CheckpointMetadata(
            run_id="run-123",
            created_at=time.time() - 100,
            ttl_seconds=60,  # 60 seconds TTL, created 100 seconds ago
        )
        assert meta.is_expired

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        meta = CheckpointMetadata(
            run_id="run-123",
            sequence=5,
            step_name="processing",
            tags=["important"],
        )
        data = meta.to_dict()
        assert data["run_id"] == "run-123"
        assert data["sequence"] == 5
        assert data["step_name"] == "processing"
        assert "important" in data["tags"]

    def test_metadata_from_dict(self):
        """Test metadata deserialization."""
        data = {
            "run_id": "run-456",
            "checkpoint_id": "ckpt-abc123",
            "sequence": 3,
            "created_at": 1234567890.0,
            "status": "completed",
            "step_name": "done",
            "parent_id": None,
            "tags": [],
            "ttl_seconds": None,
        }
        meta = CheckpointMetadata.from_dict(data)
        assert meta.run_id == "run-456"
        assert meta.checkpoint_id == "ckpt-abc123"
        assert meta.sequence == 3
        assert meta.status == CheckpointStatus.COMPLETED


# ============================================================================
# Checkpoint Tests
# ============================================================================


class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        ckpt = Checkpoint(
            metadata=CheckpointMetadata(run_id="run-123"),
            state={"step": 1},
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert ckpt.run_id == "run-123"
        assert ckpt.state == {"step": 1}
        assert len(ckpt.messages) == 1

    def test_checkpoint_to_dict(self):
        """Test checkpoint serialization."""
        ckpt = Checkpoint(
            metadata=CheckpointMetadata(run_id="run-123"),
            state={"counter": 5},
            messages=[{"role": "assistant", "content": "Hi"}],
            tool_calls=[{"name": "search", "args": {}}],
            context={"user_id": "user-1"},
        )
        data = ckpt.to_dict()
        assert data["state"]["counter"] == 5
        assert len(data["messages"]) == 1
        assert len(data["tool_calls"]) == 1
        assert data["context"]["user_id"] == "user-1"

    def test_checkpoint_from_dict(self):
        """Test checkpoint deserialization."""
        data = {
            "metadata": {
                "run_id": "run-789",
                "checkpoint_id": "ckpt-xyz",
                "sequence": 2,
                "created_at": 1234567890.0,
                "status": "active",
                "step_name": "",
                "parent_id": None,
                "tags": [],
                "ttl_seconds": None,
            },
            "state": {"progress": 50},
            "messages": [],
            "tool_calls": [],
            "context": {},
        }
        ckpt = Checkpoint.from_dict(data)
        assert ckpt.run_id == "run-789"
        assert ckpt.state["progress"] == 50

    def test_checkpoint_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = Checkpoint(
            metadata=CheckpointMetadata(run_id="run-123", sequence=5),
            state={"key": "value"},
            messages=[{"role": "user", "content": "test"}],
        )
        json_str = original.to_json()
        restored = Checkpoint.from_json(json_str)

        assert restored.run_id == original.run_id
        assert restored.metadata.sequence == original.metadata.sequence
        assert restored.state == original.state
        assert restored.messages == original.messages


# ============================================================================
# InMemoryCheckpointStore Tests
# ============================================================================


class TestInMemoryCheckpointStore:
    """Tests for InMemoryCheckpointStore."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading a checkpoint."""
        store = InMemoryCheckpointStore()
        ckpt = Checkpoint(
            metadata=CheckpointMetadata(run_id="run-123"),
            state={"x": 1},
        )

        await store.save(ckpt)
        loaded = await store.load(ckpt.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == ckpt.checkpoint_id
        assert loaded.state == {"x": 1}

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        """Test loading nonexistent checkpoint."""
        store = InMemoryCheckpointStore()
        loaded = await store.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_latest(self):
        """Test loading latest checkpoint for run."""
        store = InMemoryCheckpointStore()

        # Create multiple checkpoints
        for i in range(3):
            ckpt = Checkpoint(
                metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                state={"seq": i},
            )
            await store.save(ckpt)

        latest = await store.load_latest("run-123")
        assert latest is not None
        assert latest.metadata.sequence == 2
        assert latest.state["seq"] == 2

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints for run."""
        store = InMemoryCheckpointStore()

        for i in range(5):
            ckpt = Checkpoint(
                metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                state={},
            )
            await store.save(ckpt)

        checkpoints = await store.list_checkpoints("run-123")
        assert len(checkpoints) == 5
        # Should be sorted by sequence descending
        assert checkpoints[0].sequence == 4
        assert checkpoints[4].sequence == 0

    @pytest.mark.asyncio
    async def test_list_with_limit(self):
        """Test listing with limit."""
        store = InMemoryCheckpointStore()

        for i in range(10):
            ckpt = Checkpoint(
                metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                state={},
            )
            await store.save(ckpt)

        checkpoints = await store.list_checkpoints("run-123", limit=3)
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a checkpoint."""
        store = InMemoryCheckpointStore()
        ckpt = Checkpoint(
            metadata=CheckpointMetadata(run_id="run-123"),
            state={},
        )
        await store.save(ckpt)

        await store.delete(ckpt.checkpoint_id)
        loaded = await store.load(ckpt.checkpoint_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_run(self):
        """Test deleting all checkpoints for a run."""
        store = InMemoryCheckpointStore()

        for i in range(5):
            ckpt = Checkpoint(
                metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                state={},
            )
            await store.save(ckpt)

        count = await store.delete_run("run-123")
        assert count == 5

        checkpoints = await store.list_checkpoints("run-123")
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired checkpoints."""
        store = InMemoryCheckpointStore()

        # Create expired checkpoint
        expired = Checkpoint(
            metadata=CheckpointMetadata(
                run_id="run-123",
                created_at=time.time() - 100,
                ttl_seconds=60,
            ),
            state={},
        )
        await store.save(expired)

        # Create valid checkpoint
        valid = Checkpoint(
            metadata=CheckpointMetadata(run_id="run-456"),
            state={},
        )
        await store.save(valid)

        count = await store.cleanup_expired()
        assert count == 1

        # Expired should be gone
        assert await store.load(expired.checkpoint_id) is None
        # Valid should remain
        assert await store.load(valid.checkpoint_id) is not None


# ============================================================================
# FileCheckpointStore Tests
# ============================================================================


class TestFileCheckpointStore:
    """Tests for FileCheckpointStore."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            ckpt = Checkpoint(
                metadata=CheckpointMetadata(run_id="run-123"),
                state={"data": "test"},
            )

            await store.save(ckpt)
            loaded = await store.load(ckpt.checkpoint_id)

            assert loaded is not None
            assert loaded.state == {"data": "test"}

    @pytest.mark.asyncio
    async def test_load_latest(self):
        """Test loading latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)

            for i in range(3):
                ckpt = Checkpoint(
                    metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                    state={"seq": i},
                )
                await store.save(ckpt)

            latest = await store.load_latest("run-123")
            assert latest is not None
            assert latest.metadata.sequence == 2

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)

            for i in range(5):
                ckpt = Checkpoint(
                    metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                    state={},
                )
                await store.save(ckpt)

            checkpoints = await store.list_checkpoints("run-123")
            assert len(checkpoints) == 5

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            ckpt = Checkpoint(
                metadata=CheckpointMetadata(run_id="run-123"),
                state={},
            )
            await store.save(ckpt)

            await store.delete(ckpt.checkpoint_id)
            loaded = await store.load(ckpt.checkpoint_id)
            assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_run(self):
        """Test deleting all checkpoints for a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)

            for i in range(3):
                ckpt = Checkpoint(
                    metadata=CheckpointMetadata(run_id="run-123", sequence=i),
                    state={},
                )
                await store.save(ckpt)

            count = await store.delete_run("run-123")
            assert count == 3

            checkpoints = await store.list_checkpoints("run-123")
            assert len(checkpoints) == 0


# ============================================================================
# CheckpointManager Tests
# ============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.mark.asyncio
    async def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        ckpt = await manager.create(
            run_id="run-123",
            state={"progress": 50},
            messages=[{"role": "user", "content": "test"}],
        )

        assert ckpt is not None
        assert ckpt.run_id == "run-123"
        assert ckpt.state["progress"] == 50

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self):
        """Test restoring a checkpoint."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        await manager.create(
            run_id="run-123",
            state={"step": 1},
        )
        await manager.create(
            run_id="run-123",
            state={"step": 2},
        )

        restored = await manager.restore("run-123")
        assert restored is not None
        assert restored.state["step"] == 2  # Latest

    @pytest.mark.asyncio
    async def test_restore_from_specific(self):
        """Test restoring from specific checkpoint."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        ckpt1 = await manager.create(
            run_id="run-123",
            state={"step": 1},
        )
        await manager.create(
            run_id="run-123",
            state={"step": 2},
        )

        restored = await manager.restore_from(ckpt1.checkpoint_id)
        assert restored is not None
        assert restored.state["step"] == 1

    @pytest.mark.asyncio
    async def test_sequence_numbers(self):
        """Test sequence numbers increment."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        ckpt1 = await manager.create("run-123", {}, force=True)
        ckpt2 = await manager.create("run-123", {}, force=True)
        ckpt3 = await manager.create("run-123", {}, force=True)

        assert ckpt1.metadata.sequence == 0
        assert ckpt2.metadata.sequence == 1
        assert ckpt3.metadata.sequence == 2

    @pytest.mark.asyncio
    async def test_checkpoint_interval(self):
        """Test checkpoint interval is respected."""
        store = InMemoryCheckpointStore()
        config = CheckpointConfig(checkpoint_interval=1.0)  # 1 second
        manager = CheckpointManager(store, config)

        ckpt1 = await manager.create("run-123", {})
        assert ckpt1 is not None

        # Second checkpoint should be skipped (interval not met)
        ckpt2 = await manager.create("run-123", {})
        assert ckpt2 is None

        # Force should work
        ckpt3 = await manager.create("run-123", {}, force=True)
        assert ckpt3 is not None

    @pytest.mark.asyncio
    async def test_max_checkpoints_pruning(self):
        """Test old checkpoints are pruned."""
        store = InMemoryCheckpointStore()
        config = CheckpointConfig(max_checkpoints_per_run=3)
        manager = CheckpointManager(store, config)

        # Create 5 checkpoints
        for i in range(5):
            await manager.create("run-123", {"i": i}, force=True)

        checkpoints = await manager.list_checkpoints("run-123")
        assert len(checkpoints) == 3
        # Should keep the newest ones
        assert checkpoints[0].sequence == 4
        assert checkpoints[2].sequence == 2

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        ckpt = await manager.create("run-123", {})
        await manager.delete(ckpt.checkpoint_id)

        restored = await manager.restore_from(ckpt.checkpoint_id)
        assert restored is None

    @pytest.mark.asyncio
    async def test_delete_run(self):
        """Test deleting all checkpoints for a run."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        for i in range(3):
            await manager.create("run-123", {}, force=True)

        count = await manager.delete_run("run-123")
        assert count == 3

        checkpoints = await manager.list_checkpoints("run-123")
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_mark_completed(self):
        """Test marking run as completed."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        await manager.create("run-123", {"final": True})
        await manager.mark_completed("run-123")

        ckpt = await manager.restore("run-123")
        assert ckpt.metadata.status == CheckpointStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_mark_failed(self):
        """Test marking run as failed."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        await manager.create("run-123", {})
        await manager.mark_failed("run-123", error="Something went wrong")

        ckpt = await manager.restore("run-123")
        assert ckpt.metadata.status == CheckpointStatus.FAILED
        assert ckpt.context["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_checkpoint_with_context(self):
        """Test checkpoint with additional context."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        ckpt = await manager.create(
            run_id="run-123",
            state={"step": 1},
            context={"user_id": "user-456", "session": "sess-789"},
            step_name="processing",
            tags=["important", "retry"],
        )

        assert ckpt.context["user_id"] == "user-456"
        assert ckpt.metadata.step_name == "processing"
        assert "important" in ckpt.metadata.tags

    @pytest.mark.asyncio
    async def test_parent_checkpoint(self):
        """Test checkpoint with parent."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        parent = await manager.create("run-123", {"step": 1})
        child = await manager.create(
            "run-123",
            {"step": 2},
            parent_id=parent.checkpoint_id,
            force=True,
        )

        assert child.metadata.parent_id == parent.checkpoint_id

    @pytest.mark.asyncio
    async def test_restore_expired_returns_none(self):
        """Test restoring expired checkpoint returns None."""
        store = InMemoryCheckpointStore()
        manager = CheckpointManager(store)

        # Create expired checkpoint
        ckpt = Checkpoint(
            metadata=CheckpointMetadata(
                run_id="run-123",
                created_at=time.time() - 100,
                ttl_seconds=60,
            ),
            state={},
        )
        await store.save(ckpt)

        restored = await manager.restore("run-123")
        assert restored is None


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CheckpointConfig()
        assert config.auto_checkpoint is True
        assert config.checkpoint_interval == 0.0
        assert config.max_checkpoints_per_run == 100
        assert config.default_ttl_seconds is None
        assert config.cleanup_interval == 300.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CheckpointConfig(
            auto_checkpoint=False,
            checkpoint_interval=5.0,
            max_checkpoints_per_run=10,
            default_ttl_seconds=3600,
        )
        assert config.auto_checkpoint is False
        assert config.checkpoint_interval == 5.0
        assert config.max_checkpoints_per_run == 10
        assert config.default_ttl_seconds == 3600
