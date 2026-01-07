"""Tests for Redis memory provider."""

import json
from unittest.mock import AsyncMock

import pytest

from fastagentic.memory.redis import RedisProvider


class TestRedisProvider:
    """Tests for RedisProvider."""

    @pytest.fixture
    def provider(self):
        """Create a Redis provider."""
        return RedisProvider(
            url="redis://localhost:6379",
            prefix="test:memory:",
            ttl_seconds=None,
        )

    def test_init_default_values(self):
        """Test default initialization values."""
        provider = RedisProvider()
        assert provider.url == "redis://localhost:6379"
        assert provider.prefix == "memory:"
        assert provider.ttl_seconds is None

    def test_init_custom_values(self):
        """Test custom initialization values."""
        provider = RedisProvider(
            url="redis://localhost:6380",
            prefix="custom:",
            ttl_seconds=3600,
        )
        assert provider.url == "redis://localhost:6380"
        assert provider.prefix == "custom:"
        assert provider.ttl_seconds == 3600

    def test_key_generation(self, provider):
        """Test Redis key generation."""
        # With memory_id
        key = provider._key("user123", "memory456")
        assert key == "test:memory:user123:memory456"

        # Without memory_id (pattern)
        key = provider._key("user123")
        assert key == "test:memory:user123:*"

    def test_key_with_custom_prefix(self):
        """Test key generation with custom prefix."""
        provider = RedisProvider(prefix="myapp:")
        key = provider._key("user1", "mem1")
        assert key == "myapp:user1:mem1"

    def test_key_with_special_characters(self):
        """Test key generation with special characters in user_id."""
        provider = RedisProvider(prefix="mem:")

        # User IDs with special characters
        key = provider._key("user@example.com", "mem1")
        assert key == "mem:user@example.com:mem1"

        key = provider._key("user/name space", "mem2")
        assert key == "mem:user/name space:mem2"

    def test_key_with_empty_prefix(self):
        """Test key generation with empty prefix."""
        provider = RedisProvider(prefix="")
        key = provider._key("user1", "mem1")
        assert key == "user1:mem1"

    def test_key_pattern_without_memory_id(self):
        """Test key pattern for listing."""
        provider = RedisProvider(prefix="app:")
        key = provider._key("user123")
        assert key == "app:user123:*"


class TestRedisProviderAsync:
    """Tests for RedisProvider async methods."""

    @pytest.fixture
    def provider(self):
        """Create a Redis provider."""
        return RedisProvider(
            url="redis://localhost:6379",
            prefix="test:memory:",
            ttl_seconds=None,
        )

    @pytest.fixture
    def mock_client(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_add_memory(self, provider, mock_client):
        """Test adding a memory to Redis."""
        provider._client = mock_client

        memory_id = await provider.add(
            user_id="user123",
            content="Test memory content",
            metadata={"source": "test"},
        )

        assert memory_id is not None
        assert len(memory_id) == 36  # UUID format

        # Verify set was called
        mock_client.set.assert_called_once()

        # Verify the stored data
        call_args = mock_client.set.call_args
        stored_data = json.loads(call_args[0][1])
        assert stored_data["content"] == "Test memory content"
        assert stored_data["metadata"]["source"] == "test"

    @pytest.mark.asyncio
    async def test_add_memory_with_ttl(self):
        """Test adding a memory with TTL."""
        provider = RedisProvider(
            url="redis://localhost:6379",
            prefix="test:",
            ttl_seconds=3600,
        )
        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.add(
            user_id="user123",
            content="Temporary memory",
        )

        # Verify setex was called instead of set
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_get_memory(self, provider, mock_client):
        """Test getting a specific memory."""
        provider._client = mock_client

        # Mock the get method
        memory_data = {
            "id": "memory123",
            "user_id": "user123",
            "content": "Test content",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00",
        }
        mock_client.get.return_value = json.dumps(memory_data)

        result = await provider.get("user123", "memory123")

        assert result is not None
        assert result["id"] == "memory123"
        assert result["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, provider, mock_client):
        """Test getting non-existent memory returns None."""
        provider._client = mock_client
        mock_client.get.return_value = None

        result = await provider.get("user123", "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_memory(self, provider, mock_client):
        """Test deleting a memory."""
        provider._client = mock_client

        await provider.delete("user123", "memory123")

        # Verify delete was called
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert "test:memory:user123:memory123" in str(call_args)

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, provider, mock_client):
        """Test deleting non-existent memory doesn't raise."""
        provider._client = mock_client
        mock_client.delete.return_value = 0

        # Should not raise, just call delete
        await provider.delete("user123", "nonexistent")
        mock_client.delete.assert_called_once()
