"""Redis-based memory providers."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from fastagentic.memory.base import MemoryProvider, SessionMemory


class RedisProvider(MemoryProvider):
    """Simple Redis-based memory provider.

    Stores memories as JSON in Redis with key-based retrieval.
    Does not support semantic search - use Mem0 or Zep for that.

    Example:
        from fastagentic.memory import RedisProvider

        memory = RedisProvider(
            url="redis://localhost:6379",
            prefix="memory:",
            ttl_seconds=None,  # No expiration
        )
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "memory:",
        ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the Redis provider.

        Args:
            url: Redis connection URL
            prefix: Key prefix for memory storage
            ttl_seconds: Optional TTL for memories
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

    def _key(self, user_id: str, memory_id: str | None = None) -> str:
        """Generate a Redis key."""
        if memory_id:
            return f"{self.prefix}{user_id}:{memory_id}"
        return f"{self.prefix}{user_id}:*"

    async def add(
        self,
        user_id: str,
        content: str | list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a memory to Redis."""
        client = await self._get_client()
        memory_id = str(uuid.uuid4())

        memory = {
            "id": memory_id,
            "user_id": user_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }

        key = self._key(user_id, memory_id)
        if self.ttl_seconds:
            await client.setex(key, self.ttl_seconds, json.dumps(memory))
        else:
            await client.set(key, json.dumps(memory))

        return memory_id

    async def search(
        self,
        user_id: str,
        _query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories (returns all - no semantic search)."""
        # Redis provider doesn't support semantic search
        # Returns all memories, filtered by metadata if provided
        memories = await self.get_all(user_id, limit=limit)

        if filters:
            memories = [
                m
                for m in memories
                if all(m.get("metadata", {}).get(k) == v for k, v in filters.items())
            ]

        return memories[:limit]

    async def get(self, user_id: str, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory."""
        client = await self._get_client()
        key = self._key(user_id, memory_id)
        data = await client.get(key)
        if data:
            return json.loads(data)
        return None

    async def get_all(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get all memories for a user."""
        client = await self._get_client()
        pattern = self._key(user_id)
        keys = []

        async for key in client.scan_iter(pattern):
            keys.append(key)
            if len(keys) >= limit:
                break

        memories = []
        for key in keys:
            data = await client.get(key)
            if data:
                memories.append(json.loads(data))

        return memories

    async def update(
        self,
        user_id: str,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an existing memory."""
        memory = await self.get(user_id, memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        if content is not None:
            memory["content"] = content
        if metadata is not None:
            memory["metadata"] = {**memory.get("metadata", {}), **metadata}
        memory["updated_at"] = datetime.utcnow().isoformat()

        client = await self._get_client()
        key = self._key(user_id, memory_id)
        if self.ttl_seconds:
            await client.setex(key, self.ttl_seconds, json.dumps(memory))
        else:
            await client.set(key, json.dumps(memory))

    async def delete(self, user_id: str, memory_id: str) -> None:
        """Delete a specific memory."""
        client = await self._get_client()
        key = self._key(user_id, memory_id)
        await client.delete(key)

    async def delete_all(self, user_id: str) -> int:
        """Delete all memories for a user."""
        client = await self._get_client()
        pattern = self._key(user_id)
        count = 0

        async for key in client.scan_iter(pattern):
            await client.delete(key)
            count += 1

        return count


class RedisSessionMemory(SessionMemory):
    """Redis-based session memory.

    Stores conversation history in Redis with TTL support.

    Example:
        from fastagentic.memory import RedisSessionMemory

        session_memory = RedisSessionMemory(
            url="redis://localhost:6379",
            ttl_seconds=3600,  # 1 hour
            max_messages=50,
        )
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "session:",
        ttl_seconds: int = 3600,
        max_messages: int = 50,
    ) -> None:
        """Initialize Redis session memory.

        Args:
            url: Redis connection URL
            prefix: Key prefix for session storage
            ttl_seconds: TTL for sessions
            max_messages: Max messages to keep per session
        """
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create the Redis client."""
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self.url)
        return self._client

    def _key(self, session_id: str) -> str:
        """Generate a Redis key for session."""
        return f"{self.prefix}{session_id}"

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get conversation history."""
        client = await self._get_client()
        key = self._key(session_id)
        data = await client.get(key)
        if data:
            return json.loads(data)
        return []

    async def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Add a message to session history."""
        messages = await self.get_messages(session_id)
        messages.append(message)

        # Trim to max messages
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages :]

        client = await self._get_client()
        key = self._key(session_id)
        await client.setex(key, self.ttl_seconds, json.dumps(messages))

    async def clear(self, session_id: str) -> None:
        """Clear session history."""
        client = await self._get_client()
        key = self._key(session_id)
        await client.delete(key)
