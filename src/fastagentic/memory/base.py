"""Base memory provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryProvider(ABC):
    """Abstract base class for memory providers.

    Memory providers handle long-term memory storage and retrieval
    for agent personalization and context.

    Example:
        class MyProvider(MemoryProvider):
            async def add(self, user_id: str, content: str, metadata: dict = None) -> str:
                # Store the memory
                ...

            async def search(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
                # Search memories
                ...
    """

    @abstractmethod
    async def add(
        self,
        user_id: str,
        content: str | list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add memory content.

        Args:
            user_id: User identifier for scoping
            content: Text or messages to memorize
            metadata: Optional metadata (category, tags, etc.)

        Returns:
            Memory ID
        """
        ...

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories.

        Args:
            user_id: User identifier
            query: Search query
            limit: Max results
            filters: Optional filters (category, date range, etc.)

        Returns:
            List of memory objects with relevance score
        """
        ...

    @abstractmethod
    async def get(self, user_id: str, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory by ID.

        Args:
            user_id: User identifier
            memory_id: Memory identifier

        Returns:
            Memory object or None if not found
        """
        ...

    @abstractmethod
    async def get_all(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get all memories for a user.

        Args:
            user_id: User identifier
            limit: Max results

        Returns:
            List of memory objects
        """
        ...

    @abstractmethod
    async def update(
        self,
        user_id: str,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an existing memory.

        Args:
            user_id: User identifier
            memory_id: Memory identifier
            content: New content (optional)
            metadata: New metadata (optional)
        """
        ...

    @abstractmethod
    async def delete(self, user_id: str, memory_id: str) -> None:
        """Delete a specific memory.

        Args:
            user_id: User identifier
            memory_id: Memory identifier
        """
        ...

    @abstractmethod
    async def delete_all(self, user_id: str) -> int:
        """Delete all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        ...


class SessionMemory(ABC):
    """Abstract base class for session memory.

    Session memory handles within-conversation context,
    separate from long-term memory.
    """

    @abstractmethod
    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of message objects
        """
        ...

    @abstractmethod
    async def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Add a message to session history.

        Args:
            session_id: Session identifier
            message: Message object with role and content
        """
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear session history.

        Args:
            session_id: Session identifier
        """
        ...

    async def summarize(self, _session_id: str) -> str:
        """Get a summary of the conversation.

        Default implementation returns empty string.
        Override for summarization support.

        Args:
            session_id: Session identifier

        Returns:
            Summary text
        """
        return ""
