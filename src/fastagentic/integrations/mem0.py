"""Mem0 memory integration for FastAgentic.

Mem0 provides intelligent memory for AI agents with:
- Automatic memory extraction from conversations
- Semantic search over memories
- User and agent-specific memory scopes
- Memory consolidation and decay

https://mem0.ai

Example:
    from fastagentic import App
    from fastagentic.integrations import Mem0Integration

    app = App(
        title="My Agent",
        integrations=[
            Mem0Integration(
                api_key="m0-...",
                user_id_from="context",  # Extract from user context
            )
        ]
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastagentic.hooks.base import Hook, HookContext, HookResult
from fastagentic.integrations.base import Integration, IntegrationConfig
from fastagentic.memory.base import MemoryProvider

if TYPE_CHECKING:
    from fastagentic.app import App

try:
    from mem0 import Memory, MemoryClient

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    Memory = None
    MemoryClient = None


@dataclass
class Mem0Config(IntegrationConfig):
    """Configuration for Mem0 integration."""

    org_id: str | None = None
    project_id: str | None = None
    user_id_from: str = "context"  # "context", "header", or callable
    auto_add: bool = True
    auto_search: bool = True
    search_limit: int = 5
    search_threshold: float = 0.7
    memory_types: list[str] = field(default_factory=lambda: ["user", "agent"])


class Mem0Hook(Hook):
    """Hook for Mem0 memory operations.

    Automatically:
    - Searches relevant memories before LLM calls
    - Adds new memories from conversation
    """

    def __init__(self, client: Any, config: Mem0Config) -> None:
        self.client = client
        self.config = config

    def _get_user_id(self, ctx: HookContext) -> str | None:
        """Extract user ID based on configuration."""
        if self.config.user_id_from == "context":
            return ctx.user.user_id if ctx.user else None
        elif callable(self.config.user_id_from):
            return self.config.user_id_from(ctx)
        return None

    async def on_llm_start(self, ctx: HookContext) -> HookResult:
        """Search memories and inject into context."""
        if not self.client or not self.config.auto_search:
            return HookResult.proceed()

        if not ctx.messages:
            return HookResult.proceed()

        user_id = self._get_user_id(ctx)
        if not user_id:
            return HookResult.proceed()

        # Get the last user message as query
        user_messages = [m for m in ctx.messages if m.get("role") == "user"]
        if not user_messages:
            return HookResult.proceed()

        query = user_messages[-1].get("content", "")
        if isinstance(query, list):
            query = " ".join(p.get("text", "") for p in query if p.get("type") == "text")

        try:
            # Search memories
            results = self.client.search(
                query=query,
                user_id=user_id,
                limit=self.config.search_limit,
            )

            memories = results.get("results", []) if isinstance(results, dict) else results

            if memories:
                # Store in context for injection
                ctx.memory_results = [
                    {
                        "id": m.get("id"),
                        "content": m.get("memory"),
                        "score": m.get("score", 0),
                        "metadata": m.get("metadata", {}),
                    }
                    for m in memories
                    if m.get("score", 0) >= self.config.search_threshold
                ]

                ctx.metadata["mem0_memories_found"] = len(ctx.memory_results)

        except Exception as e:
            import structlog

            logger = structlog.get_logger()
            logger.warning("mem0_search_error", error=str(e))

        return HookResult.proceed()

    async def on_llm_end(self, ctx: HookContext) -> HookResult:
        """Add new memories from the conversation."""
        if not self.client or not self.config.auto_add:
            return HookResult.proceed()

        user_id = self._get_user_id(ctx)
        if not user_id:
            return HookResult.proceed()

        if not ctx.messages:
            return HookResult.proceed()

        try:
            # Format messages for memory addition
            formatted = []
            for msg in ctx.messages[-10:]:  # Last 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                formatted.append({"role": role, "content": content})

            # Add response if available
            if ctx.response:
                formatted.append({"role": "assistant", "content": str(ctx.response)})

            # Add to memory
            result = self.client.add(
                messages=formatted,
                user_id=user_id,
                metadata={
                    "run_id": ctx.run_id,
                    "endpoint": ctx.endpoint,
                },
            )

            ctx.metadata["mem0_memories_added"] = len(result.get("results", []))

        except Exception as e:
            import structlog

            logger = structlog.get_logger()
            logger.warning("mem0_add_error", error=str(e))

        return HookResult.proceed()


class Mem0MemoryProvider(MemoryProvider):
    """Memory provider backed by Mem0.

    Implements FastAgentic's MemoryProvider interface using Mem0.
    """

    def __init__(self, client: Any, config: Mem0Config) -> None:
        self.client = client
        self.config = config

    async def add(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a memory."""
        result = self.client.add(
            messages=[{"role": "user", "content": content}],
            user_id=user_id,
            metadata=metadata or {},
        )
        results = result.get("results", [])
        return results[0].get("id", "") if results else ""

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search memories."""
        results = self.client.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        memories = results.get("results", []) if isinstance(results, dict) else results

        return [
            {
                "id": m.get("id"),
                "content": m.get("memory"),
                "score": m.get("score", 0),
                "metadata": m.get("metadata", {}),
            }
            for m in memories
            if m.get("score", 0) >= threshold
        ]

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a specific memory."""
        try:
            result = self.client.get(memory_id)
            return {
                "id": result.get("id"),
                "content": result.get("memory"),
                "metadata": result.get("metadata", {}),
            }
        except Exception:
            return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            self.client.delete(memory_id)
            return True
        except Exception:
            return False

    async def get_all(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Get all memories for a user."""
        results = self.client.get_all(user_id=user_id)
        memories = results.get("results", []) if isinstance(results, dict) else results

        return [
            {
                "id": m.get("id"),
                "content": m.get("memory"),
                "metadata": m.get("metadata", {}),
            }
            for m in memories
        ]


class Mem0Integration(Integration):
    """Mem0 intelligent memory integration.

    Provides automatic memory extraction and retrieval for AI agents.
    Memories are automatically added from conversations and retrieved
    based on semantic relevance.

    Features:
    - **Automatic Extraction**: Extract memories from conversations
    - **Semantic Search**: Find relevant memories by meaning
    - **User Scoping**: Separate memories per user
    - **Memory Decay**: Older, less relevant memories fade
    - **Consolidation**: Similar memories are merged

    Example:
        # Basic usage
        app = App(
            integrations=[
                Mem0Integration(api_key="m0-...")
            ]
        )

        # With automatic memory injection
        app = App(
            integrations=[
                Mem0Integration(
                    api_key="m0-...",
                    auto_search=True,
                    auto_add=True,
                    search_limit=5,
                )
            ]
        )

        # As memory provider
        mem0 = Mem0Integration(api_key="m0-...")
        app = App(memory=mem0.as_memory_provider())

    Environment variables:
        MEM0_API_KEY: Mem0 API key
    """

    def __init__(
        self,
        api_key: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        auto_add: bool = True,
        auto_search: bool = True,
        search_limit: int = 5,
        search_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        config = Mem0Config(
            api_key=api_key,
            org_id=org_id,
            project_id=project_id,
            auto_add=auto_add,
            auto_search=auto_search,
            search_limit=search_limit,
            search_threshold=search_threshold,
            extra=kwargs,
        )
        super().__init__(config)
        self._config = config
        self._client: Any = None
        self._hook: Mem0Hook | None = None
        self._memory_provider: Mem0MemoryProvider | None = None

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        return MEM0_AVAILABLE

    def validate_config(self) -> list[str]:
        errors = super().validate_config()

        if not self.is_available():
            errors.append("mem0 package not installed. Run: pip install mem0ai")
            return errors

        import os

        api_key = self._config.api_key or os.getenv("MEM0_API_KEY")
        if not api_key:
            errors.append("Mem0 api_key is required")

        return errors

    def get_hooks(self) -> list[Hook]:
        if not self._hook:
            self._hook = Mem0Hook(self._client, self._config)
        return [self._hook]

    def setup(self, _app: App) -> None:
        """Initialize the Mem0 client."""
        if not self.is_available():
            return

        import os

        api_key = self._config.api_key or os.getenv("MEM0_API_KEY")

        if api_key:
            self._client = MemoryClient(api_key=api_key)

            # Update hook with client
            if self._hook:
                self._hook.client = self._client

    def get_client(self) -> Any:
        """Get the underlying Mem0 client for direct access."""
        return self._client

    def as_memory_provider(self) -> MemoryProvider:
        """Get a MemoryProvider implementation backed by Mem0.

        Use this to set Mem0 as the app's memory provider:

            mem0 = Mem0Integration(api_key="m0-...")
            app = App(memory=mem0.as_memory_provider())
        """
        if not self._memory_provider:
            self._memory_provider = Mem0MemoryProvider(self._client, self._config)
        return self._memory_provider

    async def add_memory(
        self,
        content: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Manually add a memory.

        Example:
            memory_id = await mem0.add_memory(
                "User prefers dark mode",
                user_id="user-123",
            )
        """
        if not self._client:
            raise RuntimeError("Mem0 client not initialized")

        result = self._client.add(
            messages=[{"role": "user", "content": content}],
            user_id=user_id,
            metadata=metadata or {},
        )
        results = result.get("results", [])
        return results[0].get("id", "") if results else ""

    async def search_memories(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories.

        Example:
            memories = await mem0.search_memories(
                "user preferences",
                user_id="user-123",
            )
        """
        if not self._client:
            raise RuntimeError("Mem0 client not initialized")

        results = self._client.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        memories = results.get("results", []) if isinstance(results, dict) else results

        return [
            {
                "id": m.get("id"),
                "content": m.get("memory"),
                "score": m.get("score", 0),
                "metadata": m.get("metadata", {}),
            }
            for m in memories
        ]

    async def on_shutdown(self) -> None:
        """Cleanup Mem0 client."""
        await super().on_shutdown()
        self._client = None
