"""Context objects for agent runs and requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastagentic.app import App
    from fastagentic.memory import MemoryProvider


@dataclass
class UserInfo:
    """Information about the authenticated user."""

    id: str
    email: str | None = None
    name: str | None = None
    roles: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token: str | None = None


@dataclass
class UsageInfo:
    """Token usage and cost information."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    model: str | None = None
    latency_ms: float | None = None

    def add(self, other: UsageInfo) -> None:
        """Add usage from another UsageInfo object."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.cost += other.cost


@dataclass
class RunContext:
    """Context for a single agent run.

    This context is passed through the entire execution of an agent run
    and provides access to run metadata, user info, and shared state.
    """

    run_id: str
    endpoint: str
    user: UserInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: UsageInfo = field(default_factory=UsageInfo)

    # Internal state
    _checkpoints: list[dict[str, Any]] = field(default_factory=list)
    _is_resumed: bool = False

    @property
    def is_authenticated(self) -> bool:
        """Check if the run has an authenticated user."""
        return self.user is not None

    @property
    def is_resumed(self) -> bool:
        """Check if this run was resumed from a checkpoint."""
        return self._is_resumed

    def add_checkpoint(self, state: dict[str, Any]) -> None:
        """Add a checkpoint to the run."""
        self._checkpoints.append(state)


@dataclass
class AgentContext:
    """Full context available to agent endpoints.

    Extends RunContext with access to application services like memory,
    durable storage, and configuration.
    """

    run: RunContext
    app: App
    request: Any = None  # FastAPI Request object

    @property
    def run_id(self) -> str:
        """Shortcut to run.run_id."""
        return self.run.run_id

    @property
    def user(self) -> UserInfo | None:
        """Shortcut to run.user."""
        return self.run.user

    @property
    def usage(self) -> UsageInfo:
        """Shortcut to run.usage."""
        return self.run.usage

    @property
    def metadata(self) -> dict[str, Any]:
        """Shortcut to run.metadata."""
        return self.run.metadata

    @property
    def memory(self) -> MemoryProvider | None:
        """Access the configured memory provider."""
        return self.app.memory if self.app else None

    @property
    def memories(self) -> list[dict[str, Any]]:
        """Get injected memories for this run."""
        return self.metadata.get("_memories", [])

    def set_memories(self, memories: list[dict[str, Any]]) -> None:
        """Set the injected memories for this run."""
        self.metadata["_memories"] = memories
