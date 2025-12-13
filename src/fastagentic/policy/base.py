"""Base policy classes for FastAgentic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastagentic.context import UserInfo


class PolicyAction(str, Enum):
    """Action to take after policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"  # Allow but log warning
    LIMIT = "limit"  # Allow with restrictions


@dataclass
class PolicyResult:
    """Result of a policy evaluation.

    Attributes:
        action: The action to take (allow, deny, warn, limit)
        reason: Human-readable explanation
        details: Additional context for logging/debugging
        limits: Any limits to apply (for LIMIT action)
    """

    action: PolicyAction
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, reason: str | None = None) -> PolicyResult:
        """Create an allow result."""
        return cls(action=PolicyAction.ALLOW, reason=reason)

    @classmethod
    def deny(cls, reason: str) -> PolicyResult:
        """Create a deny result."""
        return cls(action=PolicyAction.DENY, reason=reason)

    @classmethod
    def warn(cls, reason: str) -> PolicyResult:
        """Create a warn result (allow with warning)."""
        return cls(action=PolicyAction.WARN, reason=reason)

    @classmethod
    def limit(cls, reason: str, limits: dict[str, Any]) -> PolicyResult:
        """Create a limit result (allow with restrictions)."""
        return cls(action=PolicyAction.LIMIT, reason=reason, limits=limits)

    @property
    def is_allowed(self) -> bool:
        """Check if the action allows the request to proceed."""
        return self.action in (PolicyAction.ALLOW, PolicyAction.WARN, PolicyAction.LIMIT)

    @property
    def is_denied(self) -> bool:
        """Check if the action denies the request."""
        return self.action == PolicyAction.DENY


@dataclass
class PolicyContext:
    """Context for policy evaluation.

    Contains all information needed to evaluate policies:
    user identity, requested resource, action, and metadata.
    """

    # Identity
    user: UserInfo | None = None
    tenant_id: str | None = None

    # Request
    endpoint: str | None = None
    action: str | None = None  # "invoke", "read", "write", etc.
    resource: str | None = None  # Tool name, resource URI, etc.

    # Scopes
    required_scopes: list[str] = field(default_factory=list)
    user_scopes: list[str] = field(default_factory=list)

    # Usage context
    estimated_tokens: int = 0
    estimated_cost: float = 0.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def user_id(self) -> str | None:
        """Get user ID if available."""
        return self.user.id if self.user else None

    @property
    def is_authenticated(self) -> bool:
        """Check if request is authenticated."""
        return self.user is not None


class Policy(ABC):
    """Base class for policies.

    Policies evaluate requests and return allow/deny decisions.
    Multiple policies can be combined in a PolicyEngine.

    Example:
        class MyPolicy(Policy):
            @property
            def name(self) -> str:
                return "my-policy"

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                if ctx.user_id in self.blocked_users:
                    return PolicyResult.deny("User is blocked")
                return PolicyResult.allow()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging and identification."""
        ...

    @property
    def priority(self) -> int:
        """Policy priority (higher = evaluated first). Default is 0."""
        return 0

    @property
    def enabled(self) -> bool:
        """Whether this policy is enabled."""
        return True

    @abstractmethod
    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        """Evaluate the policy for the given context.

        Args:
            ctx: The policy evaluation context

        Returns:
            PolicyResult indicating allow/deny and reason
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
