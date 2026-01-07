"""Scope-based access control for FastAgentic."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastagentic.policy.base import Policy, PolicyContext, PolicyResult


@dataclass
class Scope:
    """A scope defines a permission boundary.

    Scopes follow OAuth2/OIDC conventions (e.g., "tools:read", "endpoints:invoke").

    Attributes:
        name: Scope identifier (e.g., "tools:read")
        description: Human-readable description
        implies: Other scopes this scope implies (inheritance)
    """

    name: str
    description: str = ""
    implies: list[str] = field(default_factory=list)

    def matches(self, required: str) -> bool:
        """Check if this scope satisfies the required scope.

        Args:
            required: The required scope string

        Returns:
            True if this scope grants the required access
        """
        # Exact match
        if self.name == required:
            return True

        # Wildcard: "tools:*" matches "tools:read", "tools:invoke", etc.
        if self.name.endswith(":*"):
            prefix = self.name[:-1]  # "tools:"
            if required.startswith(prefix):
                return True

        # Global wildcard
        return self.name == "*"


class ScopePolicy(Policy):
    """Scope-based access control policy.

    Validates that requests have required OAuth2-style scopes.

    Example:
        policy = ScopePolicy()

        # Define scopes with inheritance
        policy.add_scope(Scope(
            name="admin",
            description="Full administrative access",
            implies=["tools:*", "endpoints:*", "resources:*"]
        ))
        policy.add_scope(Scope(
            name="tools:read",
            description="Read tool metadata"
        ))
        policy.add_scope(Scope(
            name="tools:invoke",
            description="Invoke tools",
            implies=["tools:read"]  # Invoking implies reading
        ))

        # Use in app
        app = App(policies=[policy])
    """

    def __init__(
        self,
        *,
        allow_empty_scopes: bool = False,
        scope_claim: str = "scope",
    ) -> None:
        """Initialize scope policy.

        Args:
            allow_empty_scopes: Whether to allow requests with no scopes
            scope_claim: JWT claim name for scopes (default: "scope")
        """
        self._scopes: dict[str, Scope] = {}
        self._allow_empty_scopes = allow_empty_scopes
        self._scope_claim = scope_claim

    @property
    def name(self) -> str:
        return "scopes"

    @property
    def priority(self) -> int:
        return 90  # After RBAC (100), before budget (50)

    def add_scope(self, scope: Scope) -> None:
        """Add a scope definition.

        Args:
            scope: The scope to add
        """
        self._scopes[scope.name] = scope

    def remove_scope(self, scope_name: str) -> None:
        """Remove a scope definition.

        Args:
            scope_name: Name of the scope to remove
        """
        self._scopes.pop(scope_name, None)

    def get_scope(self, scope_name: str) -> Scope | None:
        """Get a scope by name.

        Args:
            scope_name: Name of the scope

        Returns:
            The scope or None if not found
        """
        return self._scopes.get(scope_name)

    def expand_scopes(self, scope_names: list[str]) -> set[str]:
        """Expand scopes to include implied scopes.

        Args:
            scope_names: List of scope names to expand

        Returns:
            Set of all effective scopes (including implied)
        """
        expanded: set[str] = set()
        to_process = list(scope_names)
        processed: set[str] = set()

        while to_process:
            scope_name = to_process.pop()
            if scope_name in processed:
                continue
            processed.add(scope_name)
            expanded.add(scope_name)

            # Add implied scopes
            scope = self._scopes.get(scope_name)
            if scope:
                for implied in scope.implies:
                    if implied not in processed:
                        to_process.append(implied)
                        expanded.add(implied)

        return expanded

    def _check_scope(
        self,
        required: str,
        user_scopes: set[str],
    ) -> bool:
        """Check if user scopes satisfy the required scope.

        Args:
            required: Required scope
            user_scopes: Set of user's scopes (already expanded)

        Returns:
            True if access is granted
        """
        # Check direct match
        if required in user_scopes:
            return True

        # Check wildcard matches
        for user_scope in user_scopes:
            scope = Scope(name=user_scope)
            if scope.matches(required):
                return True

        return False

    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        """Evaluate scope policy.

        Args:
            ctx: Policy evaluation context

        Returns:
            PolicyResult with allow/deny decision
        """
        # If no required scopes, allow
        if not ctx.required_scopes:
            return PolicyResult.allow()

        # Get user scopes
        user_scopes = ctx.user_scopes
        if not user_scopes:
            if self._allow_empty_scopes:
                return PolicyResult.allow()
            return PolicyResult.deny("No scopes provided")

        # Expand user scopes to include implied scopes
        expanded_scopes = self.expand_scopes(user_scopes)

        # Check each required scope
        missing_scopes: list[str] = []
        for required in ctx.required_scopes:
            if not self._check_scope(required, expanded_scopes):
                missing_scopes.append(required)

        if missing_scopes:
            return PolicyResult.deny(f"Missing required scopes: {', '.join(missing_scopes)}")

        return PolicyResult.allow("All required scopes present")


# Predefined common scopes
COMMON_SCOPES = {
    # Tool scopes
    "tools:read": Scope(
        name="tools:read",
        description="Read tool definitions and metadata",
    ),
    "tools:invoke": Scope(
        name="tools:invoke",
        description="Invoke tools",
        implies=["tools:read"],
    ),
    "tools:*": Scope(
        name="tools:*",
        description="Full tool access",
        implies=["tools:read", "tools:invoke"],
    ),
    # Endpoint scopes
    "endpoints:read": Scope(
        name="endpoints:read",
        description="Read endpoint metadata",
    ),
    "endpoints:invoke": Scope(
        name="endpoints:invoke",
        description="Invoke agent endpoints",
        implies=["endpoints:read"],
    ),
    "endpoints:*": Scope(
        name="endpoints:*",
        description="Full endpoint access",
        implies=["endpoints:read", "endpoints:invoke"],
    ),
    # Resource scopes
    "resources:read": Scope(
        name="resources:read",
        description="Read resources",
    ),
    "resources:write": Scope(
        name="resources:write",
        description="Write resources",
        implies=["resources:read"],
    ),
    "resources:*": Scope(
        name="resources:*",
        description="Full resource access",
        implies=["resources:read", "resources:write"],
    ),
    # Admin scope
    "admin": Scope(
        name="admin",
        description="Full administrative access",
        implies=["tools:*", "endpoints:*", "resources:*"],
    ),
}


def create_scope_policy_with_defaults() -> ScopePolicy:
    """Create a ScopePolicy with common predefined scopes.

    Returns:
        ScopePolicy with common scopes pre-registered
    """
    policy = ScopePolicy()
    for scope in COMMON_SCOPES.values():
        policy.add_scope(scope)
    return policy
