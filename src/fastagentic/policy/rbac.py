"""Role-Based Access Control (RBAC) for FastAgentic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastagentic.policy.base import Policy, PolicyContext, PolicyResult


@dataclass
class Permission:
    """A permission grants access to a resource/action.

    Attributes:
        resource: Resource pattern (e.g., "tools/*", "endpoints/triage")
        actions: Allowed actions (e.g., ["invoke", "read"])
        conditions: Optional conditions for the permission
    """

    resource: str
    actions: list[str] = field(default_factory=lambda: ["*"])
    conditions: dict[str, Any] = field(default_factory=dict)

    def matches(self, resource: str | None, action: str | None) -> bool:
        """Check if this permission matches the resource and action.

        Args:
            resource: The resource being accessed
            action: The action being performed

        Returns:
            True if the permission grants access
        """
        if not self._matches_resource(resource):
            return False
        return self._matches_action(action)

    def _matches_resource(self, resource: str | None) -> bool:
        """Check if resource matches the permission pattern."""
        if self.resource == "*":
            return True
        if resource is None:
            return False

        # Handle wildcard patterns
        if self.resource.endswith("/*"):
            prefix = self.resource[:-2]
            return resource.startswith(prefix)
        if self.resource.endswith("*"):
            prefix = self.resource[:-1]
            return resource.startswith(prefix)

        return self.resource == resource

    def _matches_action(self, action: str | None) -> bool:
        """Check if action matches the permission."""
        if "*" in self.actions:
            return True
        if action is None:
            return True  # No specific action required
        return action in self.actions


@dataclass
class Role:
    """A role groups permissions together.

    Roles can inherit from other roles to build hierarchies.

    Example:
        viewer = Role(
            name="viewer",
            permissions=[
                Permission(resource="tools/*", actions=["read"]),
            ]
        )

        editor = Role(
            name="editor",
            inherits=["viewer"],
            permissions=[
                Permission(resource="tools/*", actions=["invoke"]),
            ]
        )
    """

    name: str
    permissions: list[Permission] = field(default_factory=list)
    inherits: list[str] = field(default_factory=list)
    description: str = ""

    def has_permission(
        self,
        resource: str | None,
        action: str | None,
        all_roles: dict[str, Role] | None = None,
    ) -> bool:
        """Check if this role grants access to the resource/action.

        Args:
            resource: The resource being accessed
            action: The action being performed
            all_roles: Dict of all roles (for inheritance lookup)

        Returns:
            True if access is granted
        """
        # Check direct permissions
        for perm in self.permissions:
            if perm.matches(resource, action):
                return True

        # Check inherited roles
        if all_roles:
            for parent_name in self.inherits:
                parent = all_roles.get(parent_name)
                if parent and parent.has_permission(resource, action, all_roles):
                    return True

        return False


class RBACPolicy(Policy):
    """Role-Based Access Control policy.

    Evaluates access based on user roles and permissions.

    Example:
        # Define roles
        rbac = RBACPolicy()
        rbac.add_role(Role(
            name="admin",
            permissions=[Permission(resource="*", actions=["*"])]
        ))
        rbac.add_role(Role(
            name="user",
            permissions=[
                Permission(resource="tools/*", actions=["invoke"]),
                Permission(resource="endpoints/*", actions=["invoke"]),
            ]
        ))

        # Assign roles to users
        rbac.assign_role("user-123", "admin")
        rbac.assign_role("user-456", "user")

        # Use in app
        app = App(policies=[rbac])
    """

    def __init__(
        self,
        *,
        default_role: str | None = None,
        deny_unauthenticated: bool = True,
    ) -> None:
        """Initialize RBAC policy.

        Args:
            default_role: Role assigned to users without explicit assignment
            deny_unauthenticated: Whether to deny unauthenticated requests
        """
        self._roles: dict[str, Role] = {}
        self._user_roles: dict[str, list[str]] = {}
        self._default_role = default_role
        self._deny_unauthenticated = deny_unauthenticated

    @property
    def name(self) -> str:
        return "rbac"

    @property
    def priority(self) -> int:
        return 100  # High priority - auth checks first

    def add_role(self, role: Role) -> None:
        """Add a role definition.

        Args:
            role: The role to add
        """
        self._roles[role.name] = role

    def remove_role(self, role_name: str) -> None:
        """Remove a role definition.

        Args:
            role_name: Name of the role to remove
        """
        self._roles.pop(role_name, None)

    def get_role(self, role_name: str) -> Role | None:
        """Get a role by name.

        Args:
            role_name: Name of the role

        Returns:
            The role or None if not found
        """
        return self._roles.get(role_name)

    def assign_role(self, user_id: str, role_name: str) -> None:
        """Assign a role to a user.

        Args:
            user_id: The user ID
            role_name: Name of the role to assign
        """
        if user_id not in self._user_roles:
            self._user_roles[user_id] = []
        if role_name not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role_name)

    def revoke_role(self, user_id: str, role_name: str) -> None:
        """Revoke a role from a user.

        Args:
            user_id: The user ID
            role_name: Name of the role to revoke
        """
        if user_id in self._user_roles:
            self._user_roles[user_id] = [r for r in self._user_roles[user_id] if r != role_name]

    def get_user_roles(self, user_id: str) -> list[str]:
        """Get roles assigned to a user.

        Args:
            user_id: The user ID

        Returns:
            List of role names
        """
        roles = self._user_roles.get(user_id, [])
        if not roles and self._default_role:
            roles = [self._default_role]
        return roles

    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        """Evaluate RBAC policy.

        Args:
            ctx: Policy evaluation context

        Returns:
            PolicyResult with allow/deny decision
        """
        # Check authentication
        if not ctx.is_authenticated:
            if self._deny_unauthenticated:
                return PolicyResult.deny("Authentication required")
            return PolicyResult.allow()

        # Get user roles
        user_id = ctx.user_id
        if not user_id:
            return PolicyResult.deny("User ID required")

        roles = self.get_user_roles(user_id)
        if not roles:
            return PolicyResult.deny(f"No roles assigned to user {user_id}")

        # Check permissions
        for role_name in roles:
            role = self._roles.get(role_name)
            if role and role.has_permission(ctx.resource, ctx.action, self._roles):
                return PolicyResult.allow(f"Granted by role: {role_name}")

        return PolicyResult.deny(f"No permission for {ctx.action} on {ctx.resource}")
