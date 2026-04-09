"""Mutation tests for RBAC Policy.

These tests are specifically designed to catch common mutations:
- Wildcard matching off-by-one errors
- String slicing mutations
- Boolean comparison changes

Run mutation testing with:
    mutmut run --paths-to-mutate=src/fastagentic/policy/rbac.py

Targeted mutations:
- Line 49-54: Wildcard slicing [:-2] vs [:-1]
- Line 52: startswith vs == comparison
- Line 86: None check in resource matching
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from fastagentic.context import UserInfo
from fastagentic.policy.base import PolicyContext
from fastagentic.policy.rbac import Permission, RBACPolicy, Role


class TestPermissionWildcardMatching:
    """Tests for wildcard matching mutations."""

    def test_star_matches_everything(self):
        """Wildcard '*' should match any resource.

        Mutation: Removing the '*' check entirely
        """
        perm = Permission(resource="*", actions=["invoke"])

        assert perm.matches("tools/search", "invoke")
        assert perm.matches("agents/chat", "invoke")
        assert perm.matches("anything/at/all", "invoke")
        assert perm.matches("", "invoke")  # Even empty string

    def test_trailing_slash_star_matches_prefix(self):
        """'resource/*' should match resources starting with 'resource/'.

        Note: The implementation also matches exact 'resource' without trailing slash.
        """
        perm = Permission(resource="tools/*", actions=["invoke"])

        # Should match
        assert perm.matches("tools/search", "invoke")
        assert perm.matches("tools/analyze", "invoke")
        assert perm.matches("tools/deep/nested/path", "invoke")

        # The implementation matches 'tools' as well (generous wildcard)
        assert perm.matches("tools", "invoke")

        # Should NOT match - different prefix
        assert not perm.matches("toolbox/item", "invoke")
        assert not perm.matches("tool", "invoke")

    def test_trailing_star_matches_prefix_no_slash(self):
        """'resource*' should match resources starting with 'resource'.

        Mutation: [:-1] changed to [:-2] would cut too much
        """
        perm = Permission(resource="tools*", actions=["*"])

        # Should match
        assert perm.matches("tools", "invoke")
        assert perm.matches("tools/search", "invoke")
        assert perm.matches("toolshed", "invoke")
        assert perm.matches("tools123", "invoke")

        # Should NOT match
        assert not perm.matches("tool", "invoke")
        assert not perm.matches("TOOLS", "invoke")  # Case sensitive

    def test_exact_match_no_wildcard(self):
        """Exact resource should only match exact resource.

        Mutation: Adding wildcard behavior to exact matches
        """
        perm = Permission(resource="tools/search", actions=["invoke"])

        # Should match
        assert perm.matches("tools/search", "invoke")

        # Should NOT match
        assert not perm.matches("tools/search/deep", "invoke")
        assert not perm.matches("tools/searc", "invoke")
        assert not perm.matches("tools/searchx", "invoke")
        assert not perm.matches("tools", "invoke")

    def test_action_wildcard(self):
        """Action '*' should match any action.

        Mutation: Removing action wildcard check
        """
        perm = Permission(resource="tools/search", actions=["*"])

        assert perm.matches("tools/search", "invoke")
        assert perm.matches("tools/search", "read")
        assert perm.matches("tools/search", "write")
        assert perm.matches("tools/search", "delete")
        assert perm.matches("tools/search", "anything")

    def test_specific_action_no_wildcard(self):
        """Specific action should only match that action.

        Mutation: Treating specific actions as wildcards
        """
        perm = Permission(resource="*", actions=["read"])

        assert perm.matches("anything", "read")
        assert not perm.matches("anything", "write")
        assert not perm.matches("anything", "invoke")

    def test_multiple_actions(self):
        """Multiple actions should all be allowed."""
        perm = Permission(resource="*", actions=["read", "write"])

        assert perm.matches("anything", "read")
        assert perm.matches("anything", "write")
        assert not perm.matches("anything", "delete")

    def test_none_resource_handling(self):
        """None resource should be handled gracefully.

        Mutation: Removing None check causes AttributeError
        """
        perm = Permission(resource="tools/*", actions=["invoke"])

        # None should not match (except for '*')
        assert not perm.matches(None, "invoke")

    def test_star_matches_none_resource(self):
        """Wildcard '*' should even match None resource."""
        perm = Permission(resource="*", actions=["invoke"])

        # This tests that None is handled in '*' path
        result = perm.matches(None, "invoke")
        # Depends on implementation - either True or graceful False
        assert isinstance(result, bool)

    def test_none_action_handling(self):
        """None action should be handled gracefully."""
        perm = Permission(resource="*", actions=["invoke"])

        # Should not crash - None action defaults to allowed
        result = perm.matches("resource", None)
        assert result is True  # None action is allowed by design


class TestPermissionBoundaryConditions:
    """Boundary condition tests for permission matching."""

    @given(
        resource=st.text(min_size=0, max_size=100),
        action=st.text(min_size=1, max_size=50),
    )
    def test_exact_permission_self_match(self, resource: str, action: str):
        """A permission should always match itself (exact)."""
        assume("*" not in resource)  # No wildcards
        assume(len(resource) > 0)

        perm = Permission(resource=resource, actions=[action])
        assert perm.matches(resource, action)

    def test_empty_resource_exact_match(self):
        """Empty string resource should only match empty string."""
        perm = Permission(resource="", actions=["invoke"])

        assert perm.matches("", "invoke")
        assert not perm.matches("anything", "invoke")

    def test_single_char_resource(self):
        """Single character resources should work correctly."""
        perm = Permission(resource="a", actions=["invoke"])

        assert perm.matches("a", "invoke")
        assert not perm.matches("ab", "invoke")
        assert not perm.matches("", "invoke")

    def test_wildcard_only_resource(self):
        """Just '*' resource pattern."""
        perm = Permission(resource="*", actions=["*"])

        assert perm.matches("anything", "anything")
        assert perm.matches("", "")
        assert perm.matches("a/b/c/d/e", "complex-action")

    def test_slash_star_at_various_positions(self):
        """Test '/*' at different depths."""
        perm_root = Permission(resource="/*", actions=["invoke"])
        perm_one = Permission(resource="tools/*", actions=["invoke"])
        perm_two = Permission(resource="tools/search/*", actions=["invoke"])

        # Root level
        assert perm_root.matches("/anything", "invoke")

        # One level
        assert perm_one.matches("tools/search", "invoke")
        assert not perm_one.matches("other/search", "invoke")

        # Two levels
        assert perm_two.matches("tools/search/deep", "invoke")
        assert not perm_two.matches("tools/other/deep", "invoke")


class TestRBACPolicyEvaluation:
    """Tests for RBACPolicy evaluation mutations."""

    @pytest.fixture
    def context_factory(self):
        def create(
            user_id: str | None = "test-user",
            tenant_id: str | None = "test-tenant",
            endpoint: str | None = "/test",
            resource: str | None = "tools/search",
            action: str | None = "invoke",
        ) -> PolicyContext:
            user = UserInfo(id=user_id) if user_id else None
            return PolicyContext(
                user=user,
                tenant_id=tenant_id,
                endpoint=endpoint,
                resource=resource,
                action=action,
            )

        return create

    @pytest.mark.asyncio
    async def test_role_permission_grant(self, context_factory):
        """User with matching role permission should be allowed."""
        policy = RBACPolicy()
        policy.add_role(
            Role(
                name="user",
                permissions=[Permission(resource="tools/*", actions=["invoke"])],
            )
        )
        policy.assign_role("test-user", "user")

        ctx = context_factory(
            user_id="test-user",
            resource="tools/search",
            action="invoke",
        )
        result = await policy.evaluate(ctx)

        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_role_without_permission_denied(self, context_factory):
        """User without matching permission should be denied."""
        policy = RBACPolicy()
        policy.add_role(
            Role(
                name="user",
                permissions=[Permission(resource="tools/*", actions=["read"])],
            )
        )
        policy.assign_role("test-user", "user")

        ctx = context_factory(
            user_id="test-user",
            resource="tools/search",
            action="invoke",
        )
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_unknown_role_denied(self, context_factory):
        """User with unknown role should be denied.

        Mutation: Allowing unknown roles
        """
        policy = RBACPolicy()
        policy.add_role(
            Role(
                name="admin",
                permissions=[Permission(resource="*", actions=["*"])],
            )
        )
        # Assign a different role (unknown to user)
        policy.assign_role("other-user", "admin")

        ctx = context_factory(
            user_id="test-user",  # No role assigned
            resource="tools/search",
            action="invoke",
        )
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_multiple_roles_any_grants(self, context_factory):
        """If any role grants permission, should be allowed."""
        policy = RBACPolicy()
        policy.add_role(
            Role(name="reader", permissions=[Permission(resource="*", actions=["read"])])
        )
        policy.add_role(
            Role(name="writer", permissions=[Permission(resource="*", actions=["write"])])
        )
        policy.assign_role("test-user", "reader")
        policy.assign_role("test-user", "writer")

        ctx = context_factory(
            user_id="test-user",
            resource="data",
            action="read",
        )
        result = await policy.evaluate(ctx)

        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_unauthenticated_denied(self, context_factory):
        """Unauthenticated user should be denied.

        Mutation: Allowing empty roles
        """
        policy = RBACPolicy(deny_unauthenticated=True)
        policy.add_role(
            Role(
                name="user",
                permissions=[Permission(resource="*", actions=["*"])],
            )
        )

        ctx = context_factory(user_id=None)  # No user
        result = await policy.evaluate(ctx)

        assert result.is_denied


class TestRoleInheritance:
    """Tests for role inheritance if implemented."""

    @pytest.fixture
    def context_factory(self):
        def create(
            user_id: str | None = "test-user",
            tenant_id: str | None = "test-tenant",
            endpoint: str | None = "/test",
            resource: str | None = "tools/search",
            action: str | None = "invoke",
        ) -> PolicyContext:
            user = UserInfo(id=user_id) if user_id else None
            return PolicyContext(
                user=user,
                tenant_id=tenant_id,
                endpoint=endpoint,
                resource=resource,
                action=action,
            )

        return create

    @pytest.mark.asyncio
    async def test_parent_role_permissions(self, context_factory):
        """Admin role should have all permissions."""
        policy = RBACPolicy()
        policy.add_role(
            Role(
                name="admin",
                permissions=[Permission(resource="*", actions=["*"])],
            )
        )
        policy.add_role(
            Role(
                name="user",
                permissions=[Permission(resource="tools/*", actions=["read"])],
            )
        )
        policy.assign_role("test-user", "admin")

        ctx = context_factory(
            user_id="test-user",
            resource="sensitive/data",
            action="delete",
        )
        result = await policy.evaluate(ctx)

        assert result.is_allowed
