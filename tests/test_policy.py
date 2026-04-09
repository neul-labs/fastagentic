"""Tests for FastAgentic policy module."""

import pytest

from fastagentic.context import UserInfo
from fastagentic.policy.base import Policy, PolicyAction, PolicyContext, PolicyResult
from fastagentic.policy.budget import Budget, BudgetPeriod, BudgetPolicy, UsageRecord
from fastagentic.policy.engine import PolicyEngine, PolicyEngineConfig, create_default_engine
from fastagentic.policy.rbac import Permission, RBACPolicy, Role
from fastagentic.policy.scopes import (
    COMMON_SCOPES,
    Scope,
    ScopePolicy,
    create_scope_policy_with_defaults,
)


class TestPolicyAction:
    """Tests for PolicyAction enum."""

    def test_allow_value(self):
        assert PolicyAction.ALLOW.value == "allow"

    def test_deny_value(self):
        assert PolicyAction.DENY.value == "deny"

    def test_warn_value(self):
        assert PolicyAction.WARN.value == "warn"

    def test_limit_value(self):
        assert PolicyAction.LIMIT.value == "limit"


class TestPolicyResult:
    """Tests for PolicyResult class."""

    def test_allow_factory(self):
        result = PolicyResult.allow("Access granted")
        assert result.action == PolicyAction.ALLOW
        assert result.reason == "Access granted"
        assert result.is_allowed
        assert not result.is_denied

    def test_deny_factory(self):
        result = PolicyResult.deny("Access denied")
        assert result.action == PolicyAction.DENY
        assert result.reason == "Access denied"
        assert result.is_denied
        assert not result.is_allowed

    def test_warn_factory(self):
        result = PolicyResult.warn("Approaching limit")
        assert result.action == PolicyAction.WARN
        assert result.reason == "Approaching limit"
        assert result.is_allowed

    def test_limit_factory(self):
        result = PolicyResult.limit("Rate limited", {"rpm": 60})
        assert result.action == PolicyAction.LIMIT
        assert result.limits == {"rpm": 60}
        assert result.is_allowed


class TestPolicyContext:
    """Tests for PolicyContext class."""

    def test_minimal_context(self):
        ctx = PolicyContext()
        assert ctx.user is None
        assert ctx.tenant_id is None
        assert ctx.user_id is None
        assert not ctx.is_authenticated

    def test_authenticated_context(self):
        user = UserInfo(id="user-123", email="test@example.com")
        ctx = PolicyContext(user=user, tenant_id="tenant-456")
        assert ctx.user_id == "user-123"
        assert ctx.is_authenticated

    def test_request_context(self):
        ctx = PolicyContext(
            endpoint="/triage",
            action="invoke",
            resource="tool:summarize",
            required_scopes=["tools:invoke"],
            estimated_tokens=1000,
            estimated_cost=0.01,
        )
        assert ctx.endpoint == "/triage"
        assert ctx.action == "invoke"
        assert ctx.estimated_tokens == 1000


class TestPermission:
    """Tests for Permission class."""

    def test_exact_match(self):
        perm = Permission(resource="tools/search", actions=["invoke"])
        assert perm.matches("tools/search", "invoke")
        assert not perm.matches("tools/other", "invoke")

    def test_wildcard_resource(self):
        perm = Permission(resource="tools/*", actions=["invoke"])
        assert perm.matches("tools/search", "invoke")
        assert perm.matches("tools/summarize", "invoke")
        assert not perm.matches("endpoints/triage", "invoke")

    def test_global_wildcard(self):
        perm = Permission(resource="*", actions=["*"])
        assert perm.matches("anything", "any_action")
        assert perm.matches("tools/search", "invoke")

    def test_wildcard_action(self):
        perm = Permission(resource="tools/search", actions=["*"])
        assert perm.matches("tools/search", "invoke")
        assert perm.matches("tools/search", "read")

    def test_multiple_actions(self):
        perm = Permission(resource="tools/search", actions=["invoke", "read"])
        assert perm.matches("tools/search", "invoke")
        assert perm.matches("tools/search", "read")
        assert not perm.matches("tools/search", "write")


class TestRole:
    """Tests for Role class."""

    def test_basic_role(self):
        role = Role(
            name="viewer",
            permissions=[Permission(resource="tools/*", actions=["read"])],
        )
        assert role.name == "viewer"
        assert role.has_permission("tools/search", "read")
        assert not role.has_permission("tools/search", "invoke")

    def test_role_inheritance(self):
        viewer = Role(
            name="viewer",
            permissions=[Permission(resource="tools/*", actions=["read"])],
        )
        editor = Role(
            name="editor",
            inherits=["viewer"],
            permissions=[Permission(resource="tools/*", actions=["invoke"])],
        )
        all_roles = {"viewer": viewer, "editor": editor}

        # Editor has own permission
        assert editor.has_permission("tools/search", "invoke", all_roles)
        # Editor inherits viewer's permission
        assert editor.has_permission("tools/search", "read", all_roles)


class TestRBACPolicy:
    """Tests for RBACPolicy class."""

    @pytest.mark.asyncio
    async def test_unauthenticated_denied(self):
        policy = RBACPolicy(deny_unauthenticated=True)
        ctx = PolicyContext(endpoint="/test")

        result = await policy.evaluate(ctx)
        assert result.is_denied
        assert "Authentication required" in result.reason

    @pytest.mark.asyncio
    async def test_unauthenticated_allowed(self):
        policy = RBACPolicy(deny_unauthenticated=False)
        ctx = PolicyContext(endpoint="/test")

        result = await policy.evaluate(ctx)
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_no_roles_denied(self):
        policy = RBACPolicy()
        ctx = PolicyContext(
            user=UserInfo(id="user-123"),
            endpoint="/test",
        )

        result = await policy.evaluate(ctx)
        assert result.is_denied
        assert "No roles assigned" in result.reason

    @pytest.mark.asyncio
    async def test_role_grants_access(self):
        policy = RBACPolicy()
        policy.add_role(
            Role(
                name="admin",
                permissions=[Permission(resource="*", actions=["*"])],
            )
        )
        policy.assign_role("user-123", "admin")

        ctx = PolicyContext(
            user=UserInfo(id="user-123"),
            resource="tools/search",
            action="invoke",
        )

        result = await policy.evaluate(ctx)
        assert result.is_allowed
        assert "Granted by role: admin" in result.reason

    @pytest.mark.asyncio
    async def test_default_role(self):
        policy = RBACPolicy(default_role="user")
        policy.add_role(
            Role(
                name="user",
                permissions=[Permission(resource="tools/*", actions=["invoke"])],
            )
        )

        ctx = PolicyContext(
            user=UserInfo(id="user-123"),
            resource="tools/search",
            action="invoke",
        )

        result = await policy.evaluate(ctx)
        assert result.is_allowed


class TestScope:
    """Tests for Scope class."""

    def test_exact_match(self):
        scope = Scope(name="tools:invoke")
        assert scope.matches("tools:invoke")
        assert not scope.matches("tools:read")

    def test_wildcard_match(self):
        scope = Scope(name="tools:*")
        assert scope.matches("tools:invoke")
        assert scope.matches("tools:read")
        assert not scope.matches("endpoints:invoke")

    def test_global_wildcard(self):
        scope = Scope(name="*")
        assert scope.matches("anything")
        assert scope.matches("tools:invoke")


class TestScopePolicy:
    """Tests for ScopePolicy class."""

    @pytest.mark.asyncio
    async def test_no_required_scopes(self):
        policy = ScopePolicy()
        ctx = PolicyContext(required_scopes=[])

        result = await policy.evaluate(ctx)
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_missing_scopes_denied(self):
        policy = ScopePolicy()
        ctx = PolicyContext(
            required_scopes=["tools:invoke"],
            user_scopes=[],
        )

        result = await policy.evaluate(ctx)
        assert result.is_denied
        assert "No scopes provided" in result.reason

    @pytest.mark.asyncio
    async def test_matching_scopes_allowed(self):
        policy = ScopePolicy()
        ctx = PolicyContext(
            required_scopes=["tools:invoke"],
            user_scopes=["tools:invoke"],
        )

        result = await policy.evaluate(ctx)
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_wildcard_scope_grants_access(self):
        policy = ScopePolicy()
        ctx = PolicyContext(
            required_scopes=["tools:invoke"],
            user_scopes=["tools:*"],
        )

        result = await policy.evaluate(ctx)
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_scope_inheritance(self):
        policy = ScopePolicy()
        policy.add_scope(
            Scope(
                name="admin",
                implies=["tools:*", "endpoints:*"],
            )
        )
        policy.add_scope(Scope(name="tools:*", implies=["tools:read", "tools:invoke"]))

        ctx = PolicyContext(
            required_scopes=["tools:invoke"],
            user_scopes=["admin"],
        )

        expanded = policy.expand_scopes(["admin"])
        assert "tools:*" in expanded
        assert "tools:invoke" in expanded

    def test_common_scopes_defined(self):
        assert "tools:read" in COMMON_SCOPES
        assert "admin" in COMMON_SCOPES
        assert COMMON_SCOPES["admin"].implies

    def test_create_scope_policy_with_defaults(self):
        policy = create_scope_policy_with_defaults()
        assert policy.get_scope("tools:read") is not None
        assert policy.get_scope("admin") is not None


class TestBudget:
    """Tests for Budget class."""

    def test_budget_requires_limit(self):
        with pytest.raises(ValueError, match="At least one budget limit"):
            Budget()

    def test_budget_soft_limit_validation(self):
        with pytest.raises(ValueError, match="soft_limit_percent"):
            Budget(max_cost=100.0, soft_limit_percent=150)

    def test_valid_budget(self):
        budget = Budget(
            max_cost=100.0,
            max_tokens=1_000_000,
            max_requests=10000,
            period=BudgetPeriod.DAY,
        )
        assert budget.max_cost == 100.0
        assert budget.period == BudgetPeriod.DAY


class TestBudgetPeriod:
    """Tests for BudgetPeriod enum."""

    def test_period_seconds(self):
        assert BudgetPeriod.MINUTE.seconds() == 60
        assert BudgetPeriod.HOUR.seconds() == 3600
        assert BudgetPeriod.DAY.seconds() == 86400
        assert BudgetPeriod.WEEK.seconds() == 604800


class TestUsageRecord:
    """Tests for UsageRecord class."""

    def test_default_values(self):
        record = UsageRecord()
        assert record.cost == 0.0
        assert record.tokens == 0
        assert record.requests == 0

    def test_reset(self):
        record = UsageRecord(cost=10.0, tokens=1000, requests=5)
        record.reset()
        assert record.cost == 0.0
        assert record.tokens == 0
        assert record.requests == 0


class TestBudgetPolicy:
    """Tests for BudgetPolicy class."""

    @pytest.mark.asyncio
    async def test_no_budget_allows(self):
        policy = BudgetPolicy()
        ctx = PolicyContext(user=UserInfo(id="user-123"))

        result = await policy.evaluate(ctx)
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_global_budget_exceeded(self):
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_requests=2, period=BudgetPeriod.DAY))

        # Simulate previous usage
        await policy._store.increment_usage("global", requests=2)

        ctx = PolicyContext(user=UserInfo(id="user-123"))
        result = await policy.evaluate(ctx)

        assert result.is_denied
        assert "Request limit exceeded" in result.reason

    @pytest.mark.asyncio
    async def test_user_budget_exceeded(self):
        policy = BudgetPolicy()
        policy.set_user_budget("user-123", Budget(max_cost=1.0, period=BudgetPeriod.DAY))

        await policy._store.increment_usage("user:user-123", cost=1.5)

        ctx = PolicyContext(user=UserInfo(id="user-123"))
        result = await policy.evaluate(ctx)

        assert result.is_denied
        assert "Cost budget exceeded" in result.reason

    @pytest.mark.asyncio
    async def test_budget_warning(self):
        policy = BudgetPolicy()
        policy.set_global_budget(
            Budget(
                max_cost=100.0,
                soft_limit_percent=80,
                period=BudgetPeriod.DAY,
            )
        )

        await policy._store.increment_usage("global", cost=85.0)

        ctx = PolicyContext(user=UserInfo(id="user-123"))
        result = await policy.evaluate(ctx)

        assert result.action == PolicyAction.WARN
        assert "Approaching cost limit" in result.reason


class TestPolicyEngine:
    """Tests for PolicyEngine class."""

    @pytest.mark.asyncio
    async def test_empty_engine_allows(self):
        engine = PolicyEngine()
        ctx = PolicyContext()

        decision = await engine.evaluate(ctx)
        assert decision.is_allowed

    @pytest.mark.asyncio
    async def test_deny_short_circuits(self):
        engine = PolicyEngine()

        # Create a denying policy
        class DenyPolicy(Policy):
            @property
            def name(self) -> str:
                return "deny"

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                return PolicyResult.deny("Always denied")

        class AllowPolicy(Policy):
            @property
            def name(self) -> str:
                return "allow"

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                return PolicyResult.allow()

        engine.add_policy(DenyPolicy())
        engine.add_policy(AllowPolicy())

        ctx = PolicyContext()
        decision = await engine.evaluate(ctx)

        assert decision.is_denied

    @pytest.mark.asyncio
    async def test_policy_priority(self):
        engine = PolicyEngine()

        class LowPriorityPolicy(Policy):
            @property
            def name(self) -> str:
                return "low"

            @property
            def priority(self) -> int:
                return 10

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                return PolicyResult.allow()

        class HighPriorityPolicy(Policy):
            @property
            def name(self) -> str:
                return "high"

            @property
            def priority(self) -> int:
                return 100

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                return PolicyResult.deny("High priority denied")

        engine.add_policy(LowPriorityPolicy())
        engine.add_policy(HighPriorityPolicy())

        # High priority should be evaluated first
        assert engine.policies[0].name == "high"

    @pytest.mark.asyncio
    async def test_warnings_collected(self):
        engine = PolicyEngine(
            config=PolicyEngineConfig(short_circuit_deny=False),
        )

        class WarnPolicy(Policy):
            @property
            def name(self) -> str:
                return "warn"

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                return PolicyResult.warn("Warning message")

        engine.add_policy(WarnPolicy())

        ctx = PolicyContext()
        decision = await engine.evaluate(ctx)

        assert decision.action == PolicyAction.WARN
        assert len(decision.warnings) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self):
        engine = PolicyEngine()

        decision = await engine.evaluate_with_context(
            user_id="user-123",
            tenant_id="tenant-456",
            endpoint="/test",
            action="invoke",
        )

        assert decision.is_allowed

    def test_create_default_engine(self):
        engine = create_default_engine(rbac=True, scopes=True, budget=False)

        assert engine.get_policy("rbac") is not None
        assert engine.get_policy("scopes") is not None
        assert engine.get_policy("budget") is None

    def test_add_remove_policy(self):
        engine = PolicyEngine()

        class TestPolicy(Policy):
            @property
            def name(self) -> str:
                return "test"

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                return PolicyResult.allow()

        engine.add_policy(TestPolicy())
        assert engine.get_policy("test") is not None

        engine.remove_policy("test")
        assert engine.get_policy("test") is None
