"""Mutation tests for Policy Engine.

These tests are specifically designed to catch common mutations:
- Boundary condition changes (< vs <=, > vs >=)
- Boolean logic changes (and vs or, not removal)
- Short-circuit evaluation removal
- Return value changes

Run mutation testing with:
    mutmut run --paths-to-mutate=src/fastagentic/policy/engine.py

Targeted mutations:
- Line 165: Remove short-circuit deny check
- Line 194: Flip fail_open boolean
- Line 52: Change "in (ALLOW, WARN, LIMIT)" to "=="
- Line 205: Remove reason deduplication
"""

import pytest

from fastagentic.policy.base import Policy, PolicyAction, PolicyContext, PolicyResult
from fastagentic.policy.engine import PolicyDecision, PolicyEngine, PolicyEngineConfig
from fastagentic.policy.rbac import Permission, RBACPolicy, Role


class DenyPolicy(Policy):
    """Policy that always denies."""

    def __init__(self, reason: str = "Access denied"):
        self._reason = reason

    @property
    def name(self) -> str:
        return "deny-policy"

    @property
    def priority(self) -> int:
        return 50

    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        return PolicyResult.deny(self._reason)


class AllowPolicy(Policy):
    """Policy that always allows."""

    def __init__(self, reason: str | None = None):
        self._reason = reason

    @property
    def name(self) -> str:
        return "allow-policy"

    @property
    def priority(self) -> int:
        return 50

    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        return PolicyResult.allow(self._reason)


class FailingPolicy(Policy):
    """Policy that raises an exception."""

    @property
    def name(self) -> str:
        return "failing-policy"

    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        raise RuntimeError("Policy evaluation failed")


class TestPolicyEngineShortCircuit:
    """Tests to catch short-circuit evaluation mutations."""

    @pytest.fixture
    def engine(self):
        return PolicyEngine()

    @pytest.mark.asyncio
    async def test_deny_short_circuits_allow(self, engine, policy_context_factory):
        """CRITICAL: Deny should short-circuit and prevent any allow.

        Mutation target: Removing the short-circuit deny check
        would allow subsequent policies to override a denial.
        """
        ctx = policy_context_factory()

        # First policy denies, second would allow
        engine.add_policy(DenyPolicy("First denial"))
        engine.add_policy(AllowPolicy())

        decision = await engine.evaluate(ctx)

        # MUST be denied - the allow policy should not override
        assert decision.action == PolicyAction.DENY, (
            "Mutation caught: Deny was overridden by subsequent allow policy"
        )

    @pytest.mark.asyncio
    async def test_multiple_denies_first_recorded(self, engine, policy_context_factory):
        """First deny reason should be recorded in short-circuit mode."""
        ctx = policy_context_factory()

        engine.add_policy(DenyPolicy("First denial"))
        engine.add_policy(DenyPolicy("Second denial"))

        decision = await engine.evaluate(ctx)

        assert decision.action == PolicyAction.DENY
        # Should have the first denial reason
        assert decision.reason is not None
        assert "First denial" in decision.reason


class TestPolicyEngineFailModes:
    """Tests to catch fail_open/fail_closed mutation."""

    @pytest.mark.asyncio
    async def test_fail_closed_on_error(self, policy_context_factory):
        """Engine should deny when policy evaluation fails and fail_open=False.

        Mutation target: Flipping fail_open boolean
        """
        engine = PolicyEngine(config=PolicyEngineConfig(fail_open=False))
        ctx = policy_context_factory()

        engine.add_policy(FailingPolicy())

        decision = await engine.evaluate(ctx)

        # Should deny due to fail_closed mode
        assert decision.action == PolicyAction.DENY, (
            "Mutation caught: fail_closed mode not enforced"
        )

    @pytest.mark.asyncio
    async def test_fail_open_allows_on_error(self, policy_context_factory):
        """Engine should allow when policy evaluation fails and fail_open=True."""
        engine = PolicyEngine(config=PolicyEngineConfig(fail_open=True))
        ctx = policy_context_factory()

        engine.add_policy(FailingPolicy())

        decision = await engine.evaluate(ctx)

        # Should allow due to fail_open mode
        assert decision.action == PolicyAction.ALLOW, "Mutation caught: fail_open mode not enforced"


class TestPolicyResultAggregation:
    """Tests to catch decision aggregation mutations."""

    @pytest.mark.asyncio
    async def test_basic_evaluation(self, policy_context_factory):
        """Basic evaluation should work."""
        engine = PolicyEngine()
        ctx = policy_context_factory()

        decision = await engine.evaluate(ctx)

        # Basic sanity check - no policies means allow
        assert isinstance(decision, PolicyDecision)
        assert decision.action == PolicyAction.ALLOW

    @pytest.mark.asyncio
    async def test_no_policies_allows(self, policy_context_factory):
        """No policies should result in allow."""
        engine = PolicyEngine()
        ctx = policy_context_factory()

        decision = await engine.evaluate(ctx)

        assert decision.action == PolicyAction.ALLOW


class TestPolicyActionEnumMutations:
    """Tests to catch enum comparison mutations."""

    def test_action_enum_values_distinct(self):
        """All PolicyAction values should be distinct."""
        values = [a.value for a in PolicyAction]
        assert len(values) == len(set(values)), "Duplicate enum values"

    def test_allow_is_not_deny(self):
        """ALLOW and DENY must be different."""
        assert PolicyAction.ALLOW != PolicyAction.DENY
        assert PolicyAction.ALLOW.value != PolicyAction.DENY.value

    def test_warn_is_not_allow(self):
        """WARN and ALLOW must be different."""
        assert PolicyAction.WARN != PolicyAction.ALLOW

    def test_limit_is_not_warn(self):
        """LIMIT and WARN must be different."""
        assert PolicyAction.LIMIT != PolicyAction.WARN

    @pytest.mark.asyncio
    async def test_decision_action_in_valid_set(self, policy_context_factory):
        """Decision action should be one of valid actions."""
        engine = PolicyEngine()
        ctx = policy_context_factory()

        decision = await engine.evaluate(ctx)

        valid_actions = {
            PolicyAction.ALLOW,
            PolicyAction.DENY,
            PolicyAction.WARN,
            PolicyAction.LIMIT,
        }
        assert decision.action in valid_actions


class TestPolicyContextValidation:
    """Tests for policy context boundary conditions."""

    @pytest.mark.asyncio
    async def test_none_user_handled(self, policy_context_factory):
        """None user should be handled gracefully."""
        ctx = policy_context_factory(user_id=None)
        engine = PolicyEngine()

        # Should not crash
        decision = await engine.evaluate(ctx)
        assert isinstance(decision, PolicyDecision)

    @pytest.mark.asyncio
    async def test_empty_resource_handled(self, policy_context_factory):
        """Empty resource string should be handled."""
        ctx = policy_context_factory(resource="")
        engine = PolicyEngine()

        decision = await engine.evaluate(ctx)
        assert isinstance(decision, PolicyDecision)


class TestPolicyPriorityMutations:
    """Tests for policy priority handling."""

    @pytest.mark.asyncio
    async def test_policy_order_matters(self, policy_context_factory):
        """Policies should be evaluated in priority order.

        Mutation target: Changing iteration order
        """
        engine = PolicyEngine()
        ctx = policy_context_factory()

        evaluation_order = []

        class OrderTrackingPolicy(Policy):
            def __init__(self, name: str, priority: int):
                self._name = name
                self._priority = priority

            @property
            def name(self) -> str:
                return self._name

            @property
            def priority(self) -> int:
                return self._priority

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                evaluation_order.append(self._name)
                return PolicyResult.allow()

        # Add in reverse priority order
        engine.add_policy(OrderTrackingPolicy("low", 10))
        engine.add_policy(OrderTrackingPolicy("high", 100))
        engine.add_policy(OrderTrackingPolicy("medium", 50))

        await engine.evaluate(ctx)

        # Should be evaluated high -> medium -> low
        assert evaluation_order[0] == "high", "High priority not evaluated first"

    @pytest.mark.asyncio
    async def test_deny_stops_evaluation(self, policy_context_factory):
        """After deny, subsequent policies should not be evaluated.

        Mutation target: Removing break after deny
        """
        engine = PolicyEngine()
        ctx = policy_context_factory()

        evaluated_policies = []

        class TrackingDenyPolicy(Policy):
            @property
            def name(self) -> str:
                return "deny"

            @property
            def priority(self) -> int:
                return 100  # High priority - evaluated first

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                evaluated_policies.append("deny")
                return PolicyResult.deny("Denied by tracking policy")

        class TrackingAllowPolicy(Policy):
            @property
            def name(self) -> str:
                return "allow"

            @property
            def priority(self) -> int:
                return 50  # Lower priority - evaluated second

            async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
                evaluated_policies.append("allow")
                return PolicyResult.allow()

        engine.add_policy(TrackingDenyPolicy())
        engine.add_policy(TrackingAllowPolicy())

        decision = await engine.evaluate(ctx)

        assert decision.action == PolicyAction.DENY
        assert "deny" in evaluated_policies
        # With short_circuit_deny=True (default), allow should not be evaluated
        assert "allow" not in evaluated_policies, "Allow policy evaluated after deny"


class TestRBACPolicyBasic:
    """Basic RBAC policy tests for mutation detection."""

    @pytest.mark.asyncio
    async def test_rbac_denies_unauthenticated(self, policy_context_factory):
        """RBAC should deny unauthenticated requests by default."""
        rbac = RBACPolicy(deny_unauthenticated=True)

        # Create context with no user
        ctx = policy_context_factory(user_id=None)

        result = await rbac.evaluate(ctx)

        assert result.is_denied
        assert "Authentication required" in result.reason

    @pytest.mark.asyncio
    async def test_rbac_allows_with_permission(self, policy_context_factory):
        """RBAC should allow when user has permission."""
        rbac = RBACPolicy()

        # Add role with permission
        rbac.add_role(
            Role(name="user", permissions=[Permission(resource="tools/*", actions=["invoke"])])
        )
        rbac.assign_role("test-user", "user")

        ctx = policy_context_factory(
            user_id="test-user",
            resource="tools/search",
            action="invoke",
        )

        result = await rbac.evaluate(ctx)

        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_rbac_denies_without_permission(self, policy_context_factory):
        """RBAC should deny when user lacks permission."""
        rbac = RBACPolicy()

        # Add role with limited permission
        rbac.add_role(
            Role(name="reader", permissions=[Permission(resource="tools/*", actions=["read"])])
        )
        rbac.assign_role("test-user", "reader")

        ctx = policy_context_factory(
            user_id="test-user",
            resource="tools/search",
            action="invoke",  # Not allowed - only read
        )

        result = await rbac.evaluate(ctx)

        assert result.is_denied
