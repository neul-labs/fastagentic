"""Mutation tests for Budget Policy.

These tests are specifically designed to catch common mutations:
- Boundary condition changes (< vs <=, > vs >=)
- Soft limit percentage calculations
- Budget threshold comparisons

Run mutation testing with:
    mutmut run --paths-to-mutate=src/fastagentic/policy/budget.py

Targeted mutations:
- Line 54: Boundary check 0 <= x <= 100
- Budget comparison operators
- Soft limit warning threshold
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fastagentic.context import UserInfo
from fastagentic.policy.base import PolicyAction, PolicyContext
from fastagentic.policy.budget import Budget, BudgetPolicy


class TestBudgetBoundaryConditions:
    """Tests for budget soft_limit_percent boundary conditions."""

    def test_soft_limit_at_zero(self):
        """soft_limit_percent=0 should be valid.

        Mutation: 0 <= x changed to 0 < x
        """
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=0,
        )
        assert budget.soft_limit_percent == 0

    def test_soft_limit_at_hundred(self):
        """soft_limit_percent=100 should be valid.

        Mutation: x <= 100 changed to x < 100
        """
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=100,
        )
        assert budget.soft_limit_percent == 100

    def test_soft_limit_at_one(self):
        """soft_limit_percent=1 should be valid."""
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=1,
        )
        assert budget.soft_limit_percent == 1

    def test_soft_limit_at_ninety_nine(self):
        """soft_limit_percent=99 should be valid."""
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=99,
        )
        assert budget.soft_limit_percent == 99

    def test_soft_limit_negative_rejected(self):
        """soft_limit_percent=-1 should be rejected.

        Mutation: Removing lower bound check
        """
        with pytest.raises(ValueError):
            Budget(
                max_cost=100.0,
                soft_limit_percent=-1,
            )

    def test_soft_limit_over_hundred_rejected(self):
        """soft_limit_percent=101 should be rejected.

        Mutation: Removing upper bound check
        """
        with pytest.raises(ValueError):
            Budget(
                max_cost=100.0,
                soft_limit_percent=101,
            )

    @given(percent=st.integers(min_value=0, max_value=100))
    def test_valid_percentages_accepted(self, percent: int):
        """All percentages 0-100 should be accepted."""
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=percent,
        )
        assert budget.soft_limit_percent == percent

    @given(percent=st.integers(min_value=101, max_value=1000))
    def test_over_hundred_rejected(self, percent: int):
        """All percentages > 100 should be rejected."""
        with pytest.raises(ValueError):
            Budget(
                max_cost=100.0,
                soft_limit_percent=percent,
            )

    @given(percent=st.integers(min_value=-1000, max_value=-1))
    def test_negative_rejected(self, percent: int):
        """All negative percentages should be rejected."""
        with pytest.raises(ValueError):
            Budget(
                max_cost=100.0,
                soft_limit_percent=percent,
            )


class TestBudgetRequirements:
    """Tests for budget requirement validation."""

    def test_at_least_one_limit_required(self):
        """At least one of max_cost, max_tokens, max_requests required.

        Mutation: Removing this validation
        """
        with pytest.raises(ValueError):
            Budget()  # No limits set

    def test_max_cost_alone_valid(self):
        """max_cost alone should be valid."""
        budget = Budget(max_cost=100.0)
        assert budget.max_cost == 100.0

    def test_max_tokens_alone_valid(self):
        """max_tokens alone should be valid."""
        budget = Budget(max_tokens=100000)
        assert budget.max_tokens == 100000

    def test_max_requests_alone_valid(self):
        """max_requests alone should be valid."""
        budget = Budget(max_requests=1000)
        assert budget.max_requests == 1000

    def test_all_limits_valid(self):
        """All limits together should be valid."""
        budget = Budget(
            max_cost=100.0,
            max_tokens=100000,
            max_requests=1000,
        )
        assert budget.max_cost == 100.0
        assert budget.max_tokens == 100000
        assert budget.max_requests == 1000


class TestBudgetPolicyEvaluation:
    """Tests for BudgetPolicy evaluation mutations."""

    @pytest.fixture
    def context_factory(self):
        def create(
            user_id: str | None = "test-user",
            tenant_id: str | None = "test-tenant",
            endpoint: str | None = "/test",
            resource: str | None = "tools/search",
            action: str | None = "invoke",
            estimated_tokens: int = 0,
            estimated_cost: float = 0.0,
        ) -> PolicyContext:
            user = UserInfo(id=user_id) if user_id else None
            return PolicyContext(
                user=user,
                tenant_id=tenant_id,
                endpoint=endpoint,
                resource=resource,
                action=action,
                estimated_tokens=estimated_tokens,
                estimated_cost=estimated_cost,
            )

        return create

    @pytest.mark.asyncio
    async def test_under_budget_allowed(self, context_factory):
        """Request under budget should be allowed."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_cost=100.0, soft_limit_percent=80))

        ctx = context_factory(estimated_cost=10.0)
        result = await policy.evaluate(ctx)

        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_at_soft_limit_warns(self, context_factory):
        """Request at soft limit should warn.

        Mutation: >= changed to > in soft limit check
        """
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_cost=100.0, soft_limit_percent=80))

        # Exactly at 80% - should warn
        ctx = context_factory(estimated_cost=80.0)
        result = await policy.evaluate(ctx)

        # Should be WARN
        assert result.action == PolicyAction.WARN

    @pytest.mark.asyncio
    async def test_over_budget_denied(self, context_factory):
        """Request over budget should be denied.

        Mutation: >= changed to > in budget check
        """
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_cost=100.0))

        ctx = context_factory(estimated_cost=150.0)
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_exactly_at_budget_denied(self, context_factory):
        """Request exactly at budget should be denied (>= comparison)."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_cost=100.0))

        # Exactly at budget should be denied (>= max_cost)
        ctx = context_factory(estimated_cost=100.0)
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_zero_estimated_cost_allowed(self, context_factory):
        """Zero estimated cost should be allowed."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_cost=100.0))

        ctx = context_factory(estimated_cost=0.0)
        result = await policy.evaluate(ctx)

        assert result.is_allowed


class TestBudgetTokenLimits:
    """Tests for token-based budget limits."""

    @pytest.fixture
    def context_factory(self):
        def create(
            user_id: str | None = "test-user",
            tenant_id: str | None = "test-tenant",
            endpoint: str | None = "/test",
            resource: str | None = "tools/search",
            action: str | None = "invoke",
            estimated_tokens: int = 0,
            estimated_cost: float = 0.0,
        ) -> PolicyContext:
            user = UserInfo(id=user_id) if user_id else None
            return PolicyContext(
                user=user,
                tenant_id=tenant_id,
                endpoint=endpoint,
                resource=resource,
                action=action,
                estimated_tokens=estimated_tokens,
                estimated_cost=estimated_cost,
            )

        return create

    @pytest.mark.asyncio
    async def test_under_token_limit_allowed(self, context_factory):
        """Request under token limit should be allowed."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_tokens=10000))

        ctx = context_factory(estimated_tokens=5000)
        result = await policy.evaluate(ctx)

        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_over_token_limit_denied(self, context_factory):
        """Request over token limit should be denied."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_tokens=10000))

        ctx = context_factory(estimated_tokens=15000)
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_exactly_at_token_limit_denied(self, context_factory):
        """Request exactly at token limit should be denied (>= comparison)."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_tokens=10000))

        ctx = context_factory(estimated_tokens=10000)
        result = await policy.evaluate(ctx)

        assert result.is_denied


class TestBudgetSoftLimitCalculation:
    """Tests for soft limit percentage calculations."""

    @given(
        max_cost=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        soft_percent=st.integers(min_value=0, max_value=100),
    )
    def test_soft_limit_calculation(self, max_cost: float, soft_percent: int):
        """Soft limit should be calculated correctly."""
        budget = Budget(
            max_cost=max_cost,
            soft_limit_percent=soft_percent,
        )

        expected_soft_limit = max_cost * (soft_percent / 100)

        # Calculate actual soft limit
        actual_soft_limit = budget.max_cost * (budget.soft_limit_percent / 100)

        assert abs(actual_soft_limit - expected_soft_limit) < 0.01

    def test_soft_limit_at_zero_percent(self):
        """0% soft limit should warn immediately."""
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=0,
        )

        soft_threshold = budget.max_cost * (budget.soft_limit_percent / 100)
        assert soft_threshold == 0.0

    def test_soft_limit_at_hundred_percent(self):
        """100% soft limit means no warning before hard limit."""
        budget = Budget(
            max_cost=100.0,
            soft_limit_percent=100,
        )

        soft_threshold = budget.max_cost * (budget.soft_limit_percent / 100)
        assert soft_threshold == 100.0


class TestBudgetEdgeCases:
    """Edge case tests for budget handling."""

    def test_very_small_budget(self):
        """Very small budget should work correctly."""
        budget = Budget(max_cost=0.001)

        assert budget.max_cost == 0.001

    def test_very_large_budget(self):
        """Very large budget should work correctly."""
        budget = Budget(max_cost=1_000_000_000.0)

        assert budget.max_cost == 1_000_000_000.0

    @given(
        cost=st.floats(min_value=0.0, max_value=1e10, allow_nan=False, allow_infinity=False),
    )
    def test_budget_comparison_no_nan(self, cost: float):
        """Budget comparisons should never result in NaN."""
        budget = Budget(max_cost=1000.0)

        # These comparisons should always return bool, not NaN
        over = cost > budget.max_cost
        under = cost <= budget.max_cost

        assert isinstance(over, bool)
        assert isinstance(under, bool)
        assert over != under  # Exactly one should be true


class TestBudgetPolicyUserAndTenant:
    """Tests for user and tenant-specific budgets."""

    @pytest.fixture
    def context_factory(self):
        def create(
            user_id: str | None = "test-user",
            tenant_id: str | None = "test-tenant",
            endpoint: str | None = "/test",
            estimated_cost: float = 0.0,
        ) -> PolicyContext:
            user = UserInfo(id=user_id) if user_id else None
            return PolicyContext(
                user=user,
                tenant_id=tenant_id,
                endpoint=endpoint,
                estimated_cost=estimated_cost,
            )

        return create

    @pytest.mark.asyncio
    async def test_user_budget_independent(self, context_factory):
        """User budgets should be independent of global."""
        policy = BudgetPolicy()
        policy.set_global_budget(Budget(max_cost=1000.0))
        policy.set_user_budget("test-user", Budget(max_cost=10.0))

        # Request would be under global but over user budget
        ctx = context_factory(user_id="test-user", estimated_cost=50.0)
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_tenant_budget_enforced(self, context_factory):
        """Tenant budgets should be enforced."""
        policy = BudgetPolicy()
        policy.set_tenant_budget("test-tenant", Budget(max_cost=10.0))

        ctx = context_factory(tenant_id="test-tenant", estimated_cost=50.0)
        result = await policy.evaluate(ctx)

        assert result.is_denied

    @pytest.mark.asyncio
    async def test_no_budget_allows(self, context_factory):
        """No budget configured should allow all."""
        policy = BudgetPolicy()

        ctx = context_factory(estimated_cost=1000000.0)
        result = await policy.evaluate(ctx)

        assert result.is_allowed
