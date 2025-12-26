"""Budget and quota enforcement for FastAgentic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from fastagentic.policy.base import Policy, PolicyContext, PolicyResult


class BudgetPeriod(str, Enum):
    """Time period for budget tracking."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    def seconds(self) -> int:
        """Get the period duration in seconds."""
        return {
            BudgetPeriod.MINUTE: 60,
            BudgetPeriod.HOUR: 3600,
            BudgetPeriod.DAY: 86400,
            BudgetPeriod.WEEK: 604800,
            BudgetPeriod.MONTH: 2592000,  # 30 days
        }[self]


@dataclass
class Budget:
    """Budget configuration for cost/quota limits.

    Attributes:
        max_cost: Maximum cost in dollars for the period
        max_tokens: Maximum tokens for the period
        max_requests: Maximum requests for the period
        period: Time period for the budget
        soft_limit_percent: Percentage for soft limit warnings (0-100)
    """

    max_cost: float | None = None
    max_tokens: int | None = None
    max_requests: int | None = None
    period: BudgetPeriod = BudgetPeriod.DAY
    soft_limit_percent: int = 80

    def __post_init__(self) -> None:
        if not any([self.max_cost, self.max_tokens, self.max_requests]):
            raise ValueError("At least one budget limit must be set")
        if not 0 <= self.soft_limit_percent <= 100:
            raise ValueError("soft_limit_percent must be between 0 and 100")


@dataclass
class UsageRecord:
    """Track usage for a period.

    Attributes:
        cost: Total cost in dollars
        tokens: Total tokens used
        requests: Total requests made
        period_start: Unix timestamp when period started
    """

    cost: float = 0.0
    tokens: int = 0
    requests: int = 0
    period_start: float = field(default_factory=time.time)

    def reset(self) -> None:
        """Reset usage counters for a new period."""
        self.cost = 0.0
        self.tokens = 0
        self.requests = 0
        self.period_start = time.time()


class UsageStore(Protocol):
    """Protocol for usage data storage."""

    async def get_usage(self, key: str) -> UsageRecord | None:
        """Get usage record for a key."""
        ...

    async def set_usage(self, key: str, record: UsageRecord) -> None:
        """Set usage record for a key."""
        ...

    async def increment_usage(
        self,
        key: str,
        cost: float = 0.0,
        tokens: int = 0,
        requests: int = 0,
    ) -> UsageRecord:
        """Increment usage counters and return updated record."""
        ...


class InMemoryUsageStore:
    """In-memory usage store for development/testing."""

    def __init__(self) -> None:
        self._usage: dict[str, UsageRecord] = {}

    async def get_usage(self, key: str) -> UsageRecord | None:
        """Get usage record for a key."""
        return self._usage.get(key)

    async def set_usage(self, key: str, record: UsageRecord) -> None:
        """Set usage record for a key."""
        self._usage[key] = record

    async def increment_usage(
        self,
        key: str,
        cost: float = 0.0,
        tokens: int = 0,
        requests: int = 0,
    ) -> UsageRecord:
        """Increment usage counters and return updated record."""
        if key not in self._usage:
            self._usage[key] = UsageRecord()

        record = self._usage[key]
        record.cost += cost
        record.tokens += tokens
        record.requests += requests
        return record


class BudgetPolicy(Policy):
    """Budget enforcement policy.

    Tracks usage and enforces cost, token, and request limits.

    Example:
        policy = BudgetPolicy(
            store=InMemoryUsageStore(),
        )

        # Set global budget
        policy.set_global_budget(Budget(
            max_cost=100.0,
            max_tokens=1_000_000,
            period=BudgetPeriod.DAY,
        ))

        # Set per-user budget
        policy.set_user_budget("user-123", Budget(
            max_cost=10.0,
            max_requests=1000,
            period=BudgetPeriod.DAY,
        ))

        # Set tenant budget
        policy.set_tenant_budget("tenant-abc", Budget(
            max_cost=500.0,
            period=BudgetPeriod.MONTH,
        ))
    """

    def __init__(
        self,
        *,
        store: UsageStore | None = None,
        enforce_on_estimate: bool = True,
    ) -> None:
        """Initialize budget policy.

        Args:
            store: Usage data store (defaults to in-memory)
            enforce_on_estimate: Whether to enforce based on estimated cost
        """
        self._store = store or InMemoryUsageStore()
        self._enforce_on_estimate = enforce_on_estimate
        self._global_budget: Budget | None = None
        self._user_budgets: dict[str, Budget] = {}
        self._tenant_budgets: dict[str, Budget] = {}
        self._endpoint_budgets: dict[str, Budget] = {}

    @property
    def name(self) -> str:
        return "budget"

    @property
    def priority(self) -> int:
        return 50  # After auth (100) and scopes (90)

    def set_global_budget(self, budget: Budget) -> None:
        """Set the global budget limit.

        Args:
            budget: Budget configuration
        """
        self._global_budget = budget

    def set_user_budget(self, user_id: str, budget: Budget) -> None:
        """Set budget for a specific user.

        Args:
            user_id: User identifier
            budget: Budget configuration
        """
        self._user_budgets[user_id] = budget

    def set_tenant_budget(self, tenant_id: str, budget: Budget) -> None:
        """Set budget for a tenant/organization.

        Args:
            tenant_id: Tenant identifier
            budget: Budget configuration
        """
        self._tenant_budgets[tenant_id] = budget

    def set_endpoint_budget(self, endpoint: str, budget: Budget) -> None:
        """Set budget for a specific endpoint.

        Args:
            endpoint: Endpoint path
            budget: Budget configuration
        """
        self._endpoint_budgets[endpoint] = budget

    def remove_user_budget(self, user_id: str) -> None:
        """Remove budget for a user."""
        self._user_budgets.pop(user_id, None)

    def remove_tenant_budget(self, tenant_id: str) -> None:
        """Remove budget for a tenant."""
        self._tenant_budgets.pop(tenant_id, None)

    def remove_endpoint_budget(self, endpoint: str) -> None:
        """Remove budget for an endpoint."""
        self._endpoint_budgets.pop(endpoint, None)

    async def get_usage(
        self,
        key: str,
        budget: Budget,
    ) -> UsageRecord:
        """Get current usage for a key, resetting if period expired.

        Args:
            key: Usage tracking key
            budget: Budget to check period against

        Returns:
            Current usage record
        """
        record = await self._store.get_usage(key)

        if record is None:
            record = UsageRecord()
            await self._store.set_usage(key, record)
            return record

        # Check if period has expired
        period_seconds = budget.period.seconds()
        elapsed = time.time() - record.period_start

        if elapsed >= period_seconds:
            record.reset()
            await self._store.set_usage(key, record)

        return record

    async def record_usage(
        self,
        ctx: PolicyContext,
        actual_cost: float = 0.0,
        actual_tokens: int = 0,
    ) -> None:
        """Record actual usage after a request completes.

        Args:
            ctx: Policy context
            actual_cost: Actual cost incurred
            actual_tokens: Actual tokens used
        """
        # Record against all applicable budgets
        if self._global_budget:
            await self._store.increment_usage(
                "global",
                cost=actual_cost,
                tokens=actual_tokens,
                requests=1,
            )

        if ctx.user_id and ctx.user_id in self._user_budgets:
            await self._store.increment_usage(
                f"user:{ctx.user_id}",
                cost=actual_cost,
                tokens=actual_tokens,
                requests=1,
            )

        if ctx.tenant_id and ctx.tenant_id in self._tenant_budgets:
            await self._store.increment_usage(
                f"tenant:{ctx.tenant_id}",
                cost=actual_cost,
                tokens=actual_tokens,
                requests=1,
            )

        if ctx.endpoint and ctx.endpoint in self._endpoint_budgets:
            await self._store.increment_usage(
                f"endpoint:{ctx.endpoint}",
                cost=actual_cost,
                tokens=actual_tokens,
                requests=1,
            )

    def _check_budget(
        self,
        record: UsageRecord,
        budget: Budget,
        estimated_cost: float = 0.0,
        estimated_tokens: int = 0,
    ) -> tuple[bool, bool, str | None]:
        """Check if usage is within budget.

        Args:
            record: Current usage record
            budget: Budget to check against
            estimated_cost: Estimated cost for this request
            estimated_tokens: Estimated tokens for this request

        Returns:
            Tuple of (is_allowed, is_warning, reason)
        """
        # Check cost limit
        if budget.max_cost is not None:
            total_cost = record.cost
            if self._enforce_on_estimate:
                total_cost += estimated_cost

            if total_cost >= budget.max_cost:
                return (
                    False,
                    False,
                    f"Cost budget exceeded: ${total_cost:.4f} >= ${budget.max_cost:.2f}",
                )

            soft_limit = budget.max_cost * (budget.soft_limit_percent / 100)
            if total_cost >= soft_limit:
                return (
                    True,
                    True,
                    f"Approaching cost limit: ${total_cost:.4f} / ${budget.max_cost:.2f}",
                )

        # Check token limit
        if budget.max_tokens is not None:
            total_tokens = record.tokens
            if self._enforce_on_estimate:
                total_tokens += estimated_tokens

            if total_tokens >= budget.max_tokens:
                return (
                    False,
                    False,
                    f"Token budget exceeded: {total_tokens:,} >= {budget.max_tokens:,}",
                )

            soft_limit = int(budget.max_tokens * (budget.soft_limit_percent / 100))
            if total_tokens >= soft_limit:
                return (
                    True,
                    True,
                    f"Approaching token limit: {total_tokens:,} / {budget.max_tokens:,}",
                )

        # Check request limit
        if budget.max_requests is not None:
            total_requests = record.requests + 1  # Include current request

            if total_requests > budget.max_requests:
                return (
                    False,
                    False,
                    f"Request limit exceeded: {total_requests:,} > {budget.max_requests:,}",
                )

            soft_limit = int(budget.max_requests * (budget.soft_limit_percent / 100))
            if total_requests >= soft_limit:
                return (
                    True,
                    True,
                    f"Approaching request limit: {total_requests:,} / {budget.max_requests:,}",
                )

        return (True, False, None)

    async def evaluate(self, ctx: PolicyContext) -> PolicyResult:
        """Evaluate budget policy.

        Args:
            ctx: Policy evaluation context

        Returns:
            PolicyResult with allow/deny/warn decision
        """
        warnings: list[str] = []

        # Check global budget
        if self._global_budget:
            record = await self.get_usage("global", self._global_budget)
            allowed, warning, reason = self._check_budget(
                record,
                self._global_budget,
                ctx.estimated_cost,
                ctx.estimated_tokens,
            )
            if not allowed:
                return PolicyResult.deny(f"Global budget: {reason}")
            if warning and reason:
                warnings.append(f"Global: {reason}")

        # Check user budget
        if ctx.user_id and ctx.user_id in self._user_budgets:
            budget = self._user_budgets[ctx.user_id]
            record = await self.get_usage(f"user:{ctx.user_id}", budget)
            allowed, warning, reason = self._check_budget(
                record,
                budget,
                ctx.estimated_cost,
                ctx.estimated_tokens,
            )
            if not allowed:
                return PolicyResult.deny(f"User budget: {reason}")
            if warning and reason:
                warnings.append(f"User: {reason}")

        # Check tenant budget
        if ctx.tenant_id and ctx.tenant_id in self._tenant_budgets:
            budget = self._tenant_budgets[ctx.tenant_id]
            record = await self.get_usage(f"tenant:{ctx.tenant_id}", budget)
            allowed, warning, reason = self._check_budget(
                record,
                budget,
                ctx.estimated_cost,
                ctx.estimated_tokens,
            )
            if not allowed:
                return PolicyResult.deny(f"Tenant budget: {reason}")
            if warning and reason:
                warnings.append(f"Tenant: {reason}")

        # Check endpoint budget
        if ctx.endpoint and ctx.endpoint in self._endpoint_budgets:
            budget = self._endpoint_budgets[ctx.endpoint]
            record = await self.get_usage(f"endpoint:{ctx.endpoint}", budget)
            allowed, warning, reason = self._check_budget(
                record,
                budget,
                ctx.estimated_cost,
                ctx.estimated_tokens,
            )
            if not allowed:
                return PolicyResult.deny(f"Endpoint budget: {reason}")
            if warning and reason:
                warnings.append(f"Endpoint: {reason}")

        # Return with warnings if any
        if warnings:
            return PolicyResult.warn("; ".join(warnings))

        return PolicyResult.allow()
