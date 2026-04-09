"""Cost tracking implementation for FastAgentic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol


@dataclass
class ModelPricing:
    """Pricing configuration for an LLM model.

    Attributes:
        model: Model name/identifier
        input_cost_per_1k: Cost per 1K input tokens in dollars
        output_cost_per_1k: Cost per 1K output tokens in dollars
        cached_input_cost_per_1k: Cost per 1K cached input tokens (if applicable)
    """

    model: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    cached_input_cost_per_1k: float | None = None

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens

        Returns:
            Total cost in dollars
        """
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k

        cached_cost = 0.0
        if cached_tokens > 0 and self.cached_input_cost_per_1k is not None:
            cached_cost = (cached_tokens / 1000) * self.cached_input_cost_per_1k

        return input_cost + output_cost + cached_cost


# Default pricing for common models (as of late 2024/early 2025)
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing("gpt-4o", 0.0025, 0.010),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.00015, 0.0006),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", 0.01, 0.03),
    "gpt-4": ModelPricing("gpt-4", 0.03, 0.06),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015),
    "o1": ModelPricing("o1", 0.015, 0.06),
    "o1-mini": ModelPricing("o1-mini", 0.003, 0.012),
    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing("claude-3-5-sonnet-20241022", 0.003, 0.015, 0.0003),
    "claude-3-5-haiku-20241022": ModelPricing("claude-3-5-haiku-20241022", 0.0008, 0.004, 0.00008),
    "claude-3-opus-20240229": ModelPricing("claude-3-opus-20240229", 0.015, 0.075, 0.0015),
    "claude-3-sonnet-20240229": ModelPricing("claude-3-sonnet-20240229", 0.003, 0.015, 0.0003),
    "claude-3-haiku-20240307": ModelPricing("claude-3-haiku-20240307", 0.00025, 0.00125, 0.000025),
    # Google
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", 0.00125, 0.005),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", 0.000075, 0.0003),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", 0.0001, 0.0004),
    # Aliases
    "claude-3.5-sonnet": ModelPricing("claude-3.5-sonnet", 0.003, 0.015, 0.0003),
    "claude-3.5-haiku": ModelPricing("claude-3.5-haiku", 0.0008, 0.004, 0.00008),
}


@dataclass
class CostRecord:
    """Record of a single LLM call cost.

    Attributes:
        run_id: Unique run identifier
        model: Model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached tokens
        cost: Total cost in dollars
        timestamp: When the call was made
        user_id: User who made the request
        tenant_id: Tenant/organization
        endpoint: Endpoint path
        metadata: Additional context
    """

    run_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    cached_tokens: int = 0
    user_id: str | None = None
    tenant_id: str | None = None
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp)


class AggregationPeriod(str, Enum):
    """Time period for cost aggregation."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class CostAggregation:
    """Aggregated cost data.

    Attributes:
        period: Aggregation period
        start_time: Start of the period
        end_time: End of the period
        total_cost: Total cost in dollars
        total_input_tokens: Total input tokens
        total_output_tokens: Total output tokens
        total_requests: Total number of requests
        by_model: Breakdown by model
        by_user: Breakdown by user
        by_endpoint: Breakdown by endpoint
    """

    period: AggregationPeriod
    start_time: datetime
    end_time: datetime
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    by_user: dict[str, float] = field(default_factory=dict)
    by_endpoint: dict[str, float] = field(default_factory=dict)


class CostStore(Protocol):
    """Protocol for cost data storage."""

    async def record(self, record: CostRecord) -> None:
        """Store a cost record."""
        ...

    async def get_records(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        limit: int = 1000,
    ) -> list[CostRecord]:
        """Query cost records."""
        ...


class InMemoryCostStore:
    """In-memory cost store for development/testing."""

    def __init__(self, max_records: int = 10000) -> None:
        self._records: list[CostRecord] = []
        self._max_records = max_records

    async def record(self, record: CostRecord) -> None:
        """Store a cost record."""
        self._records.append(record)
        # Trim old records if over limit
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    async def get_records(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        limit: int = 1000,
    ) -> list[CostRecord]:
        """Query cost records."""
        results = []
        for record in reversed(self._records):
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            if user_id and record.user_id != user_id:
                continue
            if tenant_id and record.tenant_id != tenant_id:
                continue
            if endpoint and record.endpoint != endpoint:
                continue
            if model and record.model != model:
                continue

            results.append(record)
            if len(results) >= limit:
                break

        return results


class CostTracker:
    """Track and analyze LLM costs.

    Example:
        tracker = CostTracker()

        # Add custom pricing
        tracker.add_pricing(ModelPricing(
            model="my-custom-model",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        ))

        # Record usage
        await tracker.record(
            run_id="run-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            user_id="user-456",
        )

        # Get aggregations
        daily = await tracker.aggregate(AggregationPeriod.DAY)
        print(f"Today's cost: ${daily.total_cost:.4f}")
    """

    def __init__(
        self,
        store: CostStore | None = None,
        pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        """Initialize cost tracker.

        Args:
            store: Storage backend for cost records
            pricing: Custom pricing (merged with defaults)
        """
        self._store = store or InMemoryCostStore()
        self._pricing = {**DEFAULT_PRICING}
        if pricing:
            self._pricing.update({p.model: p for p in pricing.values()})

    def add_pricing(self, pricing: ModelPricing) -> None:
        """Add or update model pricing.

        Args:
            pricing: Model pricing configuration
        """
        self._pricing[pricing.model] = pricing

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a model.

        Args:
            model: Model name

        Returns:
            ModelPricing or None if not found
        """
        # Try exact match first
        if model in self._pricing:
            return self._pricing[model]

        # Try prefix match for versioned models
        for name, pricing in self._pricing.items():
            if model.startswith(name) or name.startswith(model):
                return pricing

        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens

        Returns:
            Total cost in dollars (0.0 if model not found)
        """
        pricing = self.get_pricing(model)
        if pricing is None:
            return 0.0
        return pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)

    async def record(
        self,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record an LLM call.

        Args:
            run_id: Unique run identifier
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens
            user_id: User who made the request
            tenant_id: Tenant/organization
            endpoint: Endpoint path
            metadata: Additional context

        Returns:
            The created CostRecord
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens, cached_tokens)

        record = CostRecord(
            run_id=run_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            metadata=metadata or {},
        )

        await self._store.record(record)
        return record

    async def get_records(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        limit: int = 1000,
    ) -> list[CostRecord]:
        """Query cost records.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            user_id: Filter by user
            tenant_id: Filter by tenant
            endpoint: Filter by endpoint
            model: Filter by model
            limit: Maximum records to return

        Returns:
            List of matching CostRecords
        """
        return await self._store.get_records(
            start_time=start_time.timestamp() if start_time else None,
            end_time=end_time.timestamp() if end_time else None,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            model=model,
            limit=limit,
        )

    async def aggregate(
        self,
        period: AggregationPeriod,
        start_time: datetime | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> CostAggregation:
        """Aggregate cost data for a time period.

        Args:
            period: Aggregation period
            start_time: Start of period (defaults to current period start)
            user_id: Filter by user
            tenant_id: Filter by tenant

        Returns:
            CostAggregation with totals and breakdowns
        """
        now = datetime.now()

        # Calculate period boundaries
        if start_time is None:
            if period == AggregationPeriod.HOUR:
                start_time = now.replace(minute=0, second=0, microsecond=0)
            elif period == AggregationPeriod.DAY:
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == AggregationPeriod.WEEK:
                start_time = now - timedelta(days=now.weekday())
                start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == AggregationPeriod.MONTH:
                start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Calculate end time
        if period == AggregationPeriod.HOUR:
            end_time = start_time + timedelta(hours=1)
        elif period == AggregationPeriod.DAY:
            end_time = start_time + timedelta(days=1)
        elif period == AggregationPeriod.WEEK:
            end_time = start_time + timedelta(weeks=1)
        elif period == AggregationPeriod.MONTH:
            # Approximate - use 30 days
            end_time = start_time + timedelta(days=30)

        # Get records
        records = await self.get_records(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=100000,
        )

        # Aggregate
        aggregation = CostAggregation(
            period=period,
            start_time=start_time,
            end_time=end_time,
        )

        for record in records:
            aggregation.total_cost += record.cost
            aggregation.total_input_tokens += record.input_tokens
            aggregation.total_output_tokens += record.output_tokens
            aggregation.total_requests += 1

            # By model
            if record.model not in aggregation.by_model:
                aggregation.by_model[record.model] = 0.0
            aggregation.by_model[record.model] += record.cost

            # By user
            if record.user_id:
                if record.user_id not in aggregation.by_user:
                    aggregation.by_user[record.user_id] = 0.0
                aggregation.by_user[record.user_id] += record.cost

            # By endpoint
            if record.endpoint:
                if record.endpoint not in aggregation.by_endpoint:
                    aggregation.by_endpoint[record.endpoint] = 0.0
                aggregation.by_endpoint[record.endpoint] += record.cost

        return aggregation

    async def get_total_cost(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> float:
        """Get total cost for a time range.

        Args:
            start_time: Start of range
            end_time: End of range
            user_id: Filter by user
            tenant_id: Filter by tenant

        Returns:
            Total cost in dollars
        """
        records = await self.get_records(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=100000,
        )
        return sum(r.cost for r in records)
