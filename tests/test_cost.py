"""Tests for FastAgentic cost tracking module."""

from datetime import datetime, timedelta

import pytest

from fastagentic.context import UsageInfo, UserInfo
from fastagentic.cost.hooks import CostTrackingHook
from fastagentic.cost.tracker import (
    DEFAULT_PRICING,
    AggregationPeriod,
    CostAggregation,
    CostRecord,
    CostTracker,
    InMemoryCostStore,
    ModelPricing,
)
from fastagentic.hooks.base import HookContext, HookResultAction


class TestModelPricing:
    """Tests for ModelPricing class."""

    def test_basic_pricing(self):
        pricing = ModelPricing(
            model="test-model",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        assert pricing.model == "test-model"
        assert pricing.input_cost_per_1k == 0.001
        assert pricing.output_cost_per_1k == 0.002

    def test_calculate_cost(self):
        pricing = ModelPricing(
            model="test-model",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        # (1000/1000 * 0.001) + (500/1000 * 0.002) = 0.001 + 0.001 = 0.002
        assert cost == pytest.approx(0.002)

    def test_calculate_cost_with_cache(self):
        pricing = ModelPricing(
            model="test-model",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            cached_input_cost_per_1k=0.0001,
        )
        cost = pricing.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
        )
        # (1000/1000 * 0.001) + (500/1000 * 0.002) + (500/1000 * 0.0001)
        # = 0.001 + 0.001 + 0.00005 = 0.00205
        assert cost == pytest.approx(0.00205)


class TestDefaultPricing:
    """Tests for default model pricing."""

    def test_gpt4o_pricing_exists(self):
        assert "gpt-4o" in DEFAULT_PRICING
        pricing = DEFAULT_PRICING["gpt-4o"]
        assert pricing.input_cost_per_1k > 0
        assert pricing.output_cost_per_1k > 0

    def test_claude_pricing_exists(self):
        assert "claude-3-5-sonnet-20241022" in DEFAULT_PRICING
        pricing = DEFAULT_PRICING["claude-3-5-sonnet-20241022"]
        assert pricing.cached_input_cost_per_1k is not None

    def test_gemini_pricing_exists(self):
        assert "gemini-1.5-pro" in DEFAULT_PRICING


class TestCostRecord:
    """Tests for CostRecord class."""

    def test_basic_record(self):
        record = CostRecord(
            run_id="run-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost=0.01,
        )
        assert record.run_id == "run-123"
        assert record.total_tokens == 1500
        assert isinstance(record.datetime, datetime)

    def test_record_with_metadata(self):
        record = CostRecord(
            run_id="run-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost=0.01,
            user_id="user-456",
            tenant_id="tenant-789",
            endpoint="/triage",
            metadata={"request_id": "req-abc"},
        )
        assert record.user_id == "user-456"
        assert record.metadata["request_id"] == "req-abc"


class TestInMemoryCostStore:
    """Tests for InMemoryCostStore class."""

    @pytest.mark.asyncio
    async def test_record_and_get(self):
        store = InMemoryCostStore()
        record = CostRecord(
            run_id="run-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost=0.01,
        )

        await store.record(record)
        records = await store.get_records()

        assert len(records) == 1
        assert records[0].run_id == "run-123"

    @pytest.mark.asyncio
    async def test_filter_by_user(self):
        store = InMemoryCostStore()
        await store.record(
            CostRecord(
                run_id="run-1",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
                user_id="user-A",
            )
        )
        await store.record(
            CostRecord(
                run_id="run-2",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
                user_id="user-B",
            )
        )

        records = await store.get_records(user_id="user-A")
        assert len(records) == 1
        assert records[0].user_id == "user-A"

    @pytest.mark.asyncio
    async def test_max_records_trimmed(self):
        store = InMemoryCostStore(max_records=5)

        for i in range(10):
            await store.record(
                CostRecord(
                    run_id=f"run-{i}",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.01,
                )
            )

        records = await store.get_records(limit=100)
        assert len(records) == 5


class TestCostTracker:
    """Tests for CostTracker class."""

    @pytest.mark.asyncio
    async def test_record_cost(self):
        tracker = CostTracker()

        record = await tracker.record(
            run_id="run-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            user_id="user-456",
        )

        assert record.run_id == "run-123"
        assert record.cost > 0

    @pytest.mark.asyncio
    async def test_calculate_cost(self):
        tracker = CostTracker()

        cost = tracker.calculate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0

    @pytest.mark.asyncio
    async def test_unknown_model_returns_zero(self):
        tracker = CostTracker()

        cost = tracker.calculate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_custom_pricing(self):
        custom_pricing = ModelPricing(
            model="custom-model",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
        )
        tracker = CostTracker()
        tracker.add_pricing(custom_pricing)

        cost = tracker.calculate_cost(
            model="custom-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # (1000/1000 * 0.01) + (500/1000 * 0.02) = 0.01 + 0.01 = 0.02
        assert cost == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_get_records(self):
        tracker = CostTracker()

        await tracker.record(
            run_id="run-1",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            user_id="user-123",
        )
        await tracker.record(
            run_id="run-2",
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            user_id="user-456",
        )

        records = await tracker.get_records()
        assert len(records) == 2

        user_records = await tracker.get_records(user_id="user-123")
        assert len(user_records) == 1

    @pytest.mark.asyncio
    async def test_get_total_cost(self):
        tracker = CostTracker()

        await tracker.record(
            run_id="run-1",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        await tracker.record(
            run_id="run-2",
            model="gpt-4o",
            input_tokens=2000,
            output_tokens=1000,
        )

        total = await tracker.get_total_cost()
        assert total > 0

    @pytest.mark.asyncio
    async def test_aggregate_by_day(self):
        tracker = CostTracker()

        await tracker.record(
            run_id="run-1",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            user_id="user-123",
            endpoint="/triage",
        )
        await tracker.record(
            run_id="run-2",
            model="claude-3-5-sonnet-20241022",
            input_tokens=2000,
            output_tokens=1000,
            user_id="user-456",
            endpoint="/chat",
        )

        aggregation = await tracker.aggregate(AggregationPeriod.DAY)

        assert aggregation.total_requests == 2
        assert aggregation.total_cost > 0
        assert len(aggregation.by_model) == 2
        assert len(aggregation.by_user) == 2
        assert len(aggregation.by_endpoint) == 2


class TestCostAggregation:
    """Tests for CostAggregation class."""

    def test_aggregation_fields(self):
        now = datetime.now()
        agg = CostAggregation(
            period=AggregationPeriod.DAY,
            start_time=now,
            end_time=now + timedelta(days=1),
            total_cost=10.5,
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_requests=100,
        )

        assert agg.period == AggregationPeriod.DAY
        assert agg.total_cost == 10.5
        assert agg.total_requests == 100


class TestAggregationPeriod:
    """Tests for AggregationPeriod enum."""

    def test_period_values(self):
        assert AggregationPeriod.HOUR.value == "hour"
        assert AggregationPeriod.DAY.value == "day"
        assert AggregationPeriod.WEEK.value == "week"
        assert AggregationPeriod.MONTH.value == "month"


class TestCostTrackingHook:
    """Tests for CostTrackingHook class."""

    @pytest.mark.asyncio
    async def test_on_llm_end_records_cost(self):
        tracker = CostTracker()
        hook = CostTrackingHook(tracker)

        ctx = HookContext(
            run_id="run-123",
            endpoint="/triage",
            user=UserInfo(id="user-456"),
            usage=UsageInfo(input_tokens=1000, output_tokens=500),
            metadata={"model": "gpt-4o"},
        )

        result = await hook.on_llm_end(ctx)

        assert result.action == HookResultAction.PROCEED

        records = await tracker.get_records()
        assert len(records) == 1
        assert records[0].model == "gpt-4o"
        assert records[0].input_tokens == 1000

    @pytest.mark.asyncio
    async def test_on_llm_end_no_usage(self):
        tracker = CostTracker()
        hook = CostTrackingHook(tracker)

        ctx = HookContext(
            run_id="run-123",
            endpoint="/triage",
        )

        result = await hook.on_llm_end(ctx)

        assert result.action == HookResultAction.PROCEED

        records = await tracker.get_records()
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_on_response_adds_cost_metadata(self):
        tracker = CostTracker()
        hook = CostTrackingHook(tracker)

        ctx = HookContext(
            run_id="run-123",
            endpoint="/triage",
            usage=UsageInfo(input_tokens=1000, output_tokens=500),
            metadata={"model": "gpt-4o"},
        )

        result = await hook.on_response(ctx)

        assert result.action == HookResultAction.PROCEED
        assert "total_cost" in ctx.metadata
        assert ctx.metadata["total_cost"] > 0
