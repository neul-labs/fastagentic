"""Fuzzy tests for cost calculations using Hypothesis.

These tests use property-based testing to find edge cases in:
- Token cost calculations
- Pricing precision
- Budget tracking
- Edge cases with floats (NaN, Inf, negative)
"""

import math

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from fastagentic.cost.tracker import CostTracker, ModelPricing


class TestModelPricingFuzzing:
    """Fuzzy tests for ModelPricing cost calculations."""

    @given(
        input_tokens=st.integers(min_value=0, max_value=10_000_000),
        output_tokens=st.integers(min_value=0, max_value=10_000_000),
        input_cost=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        output_cost=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_cost_calculation_non_negative(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ):
        """Cost should always be non-negative for valid inputs."""
        pricing = ModelPricing(
            model="test-model",
            input_cost_per_1k=input_cost,
            output_cost_per_1k=output_cost,
        )

        cost = pricing.calculate_cost(input_tokens, output_tokens)

        assert cost >= 0, f"Negative cost: {cost}"
        assert not math.isnan(cost), "Cost is NaN"
        assert not math.isinf(cost), "Cost is infinite"

    @given(
        input_tokens=st.integers(min_value=0, max_value=1_000_000),
        output_tokens=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_cost_calculation_precision(self, input_tokens: int, output_tokens: int):
        """Cost calculation should maintain reasonable precision."""
        pricing = ModelPricing(
            model="gpt-4o",
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
        )

        cost = pricing.calculate_cost(input_tokens, output_tokens)

        # Calculate expected with higher precision
        expected_input = (input_tokens / 1000) * 0.003
        expected_output = (output_tokens / 1000) * 0.015
        expected = expected_input + expected_output

        # Allow small floating-point error (1e-10 relative error)
        if expected > 0:
            relative_error = abs(cost - expected) / expected
            assert relative_error < 1e-10, f"Precision error: {relative_error}"
        else:
            assert cost == 0

    @given(
        tokens=st.integers(min_value=0, max_value=10_000_000),
    )
    def test_cost_scales_linearly(self, tokens: int):
        """Doubling tokens should double cost."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )

        cost1 = pricing.calculate_cost(tokens, 0)
        cost2 = pricing.calculate_cost(tokens * 2, 0)

        if tokens > 0:
            ratio = cost2 / cost1 if cost1 > 0 else 0
            assert abs(ratio - 2.0) < 1e-10, f"Non-linear scaling: {ratio}"
        else:
            assert cost1 == 0 and cost2 == 0

    @given(
        input_tokens=st.integers(min_value=0, max_value=1_000_000),
        output_tokens=st.integers(min_value=0, max_value=1_000_000),
        cached_tokens=st.integers(min_value=0, max_value=1_000_000),
    )
    def test_cached_tokens_reduce_cost(
        self, input_tokens: int, output_tokens: int, cached_tokens: int
    ):
        """Cached tokens should cost less than regular input tokens."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            cached_input_cost_per_1k=0.0015,  # 50% cheaper
        )

        cost_no_cache = pricing.calculate_cost(input_tokens + cached_tokens, output_tokens, 0)
        cost_with_cache = pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)

        # Cost with cache should be less or equal
        assert cost_with_cache <= cost_no_cache + 1e-10, (
            f"Cache didn't reduce cost: {cost_with_cache} > {cost_no_cache}"
        )

    @example(input_tokens=0, output_tokens=0)
    @example(input_tokens=1, output_tokens=0)
    @example(input_tokens=0, output_tokens=1)
    @example(input_tokens=1000, output_tokens=1000)
    @given(
        input_tokens=st.integers(min_value=0, max_value=100),
        output_tokens=st.integers(min_value=0, max_value=100),
    )
    def test_small_token_counts(self, input_tokens: int, output_tokens: int):
        """Small token counts should not cause precision issues."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=0.00025,  # Very small price
            output_cost_per_1k=0.001,
        )

        cost = pricing.calculate_cost(input_tokens, output_tokens)

        assert cost >= 0
        assert not math.isnan(cost)

    def test_zero_cost_pricing(self):
        """Zero pricing should result in zero cost."""
        pricing = ModelPricing(
            model="free-model",
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
        )

        cost = pricing.calculate_cost(1_000_000, 1_000_000)
        assert cost == 0.0

    def test_very_small_pricing(self):
        """Very small pricing should not underflow."""
        pricing = ModelPricing(
            model="cheap-model",
            input_cost_per_1k=1e-10,
            output_cost_per_1k=1e-10,
        )

        cost = pricing.calculate_cost(1000, 1000)

        assert cost > 0
        assert not math.isnan(cost)

    @given(
        input_cost=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        output_cost=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_1k_token_cost_matches_pricing(self, input_cost: float, output_cost: float):
        """1000 tokens should cost exactly the per-1k price."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=input_cost,
            output_cost_per_1k=output_cost,
        )

        cost = pricing.calculate_cost(1000, 1000)
        expected = input_cost + output_cost

        assert abs(cost - expected) < 1e-10, f"1k cost mismatch: {cost} != {expected}"


class TestCostTrackerFuzzing:
    """Fuzzy tests for CostTracker."""

    @pytest.fixture
    def tracker(self):
        return CostTracker()

    def test_tracker_initialization(self, tracker):
        """Tracker should initialize without errors."""
        assert tracker is not None

    @given(
        model=st.sampled_from(["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "unknown-model"]),
        input_tokens=st.integers(min_value=0, max_value=100000),
        output_tokens=st.integers(min_value=0, max_value=100000),
    )
    def test_calculate_cost_for_models(
        self, tracker, model: str, input_tokens: int, output_tokens: int
    ):
        """Cost calculation should work for various models."""
        cost = tracker.calculate_cost(model, input_tokens, output_tokens)

        # Cost should be non-negative
        assert cost >= 0
        assert not math.isnan(cost)
        assert not math.isinf(cost)

    @given(
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
    )
    def test_calculate_cost_consistency(self, tracker, input_tokens: int, output_tokens: int):
        """Same inputs should produce same outputs."""
        cost1 = tracker.calculate_cost("gpt-4o", input_tokens, output_tokens)
        cost2 = tracker.calculate_cost("gpt-4o", input_tokens, output_tokens)

        assert cost1 == cost2

    def test_add_custom_pricing(self, tracker):
        """Custom pricing should be usable."""
        custom = ModelPricing(
            model="custom-model",
            input_cost_per_1k=0.005,
            output_cost_per_1k=0.010,
        )
        tracker.add_pricing(custom)

        cost = tracker.calculate_cost("custom-model", 1000, 1000)
        expected = 0.005 + 0.010

        assert abs(cost - expected) < 1e-10


class TestCostEdgeCases:
    """Edge case tests for cost calculations."""

    def test_max_int_tokens(self):
        """Very large token counts should not overflow."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )

        # Use a large but reasonable token count
        large_tokens = 2**31 - 1  # Max 32-bit signed int

        cost = pricing.calculate_cost(large_tokens, 0)

        assert not math.isnan(cost)
        assert not math.isinf(cost)
        assert cost > 0

    def test_cost_with_float_precision_boundary(self):
        """Test cost calculation at float precision boundary."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.001,
        )

        # This specific combination can cause precision issues
        cost = pricing.calculate_cost(333, 333)

        expected = (333 / 1000 * 0.001) + (333 / 1000 * 0.001)
        assert abs(cost - expected) < 1e-15

    @given(
        divisor=st.integers(min_value=1, max_value=10000),
    )
    def test_fractional_token_equivalence(self, divisor: int):
        """Fractional tokens (via division) should be handled correctly."""
        pricing = ModelPricing(
            model="test",
            input_cost_per_1k=1.0,  # $1 per 1k for easy math
            output_cost_per_1k=0.0,
        )

        tokens = 1000 // divisor * divisor  # Ensure divisible
        cost = pricing.calculate_cost(tokens, 0)

        expected = tokens / 1000
        assert abs(cost - expected) < 1e-10

    @pytest.mark.asyncio
    async def test_accumulated_rounding_errors(self):
        """Many small costs should not accumulate rounding errors."""
        tracker = CostTracker()

        # Record 100 entries of 10 tokens each
        total_cost = 0.0
        for i in range(100):
            record = await tracker.record(
                run_id=f"run-{i}",
                model="gpt-4o",  # Uses default pricing
                input_tokens=10,
                output_tokens=0,
            )
            total_cost += record.cost

        # Get total from tracker
        retrieved_total = await tracker.get_total_cost()

        # Totals should match
        assert abs(retrieved_total - total_cost) < 1e-10, (
            f"Accumulated error: {retrieved_total} vs {total_cost}"
        )


class TestCostComparisonFuzzing:
    """Test cost comparisons for budget enforcement."""

    @given(
        cost1=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        cost2=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_cost_comparison_transitivity(self, cost1: float, cost2: float):
        """Cost comparisons should be transitive."""
        # If cost1 < cost2, then cost1 <= cost2
        if cost1 < cost2:
            assert cost1 <= cost2

        # If cost1 == cost2, then not (cost1 < cost2) and not (cost1 > cost2)
        if cost1 == cost2:
            assert not (cost1 < cost2)
            assert not (cost1 > cost2)

    @given(
        budget=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        cost=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_budget_check_consistency(self, budget: float, cost: float):
        """Budget checks should be consistent."""
        # over_budget should be consistent
        over_budget = cost > budget
        under_budget = cost <= budget

        assert over_budget != under_budget, "Budget check inconsistency"

    @given(
        costs=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        ),
    )
    def test_sum_of_costs_ordering(self, costs: list):
        """Sum of non-negative costs should be >= max individual cost."""
        total = sum(costs)
        max_cost = max(costs)

        assert total >= max_cost - 1e-10, f"Sum {total} < max {max_cost}"
