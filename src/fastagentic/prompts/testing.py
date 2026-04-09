"""A/B testing for prompts in FastAgentic."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastagentic.prompts.template import PromptTemplate


class VariantSelectionStrategy(str, Enum):
    """Strategy for selecting variants."""

    RANDOM = "random"  # Random selection based on weights
    ROUND_ROBIN = "round_robin"  # Cycle through variants
    STICKY = "sticky"  # Same user always gets same variant
    GRADUAL_ROLLOUT = "gradual_rollout"  # Gradually shift to challenger


@dataclass
class PromptVariant:
    """A variant in an A/B test.

    Attributes:
        name: Variant identifier (e.g., "control", "challenger")
        template: The prompt template for this variant
        weight: Selection weight (higher = more likely)
        metadata: Additional variant metadata
    """

    name: str
    template: PromptTemplate
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantMetrics:
    """Metrics for a single variant.

    Attributes:
        impressions: Number of times shown
        conversions: Number of successful outcomes
        total_latency_ms: Total latency in milliseconds
        total_tokens: Total tokens used
        total_cost: Total cost incurred
        errors: Number of errors
        custom_metrics: Additional tracked metrics
    """

    impressions: int = 0
    conversions: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    errors: int = 0
    custom_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.impressions == 0:
            return 0.0
        return self.total_latency_ms / self.impressions

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens per request."""
        if self.impressions == 0:
            return 0.0
        return self.total_tokens / self.impressions

    @property
    def avg_cost(self) -> float:
        """Calculate average cost per request."""
        if self.impressions == 0:
            return 0.0
        return self.total_cost / self.impressions


@dataclass
class ABTestResult:
    """Results of an A/B test.

    Attributes:
        test_name: Name of the test
        variant_metrics: Metrics per variant
        winner: Winning variant (if determined)
        confidence: Statistical confidence in the winner
        started_at: When the test started
        ended_at: When the test ended (if completed)
    """

    test_name: str
    variant_metrics: dict[str, VariantMetrics] = field(default_factory=dict)
    winner: str | None = None
    confidence: float = 0.0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def is_complete(self) -> bool:
        """Check if test is complete."""
        return self.ended_at is not None

    @property
    def total_impressions(self) -> int:
        """Get total impressions across all variants."""
        return sum(m.impressions for m in self.variant_metrics.values())

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of results."""
        summary: dict[str, Any] = {
            "test_name": self.test_name,
            "total_impressions": self.total_impressions,
            "is_complete": self.is_complete,
            "variants": {},
        }

        for name, metrics in self.variant_metrics.items():
            summary["variants"][name] = {
                "impressions": metrics.impressions,
                "conversion_rate": f"{metrics.conversion_rate:.2%}",
                "avg_latency_ms": f"{metrics.avg_latency_ms:.1f}",
                "avg_tokens": f"{metrics.avg_tokens:.0f}",
                "avg_cost": f"${metrics.avg_cost:.4f}",
            }

        if self.winner:
            summary["winner"] = self.winner
            summary["confidence"] = f"{self.confidence:.1%}"

        return summary


class ABTest:
    """A/B test for comparing prompt variants.

    Example:
        # Create test with control and challenger
        test = ABTest(
            name="greeting-test",
            variants=[
                PromptVariant(
                    name="control",
                    template=PromptTemplate(
                        name="greeting",
                        content="Hello {{name}}, how can I help?",
                    ),
                    weight=0.5,
                ),
                PromptVariant(
                    name="challenger",
                    template=PromptTemplate(
                        name="greeting",
                        content="Hi {{name}}! What brings you here today?",
                    ),
                    weight=0.5,
                ),
            ],
        )

        # Select variant for a user
        variant = test.select_variant(user_id="user-123")
        rendered = variant.template.render(name="Alice")

        # Record outcome
        test.record_impression(variant.name)
        test.record_conversion(variant.name)
        test.record_latency(variant.name, latency_ms=150)

        # Get results
        results = test.get_results()
        print(results.get_summary())
    """

    def __init__(
        self,
        name: str,
        variants: list[PromptVariant],
        *,
        strategy: VariantSelectionStrategy = VariantSelectionStrategy.RANDOM,
        min_impressions: int = 100,
        confidence_threshold: float = 0.95,
    ) -> None:
        """Initialize A/B test.

        Args:
            name: Test name
            variants: List of variants to test
            strategy: How to select variants
            min_impressions: Minimum impressions before declaring winner
            confidence_threshold: Required confidence for winner
        """
        if len(variants) < 2:
            raise ValueError("A/B test requires at least 2 variants")

        self.name = name
        self.variants = {v.name: v for v in variants}
        self.strategy = strategy
        self.min_impressions = min_impressions
        self.confidence_threshold = confidence_threshold

        self._metrics: dict[str, VariantMetrics] = {v.name: VariantMetrics() for v in variants}
        self._started_at = time.time()
        self._ended_at: float | None = None
        self._winner: str | None = None
        self._round_robin_index = 0

    @property
    def is_active(self) -> bool:
        """Check if test is still active."""
        return self._ended_at is None

    def select_variant(
        self,
        user_id: str | None = None,
    ) -> PromptVariant:
        """Select a variant for a request.

        Args:
            user_id: Optional user ID for sticky assignment

        Returns:
            Selected variant
        """
        if not self.is_active and self._winner and self._winner in self.variants:
            # Return winner if test is complete
            return self.variants[self._winner]

        variant_list = list(self.variants.values())

        if self.strategy == VariantSelectionStrategy.RANDOM:
            return self._select_random(variant_list)

        elif self.strategy == VariantSelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(variant_list)

        elif self.strategy == VariantSelectionStrategy.STICKY:
            if user_id:
                return self._select_sticky(variant_list, user_id)
            return self._select_random(variant_list)

        elif self.strategy == VariantSelectionStrategy.GRADUAL_ROLLOUT:
            return self._select_gradual(variant_list)

        return self._select_random(variant_list)

    def _select_random(self, variants: list[PromptVariant]) -> PromptVariant:
        """Random selection based on weights."""
        total_weight = sum(v.weight for v in variants)
        r = random.random() * total_weight

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant

        return variants[-1]

    def _select_round_robin(self, variants: list[PromptVariant]) -> PromptVariant:
        """Round-robin selection."""
        variant = variants[self._round_robin_index % len(variants)]
        self._round_robin_index += 1
        return variant

    def _select_sticky(
        self,
        variants: list[PromptVariant],
        user_id: str,
    ) -> PromptVariant:
        """Sticky selection - same user always gets same variant."""
        # Hash user_id to get consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        index = hash_value % len(variants)
        return variants[index]

    def _select_gradual(self, variants: list[PromptVariant]) -> PromptVariant:
        """Gradual rollout - shift traffic over time."""
        elapsed = time.time() - self._started_at
        hours_elapsed = elapsed / 3600

        # Start at 10% challenger, increase 5% per hour up to 50%
        challenger_percent = min(0.1 + (hours_elapsed * 0.05), 0.5)

        if random.random() < challenger_percent:
            # Find challenger (non-control variant)
            for v in variants:
                if v.name != "control":
                    return v

        # Return control
        for v in variants:
            if v.name == "control":
                return v

        return variants[0]

    def record_impression(self, variant_name: str) -> None:
        """Record that a variant was shown.

        Args:
            variant_name: Name of the variant
        """
        if variant_name in self._metrics:
            self._metrics[variant_name].impressions += 1

    def record_conversion(self, variant_name: str) -> None:
        """Record a successful conversion.

        Args:
            variant_name: Name of the variant
        """
        if variant_name in self._metrics:
            self._metrics[variant_name].conversions += 1

    def record_latency(self, variant_name: str, latency_ms: float) -> None:
        """Record request latency.

        Args:
            variant_name: Name of the variant
            latency_ms: Latency in milliseconds
        """
        if variant_name in self._metrics:
            self._metrics[variant_name].total_latency_ms += latency_ms

    def record_tokens(self, variant_name: str, tokens: int) -> None:
        """Record token usage.

        Args:
            variant_name: Name of the variant
            tokens: Number of tokens used
        """
        if variant_name in self._metrics:
            self._metrics[variant_name].total_tokens += tokens

    def record_cost(self, variant_name: str, cost: float) -> None:
        """Record cost.

        Args:
            variant_name: Name of the variant
            cost: Cost in dollars
        """
        if variant_name in self._metrics:
            self._metrics[variant_name].total_cost += cost

    def record_error(self, variant_name: str) -> None:
        """Record an error.

        Args:
            variant_name: Name of the variant
        """
        if variant_name in self._metrics:
            self._metrics[variant_name].errors += 1

    def record_custom_metric(
        self,
        variant_name: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a custom metric.

        Args:
            variant_name: Name of the variant
            metric_name: Name of the metric
            value: Metric value (will be summed)
        """
        if variant_name in self._metrics:
            metrics = self._metrics[variant_name]
            if metric_name not in metrics.custom_metrics:
                metrics.custom_metrics[metric_name] = 0.0
            metrics.custom_metrics[metric_name] += value

    def get_results(self) -> ABTestResult:
        """Get current test results.

        Returns:
            ABTestResult with metrics and winner (if determined)
        """
        # Use manually set winner if available, otherwise calculate
        if self._winner:
            winner = self._winner
            confidence = 1.0  # Manual override has full confidence
        else:
            winner, confidence = self._determine_winner()

        return ABTestResult(
            test_name=self.name,
            variant_metrics=dict(self._metrics),
            winner=winner,
            confidence=confidence,
            started_at=self._started_at,
            ended_at=self._ended_at,
        )

    def _determine_winner(self) -> tuple[str | None, float]:
        """Determine if there's a statistically significant winner.

        Returns:
            Tuple of (winner_name, confidence)
        """
        # Check minimum impressions
        total_impressions = sum(m.impressions for m in self._metrics.values())
        if total_impressions < self.min_impressions:
            return None, 0.0

        # Simple winner determination based on conversion rate
        # In production, use proper statistical tests (chi-squared, etc.)
        best_variant: str | None = None
        best_rate = 0.0
        second_best_rate = 0.0

        for name, metrics in self._metrics.items():
            rate = metrics.conversion_rate
            if rate > best_rate:
                second_best_rate = best_rate
                best_rate = rate
                best_variant = name
            elif rate > second_best_rate:
                second_best_rate = rate

        if best_variant is None:
            return None, 0.0

        # Simple confidence calculation (should use proper stats)
        if second_best_rate == 0:
            confidence = 0.99 if best_rate > 0 else 0.0
        else:
            # Relative improvement
            improvement = (best_rate - second_best_rate) / second_best_rate
            # More impressions = more confidence
            impression_factor = min(total_impressions / (self.min_impressions * 2), 1.0)
            confidence = min(improvement * impression_factor + 0.5, 0.99)

        if confidence >= self.confidence_threshold:
            return best_variant, confidence

        return None, confidence

    def complete_test(self, winner: str | None = None) -> ABTestResult:
        """Complete the test and optionally declare a winner.

        Args:
            winner: Override winner (uses calculated if None)

        Returns:
            Final test results
        """
        self._ended_at = time.time()

        if winner:
            self._winner = winner
        else:
            self._winner, _ = self._determine_winner()

        return self.get_results()
