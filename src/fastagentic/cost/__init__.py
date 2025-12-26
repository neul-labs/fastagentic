"""Cost tracking and analytics for FastAgentic.

Provides automatic cost logging, aggregation, and reporting
for LLM usage across your agentic applications.
"""

from fastagentic.cost.hooks import CostTrackingHook
from fastagentic.cost.tracker import (
    DEFAULT_PRICING,
    CostAggregation,
    CostRecord,
    CostTracker,
    ModelPricing,
)

__all__ = [
    "CostTracker",
    "CostRecord",
    "CostAggregation",
    "ModelPricing",
    "DEFAULT_PRICING",
    "CostTrackingHook",
]
