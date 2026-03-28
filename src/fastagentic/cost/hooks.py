"""Cost tracking hooks for automatic cost capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastagentic.hooks.base import Hook, HookContext, HookResult

if TYPE_CHECKING:
    from fastagentic.cost.tracker import CostTracker


class CostTrackingHook(Hook):
    """Hook for automatic cost tracking.

    Captures LLM usage from hook context and records costs.

    Example:
        tracker = CostTracker()
        hook = CostTrackingHook(tracker)

        app = App(
            hooks=[hook],
            # or
            cost_tracker=tracker,  # App auto-creates the hook
        )
    """

    def __init__(
        self,
        tracker: CostTracker,
        *,
        track_tool_calls: bool = False,
    ) -> None:
        """Initialize cost tracking hook.

        Args:
            tracker: CostTracker instance
            track_tool_calls: Whether to track tool call costs separately
        """
        self._tracker = tracker
        self._track_tool_calls = track_tool_calls

    async def on_llm_end(self, ctx: HookContext) -> HookResult:
        """Record LLM cost after completion.

        Args:
            ctx: Hook context with usage info

        Returns:
            HookResult to proceed
        """
        if ctx.usage is None:
            return HookResult.proceed()

        # Extract model from metadata or default
        model = ctx.metadata.get("model", "unknown")

        # Get user/tenant info
        user_id = ctx.user.id if ctx.user else None
        tenant_id = ctx.metadata.get("tenant_id")

        # Record the cost
        await self._tracker.record(
            run_id=ctx.run_id,
            model=model,
            input_tokens=ctx.usage.input_tokens,
            output_tokens=ctx.usage.output_tokens,
            cached_tokens=ctx.usage.cached_tokens,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=ctx.endpoint,
            metadata={
                "request_id": ctx.metadata.get("request_id"),
                "latency_ms": ctx.usage.latency_ms,
            },
        )

        return HookResult.proceed()

    async def on_tool_result(self, ctx: HookContext) -> HookResult:
        """Optionally track tool call costs.

        Some tools may incur additional costs (e.g., web search, API calls).

        Args:
            ctx: Hook context with tool info

        Returns:
            HookResult to proceed
        """
        if not self._track_tool_calls:
            return HookResult.proceed()

        # Check if tool has cost metadata
        tool_cost = ctx.metadata.get("tool_cost")
        if tool_cost is None:
            return HookResult.proceed()

        user_id = ctx.user.id if ctx.user else None
        tenant_id = ctx.metadata.get("tenant_id")

        # Record tool cost as a separate record
        await self._tracker.record(
            run_id=ctx.run_id,
            model=f"tool:{ctx.tool_name}",
            input_tokens=0,
            output_tokens=0,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=ctx.endpoint,
            metadata={
                "tool_name": ctx.tool_name,
                "tool_cost": tool_cost,
            },
        )

        return HookResult.proceed()

    async def on_response(self, ctx: HookContext) -> HookResult:
        """Final cost summary in response metadata.

        Args:
            ctx: Hook context

        Returns:
            HookResult with cost metadata
        """
        # Add cost info to response metadata if available
        if ctx.usage is not None:
            model = ctx.metadata.get("model", "unknown")
            cost = self._tracker.calculate_cost(
                model=model,
                input_tokens=ctx.usage.input_tokens,
                output_tokens=ctx.usage.output_tokens,
                cached_tokens=ctx.usage.cached_tokens,
            )
            ctx.metadata["total_cost"] = cost

        return HookResult.proceed()
