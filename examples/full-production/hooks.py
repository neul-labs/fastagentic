"""Custom lifecycle hooks for production observability and auditing."""

from fastagentic.hooks import Hook, HookContext


class ObservabilityHook(Hook):
    """Hook for custom observability."""

    async def on_request(self, ctx: HookContext) -> None:
        """Log incoming requests."""
        ctx.logger.info("request_received", path=ctx.request.path)

    async def on_response(self, ctx: HookContext) -> None:
        """Log outgoing responses."""
        ctx.logger.info("response_sent", latency_ms=ctx.latency_ms)

    async def on_error(self, ctx: HookContext, error: Exception) -> None:
        """Log errors."""
        ctx.logger.error("request_error", error=str(error))


class AuditHook(Hook):
    """Hook for audit logging."""

    async def on_tool_call(self, ctx: HookContext, tool_name: str, args: dict) -> None:
        """Audit tool calls."""
        ctx.logger.info("tool_called", tool=tool_name, user=ctx.user_id)


observability_hook = ObservabilityHook()
audit_hook = AuditHook()
