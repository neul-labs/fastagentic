"""Audit hooks for automatic event logging."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastagentic.audit.logger import AuditEvent, AuditEventType, AuditSeverity
from fastagentic.hooks.base import Hook, HookContext, HookResult

if TYPE_CHECKING:
    from fastagentic.audit.logger import AuditLogger


class AuditHook(Hook):
    """Hook for automatic audit logging.

    Captures request lifecycle events and logs them to the audit system.

    Example:
        audit_logger = AuditLogger()
        hook = AuditHook(audit_logger)

        app = App(
            hooks=[hook],
            # or
            audit_logger=audit_logger,  # App auto-creates the hook
        )
    """

    def __init__(
        self,
        audit_logger: AuditLogger,
        *,
        log_requests: bool = True,
        log_tools: bool = True,
        log_llm: bool = False,
        include_metadata: bool = True,
    ) -> None:
        """Initialize audit hook.

        Args:
            audit_logger: AuditLogger instance
            log_requests: Log request start/end events
            log_tools: Log tool invocation events
            log_llm: Log LLM call events
            include_metadata: Include additional metadata in events
        """
        self._logger = audit_logger
        self._log_requests = log_requests
        self._log_tools = log_tools
        self._log_llm = log_llm
        self._include_metadata = include_metadata
        self._request_starts: dict[str, float] = {}

    async def on_request(self, ctx: HookContext) -> HookResult:
        """Log request start.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_requests:
            return HookResult.proceed()

        # Track start time
        self._request_starts[ctx.run_id] = time.time()

        user_id = ctx.user.id if ctx.user else None
        ip_address = ctx.metadata.get("ip_address")

        await self._logger.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_START,
                severity=AuditSeverity.INFO,
                run_id=ctx.run_id,
                user_id=user_id,
                tenant_id=ctx.metadata.get("tenant_id"),
                endpoint=ctx.endpoint,
                action="request",
                outcome="started",
                ip_address=ip_address,
                metadata=self._extract_metadata(ctx) if self._include_metadata else {},
            )
        )

        return HookResult.proceed()

    async def on_response(self, ctx: HookContext) -> HookResult:
        """Log request completion.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_requests:
            return HookResult.proceed()

        # Calculate duration
        start_time = self._request_starts.pop(ctx.run_id, None)
        duration_ms = None
        if start_time:
            duration_ms = (time.time() - start_time) * 1000

        user_id = ctx.user.id if ctx.user else None

        await self._logger.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_END,
                severity=AuditSeverity.INFO,
                run_id=ctx.run_id,
                user_id=user_id,
                tenant_id=ctx.metadata.get("tenant_id"),
                endpoint=ctx.endpoint,
                action="request",
                outcome="success",
                ip_address=ctx.metadata.get("ip_address"),
                duration_ms=duration_ms,
                metadata={
                    **(self._extract_metadata(ctx) if self._include_metadata else {}),
                    "duration_ms": duration_ms,
                },
            )
        )

        return HookResult.proceed()

    async def on_error(self, ctx: HookContext) -> HookResult:
        """Log request error.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_requests:
            return HookResult.proceed()

        # Calculate duration
        start_time = self._request_starts.pop(ctx.run_id, None)
        duration_ms = None
        if start_time:
            duration_ms = (time.time() - start_time) * 1000

        user_id = ctx.user.id if ctx.user else None
        error_msg = str(ctx.error) if ctx.error else "Unknown error"

        await self._logger.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_ERROR,
                severity=AuditSeverity.ERROR,
                run_id=ctx.run_id,
                user_id=user_id,
                tenant_id=ctx.metadata.get("tenant_id"),
                endpoint=ctx.endpoint,
                action="request",
                outcome="error",
                reason=error_msg,
                ip_address=ctx.metadata.get("ip_address"),
                duration_ms=duration_ms,
                metadata={
                    **(self._extract_metadata(ctx) if self._include_metadata else {}),
                    "duration_ms": duration_ms,
                    "error_type": type(ctx.error).__name__ if ctx.error else None,
                },
            )
        )

        return HookResult.proceed()

    async def on_tool_call(self, ctx: HookContext) -> HookResult:
        """Log tool invocation start.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_tools or not ctx.tool_name:
            return HookResult.proceed()

        user_id = ctx.user.id if ctx.user else None

        await self._logger.log_tool(
            event_type=AuditEventType.TOOL_INVOKE,
            tool_name=ctx.tool_name,
            user_id=user_id,
            tenant_id=ctx.metadata.get("tenant_id"),
            endpoint=ctx.endpoint,
            outcome="started",
            run_id=ctx.run_id,
            metadata={
                "tool_input": ctx.tool_input if self._include_metadata else None,
            },
        )

        return HookResult.proceed()

    async def on_tool_result(self, ctx: HookContext) -> HookResult:
        """Log tool invocation completion.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_tools or not ctx.tool_name:
            return HookResult.proceed()

        user_id = ctx.user.id if ctx.user else None

        # Check if tool errored
        tool_error = ctx.metadata.get("tool_error")
        event_type = AuditEventType.TOOL_ERROR if tool_error else AuditEventType.TOOL_COMPLETE

        await self._logger.log_tool(
            event_type=event_type,
            tool_name=ctx.tool_name,
            user_id=user_id,
            tenant_id=ctx.metadata.get("tenant_id"),
            endpoint=ctx.endpoint,
            outcome="error" if tool_error else "success",
            reason=str(tool_error) if tool_error else None,
            run_id=ctx.run_id,
            duration_ms=ctx.metadata.get("tool_duration_ms"),
            metadata={
                "tool_output": ctx.tool_output if self._include_metadata else None,
            },
        )

        return HookResult.proceed()

    async def on_llm_start(self, ctx: HookContext) -> HookResult:
        """Log LLM call start.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_llm:
            return HookResult.proceed()

        user_id = ctx.user.id if ctx.user else None
        model = ctx.metadata.get("model", "unknown")

        await self._logger.log(
            AuditEvent(
                event_type=AuditEventType.AGENT_INVOKE,
                severity=AuditSeverity.INFO,
                run_id=ctx.run_id,
                user_id=user_id,
                tenant_id=ctx.metadata.get("tenant_id"),
                endpoint=ctx.endpoint,
                resource=f"model:{model}",
                outcome="started",
                metadata={
                    "model": model,
                    "message_count": len(ctx.messages) if ctx.messages else 0,
                },
            )
        )

        return HookResult.proceed()

    async def on_llm_end(self, ctx: HookContext) -> HookResult:
        """Log LLM call completion.

        Args:
            ctx: Hook context

        Returns:
            HookResult to proceed
        """
        if not self._log_llm:
            return HookResult.proceed()

        user_id = ctx.user.id if ctx.user else None
        model = ctx.metadata.get("model", "unknown")

        usage_data = {}
        if ctx.usage:
            usage_data = {
                "input_tokens": ctx.usage.input_tokens,
                "output_tokens": ctx.usage.output_tokens,
                "total_tokens": ctx.usage.total_tokens,
                "latency_ms": ctx.usage.latency_ms,
            }

        await self._logger.log(
            AuditEvent(
                event_type=AuditEventType.AGENT_COMPLETE,
                severity=AuditSeverity.INFO,
                run_id=ctx.run_id,
                user_id=user_id,
                tenant_id=ctx.metadata.get("tenant_id"),
                endpoint=ctx.endpoint,
                resource=f"model:{model}",
                outcome="success",
                duration_ms=ctx.usage.latency_ms if ctx.usage else None,
                metadata={
                    "model": model,
                    **usage_data,
                },
            )
        )

        return HookResult.proceed()

    def _extract_metadata(self, ctx: HookContext) -> dict[str, Any]:
        """Extract relevant metadata from context.

        Args:
            ctx: Hook context

        Returns:
            Metadata dictionary
        """
        metadata: dict[str, Any] = {}

        if ctx.user:
            metadata["user_email"] = ctx.user.email
            metadata["user_roles"] = ctx.user.roles

        if ctx.usage:
            metadata["tokens"] = {
                "input": ctx.usage.input_tokens,
                "output": ctx.usage.output_tokens,
                "total": ctx.usage.total_tokens,
            }

        return metadata
