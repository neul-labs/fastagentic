"""Audit logging implementation for FastAgentic."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token_refresh"

    # Authorization events
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    PERMISSION_CHECK = "access.permission_check"

    # Request events
    REQUEST_START = "request.start"
    REQUEST_END = "request.end"
    REQUEST_ERROR = "request.error"

    # Agent events
    AGENT_INVOKE = "agent.invoke"
    AGENT_COMPLETE = "agent.complete"
    AGENT_ERROR = "agent.error"

    # Tool events
    TOOL_INVOKE = "tool.invoke"
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"
    TOOL_BLOCKED = "tool.blocked"

    # Resource events
    RESOURCE_READ = "resource.read"
    RESOURCE_WRITE = "resource.write"
    RESOURCE_DELETE = "resource.delete"

    # Policy events
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_WARN = "policy.warn"
    BUDGET_EXCEEDED = "policy.budget_exceeded"
    RATE_LIMITED = "policy.rate_limited"

    # Security events
    SECURITY_ALERT = "security.alert"
    PROMPT_INJECTION = "security.prompt_injection"
    PII_DETECTED = "security.pii_detected"
    CONTENT_BLOCKED = "security.content_blocked"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGE = "system.config_change"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event.

    Attributes:
        event_type: Type of the event
        severity: Event severity
        timestamp: Unix timestamp
        run_id: Associated run ID
        user_id: User who triggered the event
        tenant_id: Tenant/organization
        endpoint: Endpoint path
        action: Action performed
        resource: Resource affected
        outcome: Result (success/failure)
        reason: Explanation of the outcome
        ip_address: Client IP address
        user_agent: Client user agent
        duration_ms: Duration in milliseconds
        metadata: Additional context
    """

    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: float = field(default_factory=time.time)
    run_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    endpoint: str | None = None
    action: str | None = None
    resource: str | None = None
    outcome: str | None = None
    reason: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditStore(Protocol):
    """Protocol for audit log storage."""

    async def write(self, event: AuditEvent) -> None:
        """Write an audit event."""
        ...

    async def query(
        self,
        event_types: list[AuditEventType] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        severity: AuditSeverity | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query audit events."""
        ...


class InMemoryAuditStore:
    """In-memory audit store for development/testing."""

    def __init__(self, max_events: int = 10000) -> None:
        self._events: list[AuditEvent] = []
        self._max_events = max_events

    async def write(self, event: AuditEvent) -> None:
        """Write an audit event."""
        self._events.append(event)
        # Trim old events if over limit
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

    async def query(
        self,
        event_types: list[AuditEventType] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        severity: AuditSeverity | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query audit events."""
        results = []
        for event in reversed(self._events):
            if event_types and event.event_type not in event_types:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if user_id and event.user_id != user_id:
                continue
            if tenant_id and event.tenant_id != tenant_id:
                continue
            if severity and event.severity != severity:
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results


class LoggingAuditStore:
    """Audit store that writes to Python logging.

    Useful for simple setups or when you want audit logs
    in your existing log aggregation system.
    """

    def __init__(
        self,
        logger_name: str = "fastagentic.audit",
        log_format: str = "json",
    ) -> None:
        """Initialize logging audit store.

        Args:
            logger_name: Name for the audit logger
            log_format: "json" or "text"
        """
        self._logger = logging.getLogger(logger_name)
        self._format = log_format

    async def write(self, event: AuditEvent) -> None:
        """Write an audit event to logs."""
        level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(event.severity, logging.INFO)

        if self._format == "json":
            self._logger.log(level, event.to_json())
        else:
            self._logger.log(
                level,
                f"[{event.event_type.value}] "
                f"user={event.user_id} "
                f"endpoint={event.endpoint} "
                f"outcome={event.outcome} "
                f"reason={event.reason}",
            )

    async def query(
        self,
        event_types: list[AuditEventType] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        severity: AuditSeverity | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query is not supported for logging store."""
        raise NotImplementedError(
            "LoggingAuditStore does not support querying. "
            "Use your log aggregation system to query logs."
        )


class AuditLogger:
    """Audit logger for recording security and compliance events.

    Example:
        audit = AuditLogger()

        # Log access
        await audit.log_access(
            user_id="user-123",
            endpoint="/triage",
            action="invoke",
            outcome="success",
        )

        # Log security event
        await audit.log_security(
            event_type=AuditEventType.PROMPT_INJECTION,
            user_id="user-123",
            severity=AuditSeverity.WARNING,
            reason="Potential prompt injection detected",
        )

        # Query events
        events = await audit.query(
            event_types=[AuditEventType.ACCESS_DENIED],
            user_id="user-123",
        )
    """

    def __init__(
        self,
        store: AuditStore | None = None,
        *,
        emit_to_logging: bool = True,
    ) -> None:
        """Initialize audit logger.

        Args:
            store: Storage backend for audit events
            emit_to_logging: Also emit to Python logging
        """
        self._store = store or InMemoryAuditStore()
        self._emit_to_logging = emit_to_logging

    async def log(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: The audit event to log
        """
        await self._store.write(event)

        if self._emit_to_logging:
            level = {
                AuditSeverity.DEBUG: logging.DEBUG,
                AuditSeverity.INFO: logging.INFO,
                AuditSeverity.WARNING: logging.WARNING,
                AuditSeverity.ERROR: logging.ERROR,
                AuditSeverity.CRITICAL: logging.CRITICAL,
            }.get(event.severity, logging.INFO)

            logger.log(
                level,
                f"AUDIT: [{event.event_type.value}] user={event.user_id} outcome={event.outcome}",
            )

    async def log_access(
        self,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        outcome: str,
        reason: str | None = None,
        run_id: str | None = None,
        ip_address: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an access event.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            endpoint: Endpoint accessed
            action: Action performed
            resource: Resource accessed
            outcome: "success" or "denied"
            reason: Reason for the outcome
            run_id: Request/run ID
            ip_address: Client IP
            metadata: Additional context
        """
        event_type = (
            AuditEventType.ACCESS_GRANTED if outcome == "success" else AuditEventType.ACCESS_DENIED
        )
        severity = AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            action=action,
            resource=resource,
            outcome=outcome,
            reason=reason,
            ip_address=ip_address,
            metadata=metadata or {},
        )

        await self.log(event)

    async def log_auth(
        self,
        *,
        event_type: AuditEventType,
        user_id: str | None = None,
        outcome: str,
        reason: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an authentication event.

        Args:
            event_type: AUTH_SUCCESS, AUTH_FAILURE, etc.
            user_id: User identifier
            outcome: Result of auth attempt
            reason: Reason for failure (if any)
            ip_address: Client IP
            user_agent: Client user agent
            metadata: Additional context
        """
        severity = (
            AuditSeverity.INFO
            if event_type == AuditEventType.AUTH_SUCCESS
            else AuditSeverity.WARNING
        )

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            outcome=outcome,
            reason=reason,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        await self.log(event)

    async def log_security(
        self,
        *,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.WARNING,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        reason: str | None = None,
        run_id: str | None = None,
        ip_address: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a security event.

        Args:
            event_type: SECURITY_ALERT, PROMPT_INJECTION, etc.
            severity: Event severity
            user_id: User identifier
            tenant_id: Tenant identifier
            endpoint: Endpoint involved
            reason: Description of security event
            run_id: Request/run ID
            ip_address: Client IP
            metadata: Additional context
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            reason=reason,
            ip_address=ip_address,
            metadata=metadata or {},
        )

        await self.log(event)

    async def log_tool(
        self,
        *,
        event_type: AuditEventType,
        tool_name: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        run_id: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a tool invocation event.

        Args:
            event_type: TOOL_INVOKE, TOOL_COMPLETE, etc.
            tool_name: Name of the tool
            user_id: User identifier
            tenant_id: Tenant identifier
            endpoint: Endpoint involved
            outcome: Result of tool call
            reason: Reason for failure (if any)
            run_id: Request/run ID
            duration_ms: Duration in milliseconds
            metadata: Additional context
        """
        severity = (
            AuditSeverity.ERROR if event_type == AuditEventType.TOOL_ERROR else AuditSeverity.INFO
        )

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            resource=f"tool:{tool_name}",
            outcome=outcome,
            reason=reason,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        await self.log(event)

    async def log_policy(
        self,
        *,
        event_type: AuditEventType,
        policy_name: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a policy evaluation event.

        Args:
            event_type: POLICY_EVALUATED, BUDGET_EXCEEDED, etc.
            policy_name: Name of the policy
            user_id: User identifier
            tenant_id: Tenant identifier
            endpoint: Endpoint involved
            action: Action being evaluated
            resource: Resource being accessed
            outcome: Result of evaluation
            reason: Reason for the outcome
            run_id: Request/run ID
            metadata: Additional context
        """
        severity = (
            AuditSeverity.WARNING
            if event_type in (AuditEventType.BUDGET_EXCEEDED, AuditEventType.RATE_LIMITED)
            else AuditSeverity.INFO
        )

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            action=action,
            resource=resource,
            outcome=outcome,
            reason=reason,
            metadata={**(metadata or {}), "policy": policy_name},
        )

        await self.log(event)

    async def query(
        self,
        event_types: list[AuditEventType] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        severity: AuditSeverity | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query audit events.

        Args:
            event_types: Filter by event types
            start_time: Filter by start time
            end_time: Filter by end time
            user_id: Filter by user
            tenant_id: Filter by tenant
            severity: Filter by severity
            limit: Maximum events to return

        Returns:
            List of matching AuditEvents
        """
        return await self._store.query(
            event_types=event_types,
            start_time=start_time.timestamp() if start_time else None,
            end_time=end_time.timestamp() if end_time else None,
            user_id=user_id,
            tenant_id=tenant_id,
            severity=severity,
            limit=limit,
        )
