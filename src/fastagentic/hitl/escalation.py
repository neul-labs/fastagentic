"""Escalation handling for FastAgentic."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EscalationLevel(str, Enum):
    """Escalation severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationTriggerType(str, Enum):
    """Types of escalation triggers."""

    ERROR_COUNT = "error_count"  # Too many errors
    CONFIDENCE_LOW = "confidence_low"  # Model confidence below threshold
    SENTIMENT_NEGATIVE = "sentiment_negative"  # Negative user sentiment
    TOPIC_SENSITIVE = "topic_sensitive"  # Sensitive topic detected
    TIMEOUT = "timeout"  # Response timeout
    USER_REQUEST = "user_request"  # User explicitly requested
    POLICY_VIOLATION = "policy_violation"  # Policy check failed
    CUSTOM = "custom"  # Custom trigger


@dataclass
class EscalationTrigger:
    """Definition of what triggers an escalation.

    Attributes:
        name: Trigger name
        trigger_type: Type of trigger
        level: Escalation level when triggered
        threshold: Threshold value (meaning depends on type)
        condition: Optional custom condition function
        description: Human-readable description
    """

    name: str
    trigger_type: EscalationTriggerType
    level: EscalationLevel = EscalationLevel.MEDIUM
    threshold: float | int | None = None
    condition: Callable[[dict[str, Any]], bool] | None = None
    description: str = ""

    def check(self, context: dict[str, Any]) -> bool:
        """Check if this trigger should fire.

        Args:
            context: Context data to check against

        Returns:
            True if trigger should fire
        """
        if self.condition:
            return self.condition(context)

        # Built-in trigger logic
        if self.trigger_type == EscalationTriggerType.ERROR_COUNT:
            error_count = context.get("error_count", 0)
            return error_count >= (self.threshold or 3)  # type: ignore[no-any-return]

        elif self.trigger_type == EscalationTriggerType.CONFIDENCE_LOW:
            confidence = context.get("confidence", 1.0)
            return confidence < (self.threshold or 0.5)  # type: ignore[no-any-return]

        elif self.trigger_type == EscalationTriggerType.SENTIMENT_NEGATIVE:
            sentiment = context.get("sentiment", 0.0)
            return sentiment < (self.threshold or -0.5)  # type: ignore[no-any-return]

        elif self.trigger_type == EscalationTriggerType.TIMEOUT:
            elapsed = context.get("elapsed_seconds", 0)
            return elapsed > (self.threshold or 30)  # type: ignore[no-any-return]

        elif self.trigger_type == EscalationTriggerType.USER_REQUEST:
            return context.get("user_requested_escalation", False)  # type: ignore[no-any-return]

        elif self.trigger_type == EscalationTriggerType.POLICY_VIOLATION:
            return context.get("policy_violated", False)  # type: ignore[no-any-return]

        elif self.trigger_type == EscalationTriggerType.TOPIC_SENSITIVE:
            topics = context.get("detected_topics", [])
            sensitive_topics = context.get("sensitive_topics", [])
            return bool(set(topics) & set(sensitive_topics))  # type: ignore[no-any-return]

        return False


@dataclass
class Escalation:
    """An escalation event.

    Attributes:
        id: Unique identifier
        run_id: Associated run ID
        level: Escalation level
        trigger: What triggered the escalation
        context: Context when escalation occurred
        handler_id: Who should handle this
        status: Current status
        created_at: When created
        resolved_at: When resolved
        resolution: How it was resolved
        notes: Handler notes
    """

    run_id: str
    level: EscalationLevel
    trigger: EscalationTrigger
    id: str = field(default_factory=lambda: f"esc-{uuid.uuid4().hex[:12]}")
    context: dict[str, Any] = field(default_factory=dict)
    handler_id: str | None = None
    status: str = "pending"  # pending, assigned, resolved, timeout
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None
    resolution: str = ""
    notes: str = ""


# Handler callback type
EscalationCallback = Callable[[Escalation], Awaitable[None]]


@dataclass
class EscalationHandler:
    """Handler for escalations at a specific level.

    Attributes:
        name: Handler name
        levels: Levels this handler handles
        handler_ids: IDs of people/systems that can handle
        callback: Async callback when escalation occurs
        auto_assign: Whether to auto-assign to first available
    """

    name: str
    levels: list[EscalationLevel]
    handler_ids: list[str] = field(default_factory=list)
    callback: EscalationCallback | None = None
    auto_assign: bool = False


class EscalationManager:
    """Manager for escalation workflows.

    Example:
        manager = EscalationManager()

        # Add triggers
        manager.add_trigger(EscalationTrigger(
            name="too-many-errors",
            trigger_type=EscalationTriggerType.ERROR_COUNT,
            threshold=3,
            level=EscalationLevel.HIGH,
        ))

        manager.add_trigger(EscalationTrigger(
            name="low-confidence",
            trigger_type=EscalationTriggerType.CONFIDENCE_LOW,
            threshold=0.3,
            level=EscalationLevel.MEDIUM,
        ))

        # Add handlers
        manager.add_handler(EscalationHandler(
            name="support-team",
            levels=[EscalationLevel.LOW, EscalationLevel.MEDIUM],
            handler_ids=["support@example.com"],
            callback=notify_support,
        ))

        manager.add_handler(EscalationHandler(
            name="engineering",
            levels=[EscalationLevel.HIGH, EscalationLevel.CRITICAL],
            handler_ids=["engineering@example.com"],
            callback=page_engineering,
        ))

        # Check for escalation
        context = {"error_count": 5, "confidence": 0.2}
        escalations = await manager.check_escalation(
            run_id="run-123",
            context=context,
        )
    """

    def __init__(self) -> None:
        """Initialize escalation manager."""
        self._triggers: list[EscalationTrigger] = []
        self._handlers: list[EscalationHandler] = []
        self._escalations: dict[str, Escalation] = {}
        self._on_escalation_callbacks: list[EscalationCallback] = []

    def add_trigger(self, trigger: EscalationTrigger) -> None:
        """Add an escalation trigger.

        Args:
            trigger: The trigger to add
        """
        self._triggers.append(trigger)

    def remove_trigger(self, name: str) -> None:
        """Remove a trigger by name.

        Args:
            name: Trigger name
        """
        self._triggers = [t for t in self._triggers if t.name != name]

    def add_handler(self, handler: EscalationHandler) -> None:
        """Add an escalation handler.

        Args:
            handler: The handler to add
        """
        self._handlers.append(handler)

    def remove_handler(self, name: str) -> None:
        """Remove a handler by name.

        Args:
            name: Handler name
        """
        self._handlers = [h for h in self._handlers if h.name != name]

    def on_escalation(self, callback: EscalationCallback) -> None:
        """Register callback for all escalations.

        Args:
            callback: Async function to call
        """
        self._on_escalation_callbacks.append(callback)

    async def check_escalation(
        self,
        run_id: str,
        context: dict[str, Any],
    ) -> list[Escalation]:
        """Check if context triggers any escalations.

        Args:
            run_id: Associated run ID
            context: Context to check

        Returns:
            List of triggered escalations
        """
        escalations: list[Escalation] = []

        for trigger in self._triggers:
            if trigger.check(context):
                escalation = await self._create_escalation(
                    run_id=run_id,
                    trigger=trigger,
                    context=context,
                )
                escalations.append(escalation)

        return escalations

    async def _create_escalation(
        self,
        run_id: str,
        trigger: EscalationTrigger,
        context: dict[str, Any],
    ) -> Escalation:
        """Create and process an escalation."""
        escalation = Escalation(
            run_id=run_id,
            level=trigger.level,
            trigger=trigger,
            context=context,
        )

        # Find appropriate handler
        handler = self._find_handler(trigger.level)
        if handler:
            if handler.auto_assign and handler.handler_ids:
                escalation.handler_id = handler.handler_ids[0]
                escalation.status = "assigned"

            # Call handler callback
            if handler.callback:
                try:
                    await handler.callback(escalation)
                except Exception:
                    pass  # Don't fail on callback errors

        # Store escalation
        self._escalations[escalation.id] = escalation

        # Notify global callbacks
        for callback in self._on_escalation_callbacks:
            try:
                await callback(escalation)
            except Exception:
                pass

        return escalation

    def _find_handler(self, level: EscalationLevel) -> EscalationHandler | None:
        """Find handler for a level."""
        for handler in self._handlers:
            if level in handler.levels:
                return handler
        return None

    async def escalate(
        self,
        run_id: str,
        level: EscalationLevel,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> Escalation:
        """Manually escalate.

        Args:
            run_id: Associated run ID
            level: Escalation level
            reason: Reason for escalation
            context: Additional context

        Returns:
            Created escalation
        """
        trigger = EscalationTrigger(
            name="manual",
            trigger_type=EscalationTriggerType.CUSTOM,
            level=level,
            description=reason,
        )

        return await self._create_escalation(
            run_id=run_id,
            trigger=trigger,
            context=context or {},
        )

    async def resolve(
        self,
        escalation_id: str,
        resolution: str,
        notes: str = "",
    ) -> Escalation | None:
        """Resolve an escalation.

        Args:
            escalation_id: Escalation to resolve
            resolution: How it was resolved
            notes: Additional notes

        Returns:
            Updated escalation or None
        """
        escalation = self._escalations.get(escalation_id)
        if escalation is None:
            return None

        escalation.status = "resolved"
        escalation.resolved_at = time.time()
        escalation.resolution = resolution
        escalation.notes = notes

        return escalation

    def get_escalation(self, escalation_id: str) -> Escalation | None:
        """Get an escalation by ID.

        Args:
            escalation_id: Escalation ID

        Returns:
            The escalation or None
        """
        return self._escalations.get(escalation_id)

    def list_pending(
        self,
        level: EscalationLevel | None = None,
        handler_id: str | None = None,
    ) -> list[Escalation]:
        """List pending escalations.

        Args:
            level: Filter by level
            handler_id: Filter by handler

        Returns:
            List of pending escalations
        """
        results = []
        for escalation in self._escalations.values():
            if escalation.status in ("resolved", "timeout"):
                continue
            if level and escalation.level != level:
                continue
            if handler_id and escalation.handler_id != handler_id:
                continue
            results.append(escalation)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get escalation statistics.

        Returns:
            Stats dictionary
        """
        total = len(self._escalations)
        by_level: dict[str, int] = {}
        by_status: dict[str, int] = {}
        avg_resolution_time = 0.0
        resolved_count = 0

        for escalation in self._escalations.values():
            # By level
            level = escalation.level.value
            by_level[level] = by_level.get(level, 0) + 1

            # By status
            status = escalation.status
            by_status[status] = by_status.get(status, 0) + 1

            # Resolution time
            if escalation.resolved_at:
                resolved_count += 1
                resolution_time = escalation.resolved_at - escalation.created_at
                avg_resolution_time += resolution_time

        if resolved_count > 0:
            avg_resolution_time /= resolved_count

        return {
            "total": total,
            "by_level": by_level,
            "by_status": by_status,
            "avg_resolution_time_seconds": avg_resolution_time,
        }
