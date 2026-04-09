"""Tests for FastAgentic HITL module."""

import pytest

from fastagentic.hitl.approval import (
    ApprovalManager,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalStatus,
)
from fastagentic.hitl.confirmation import (
    ConfirmationRequest,
    ConfirmationType,
    QueuedConfirmationHandler,
    request_confirmation,
    set_confirmation_handler,
)
from fastagentic.hitl.escalation import (
    Escalation,
    EscalationHandler,
    EscalationLevel,
    EscalationManager,
    EscalationTrigger,
    EscalationTriggerType,
)


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_status_values(self):
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.EXPIRED.value == "expired"


class TestApprovalRequest:
    """Tests for ApprovalRequest class."""

    def test_basic_request(self):
        request = ApprovalRequest(
            action="delete",
            resource="database",
            description="Delete the database",
        )
        assert request.action == "delete"
        assert request.is_pending

    def test_request_with_expiry(self):
        import time

        request = ApprovalRequest(
            action="delete",
            resource="database",
            description="Delete the database",
            expires_at=time.time() - 1,  # Already expired
        )
        assert not request.is_pending
        assert request.is_expired


class TestApprovalPolicy:
    """Tests for ApprovalPolicy class."""

    def test_basic_policy(self):
        policy = ApprovalPolicy(
            name="dangerous",
            actions=["delete", "modify"],
            resources=["production/*"],
        )
        assert policy.requires_approval("delete", "production/db")
        assert not policy.requires_approval("read", "production/db")
        assert not policy.requires_approval("delete", "staging/db")

    def test_wildcard_policy(self):
        policy = ApprovalPolicy(
            name="all",
            actions=["*"],
            resources=["*"],
        )
        assert policy.requires_approval("anything", "anywhere")

    def test_custom_condition(self):
        policy = ApprovalPolicy(
            name="high-value",
            condition=lambda a, r, ctx: ctx.get("value", 0) > 1000,
        )
        assert policy.requires_approval("", "", {"value": 2000})
        assert not policy.requires_approval("", "", {"value": 500})


class TestApprovalManager:
    """Tests for ApprovalManager class."""

    @pytest.mark.asyncio
    async def test_request_approval(self):
        manager = ApprovalManager()
        manager.add_policy(
            ApprovalPolicy(
                name="test",
                actions=["delete"],
                approvers=["admin@example.com"],
            )
        )

        request = await manager.request_approval(
            action="delete",
            resource="database",
            description="Delete the database",
        )

        assert request.id is not None
        assert request.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_approve_request(self):
        manager = ApprovalManager()

        request = await manager.request_approval(
            action="delete",
            resource="database",
            description="Test",
        )

        response = await manager.approve(
            request.id,
            reviewer_id="admin@example.com",
            notes="Approved for cleanup",
        )

        assert response.status == ApprovalStatus.APPROVED

        updated = await manager.get_request(request.id)
        assert updated.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_reject_request(self):
        manager = ApprovalManager()

        request = await manager.request_approval(
            action="delete",
            resource="database",
            description="Test",
        )

        response = await manager.reject(
            request.id,
            reviewer_id="admin@example.com",
            notes="Not authorized",
        )

        assert response.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_list_pending(self):
        manager = ApprovalManager()

        await manager.request_approval(
            action="delete",
            resource="db1",
            description="Test 1",
            approvers=["admin@example.com"],
        )
        await manager.request_approval(
            action="delete",
            resource="db2",
            description="Test 2",
            approvers=["admin@example.com"],
        )

        pending = await manager.list_pending()
        assert len(pending) == 2

        pending_for_admin = await manager.list_pending(approver_id="admin@example.com")
        assert len(pending_for_admin) == 2

    @pytest.mark.asyncio
    async def test_requires_approval(self):
        manager = ApprovalManager()
        manager.add_policy(
            ApprovalPolicy(
                name="dangerous",
                actions=["delete"],
            )
        )

        policy = manager.requires_approval("delete", "anything")
        assert policy is not None
        assert policy.name == "dangerous"

        policy = manager.requires_approval("read", "anything")
        assert policy is None

    @pytest.mark.asyncio
    async def test_cancel_request(self):
        manager = ApprovalManager()

        request = await manager.request_approval(
            action="delete",
            resource="database",
            description="Test",
        )

        response = await manager.cancel(request.id, notes="No longer needed")
        assert response.status == ApprovalStatus.CANCELLED


class TestConfirmationType:
    """Tests for ConfirmationType enum."""

    def test_type_values(self):
        assert ConfirmationType.SIMPLE.value == "simple"
        assert ConfirmationType.DESTRUCTIVE.value == "destructive"


class TestConfirmationRequest:
    """Tests for ConfirmationRequest class."""

    def test_basic_request(self):
        request = ConfirmationRequest(message="Are you sure?")
        assert request.message == "Are you sure?"
        assert request.options == ["Yes", "No"]


class TestQueuedConfirmationHandler:
    """Tests for QueuedConfirmationHandler class."""

    def test_queue_confirmation(self):
        handler = QueuedConfirmationHandler()
        request = ConfirmationRequest(message="Test?")

        future = handler.handle(request)
        assert not future.done()

        pending = handler.get_pending()
        assert len(pending) == 1

    def test_respond_to_confirmation(self):
        handler = QueuedConfirmationHandler()
        request = ConfirmationRequest(message="Test?")

        future = handler.handle(request)

        result = handler.respond(request.id, confirmed=True)
        assert result

        assert future.done()
        response = future.result()
        assert response.confirmed


class TestRequestConfirmation:
    """Tests for request_confirmation function."""

    @pytest.mark.asyncio
    async def test_auto_confirm_no_handler(self):
        # Clear any existing handler
        set_confirmation_handler(None)

        response = await request_confirmation("Test?")
        assert response.confirmed


class TestEscalationLevel:
    """Tests for EscalationLevel enum."""

    def test_level_values(self):
        assert EscalationLevel.LOW.value == "low"
        assert EscalationLevel.CRITICAL.value == "critical"


class TestEscalationTrigger:
    """Tests for EscalationTrigger class."""

    def test_error_count_trigger(self):
        trigger = EscalationTrigger(
            name="errors",
            trigger_type=EscalationTriggerType.ERROR_COUNT,
            threshold=3,
        )
        assert not trigger.check({"error_count": 2})
        assert trigger.check({"error_count": 3})

    def test_confidence_trigger(self):
        trigger = EscalationTrigger(
            name="low-confidence",
            trigger_type=EscalationTriggerType.CONFIDENCE_LOW,
            threshold=0.5,
        )
        assert not trigger.check({"confidence": 0.8})
        assert trigger.check({"confidence": 0.3})

    def test_user_request_trigger(self):
        trigger = EscalationTrigger(
            name="user-request",
            trigger_type=EscalationTriggerType.USER_REQUEST,
        )
        assert not trigger.check({})
        assert trigger.check({"user_requested_escalation": True})

    def test_custom_trigger(self):
        trigger = EscalationTrigger(
            name="custom",
            trigger_type=EscalationTriggerType.CUSTOM,
            condition=lambda ctx: ctx.get("special", False),
        )
        assert not trigger.check({})
        assert trigger.check({"special": True})


class TestEscalationManager:
    """Tests for EscalationManager class."""

    @pytest.mark.asyncio
    async def test_add_trigger(self):
        manager = EscalationManager()
        manager.add_trigger(
            EscalationTrigger(
                name="errors",
                trigger_type=EscalationTriggerType.ERROR_COUNT,
                threshold=3,
            )
        )

        escalations = await manager.check_escalation(
            run_id="run-123",
            context={"error_count": 5},
        )

        assert len(escalations) == 1
        assert escalations[0].trigger.name == "errors"

    @pytest.mark.asyncio
    async def test_no_escalation(self):
        manager = EscalationManager()
        manager.add_trigger(
            EscalationTrigger(
                name="errors",
                trigger_type=EscalationTriggerType.ERROR_COUNT,
                threshold=3,
            )
        )

        escalations = await manager.check_escalation(
            run_id="run-123",
            context={"error_count": 1},
        )

        assert len(escalations) == 0

    @pytest.mark.asyncio
    async def test_manual_escalate(self):
        manager = EscalationManager()

        escalation = await manager.escalate(
            run_id="run-123",
            level=EscalationLevel.HIGH,
            reason="User requested help",
        )

        assert escalation.level == EscalationLevel.HIGH
        assert escalation.status == "pending"

    @pytest.mark.asyncio
    async def test_resolve_escalation(self):
        manager = EscalationManager()

        escalation = await manager.escalate(
            run_id="run-123",
            level=EscalationLevel.HIGH,
            reason="Test",
        )

        resolved = await manager.resolve(
            escalation.id,
            resolution="Issue fixed",
            notes="User was confused",
        )

        assert resolved.status == "resolved"
        assert resolved.resolution == "Issue fixed"

    @pytest.mark.asyncio
    async def test_list_pending(self):
        manager = EscalationManager()

        await manager.escalate(
            run_id="run-1",
            level=EscalationLevel.LOW,
            reason="Test 1",
        )
        await manager.escalate(
            run_id="run-2",
            level=EscalationLevel.HIGH,
            reason="Test 2",
        )

        pending = manager.list_pending()
        assert len(pending) == 2

        pending_high = manager.list_pending(level=EscalationLevel.HIGH)
        assert len(pending_high) == 1

    @pytest.mark.asyncio
    async def test_handler_callback(self):
        manager = EscalationManager()
        called = []

        async def callback(esc: Escalation) -> None:
            called.append(esc.id)

        manager.add_handler(
            EscalationHandler(
                name="test",
                levels=[EscalationLevel.HIGH],
                callback=callback,
            )
        )

        manager.add_trigger(
            EscalationTrigger(
                name="errors",
                trigger_type=EscalationTriggerType.ERROR_COUNT,
                threshold=3,
                level=EscalationLevel.HIGH,
            )
        )

        await manager.check_escalation(
            run_id="run-123",
            context={"error_count": 5},
        )

        assert len(called) == 1

    def test_get_stats(self):
        manager = EscalationManager()
        stats = manager.get_stats()

        assert stats["total"] == 0
        assert "by_level" in stats
        assert "by_status" in stats
