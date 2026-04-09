"""Approval workflow system for FastAgentic."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """A request for human approval.

    Attributes:
        id: Unique request identifier
        run_id: Associated run ID
        action: Action requiring approval
        resource: Resource being acted upon
        description: Human-readable description
        requester_id: Who requested the action
        approvers: List of users who can approve
        data: Additional context data
        created_at: When the request was created
        expires_at: When the request expires
        status: Current status
        reviewed_by: Who reviewed the request
        reviewed_at: When it was reviewed
        review_notes: Notes from reviewer
    """

    action: str
    resource: str
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str | None = None
    requester_id: str | None = None
    approvers: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: float | None = None
    review_notes: str = ""

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        if self.status != ApprovalStatus.PENDING:
            return False
        return not (self.expires_at and time.time() > self.expires_at)

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at and time.time() > self.expires_at:
            return True
        return self.status == ApprovalStatus.EXPIRED


@dataclass
class ApprovalResponse:
    """Response to an approval request."""

    request_id: str
    status: ApprovalStatus
    reviewer_id: str | None = None
    notes: str = ""
    timestamp: float = field(default_factory=time.time)


class ApprovalStore(Protocol):
    """Protocol for approval request storage."""

    async def save(self, request: ApprovalRequest) -> None:
        """Save an approval request."""
        ...

    async def get(self, request_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID."""
        ...

    async def list_pending(
        self,
        approver_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """List pending requests."""
        ...

    async def update(self, request: ApprovalRequest) -> None:
        """Update an approval request."""
        ...


class InMemoryApprovalStore:
    """In-memory approval store for development/testing."""

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    async def save(self, request: ApprovalRequest) -> None:
        self._requests[request.id] = request

    async def get(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    async def list_pending(
        self,
        approver_id: str | None = None,
    ) -> list[ApprovalRequest]:
        results = []
        for request in self._requests.values():
            if not request.is_pending:
                continue
            if approver_id and approver_id not in request.approvers:
                continue
            results.append(request)
        return results

    async def update(self, request: ApprovalRequest) -> None:
        self._requests[request.id] = request


@dataclass
class ApprovalPolicy:
    """Policy defining when approval is required.

    Attributes:
        name: Policy name
        actions: Actions requiring approval (or "*" for all)
        resources: Resource patterns requiring approval
        condition: Optional condition function
        approvers: Default approvers for this policy
        timeout_seconds: How long until request expires
        auto_approve: Whether to auto-approve if no response
    """

    name: str
    actions: list[str] = field(default_factory=lambda: ["*"])
    resources: list[str] = field(default_factory=lambda: ["*"])
    condition: Callable[[str, str, dict[str, Any]], bool] | None = None
    approvers: list[str] = field(default_factory=list)
    timeout_seconds: int = 3600  # 1 hour
    auto_approve: bool = False

    def requires_approval(
        self,
        action: str,
        resource: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if this policy requires approval.

        Args:
            action: The action being performed
            resource: The resource being acted upon
            context: Additional context

        Returns:
            True if approval is required
        """
        # Check action match
        if "*" not in self.actions and action not in self.actions:
            return False

        # Check resource match
        resource_match = False
        for pattern in self.resources:
            if pattern == "*":
                resource_match = True
                break
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if resource.startswith(prefix):
                    resource_match = True
                    break
            elif pattern == resource:
                resource_match = True
                break

        if not resource_match:
            return False

        # Check custom condition
        if self.condition:
            return self.condition(action, resource, context or {})

        return True


# Type for approval callbacks
ApprovalCallback = Callable[[ApprovalRequest], Awaitable[None]]


class ApprovalManager:
    """Manager for approval workflows.

    Example:
        manager = ApprovalManager()

        # Add approval policy
        manager.add_policy(ApprovalPolicy(
            name="dangerous-actions",
            actions=["delete", "modify"],
            resources=["production/*"],
            approvers=["admin@example.com"],
            timeout_seconds=1800,
        ))

        # Request approval
        request = await manager.request_approval(
            action="delete",
            resource="production/database",
            description="Delete production database",
            requester_id="user-123",
        )

        # Wait for approval (or poll)
        response = await manager.wait_for_approval(request.id, timeout=300)

        if response.status == ApprovalStatus.APPROVED:
            # Proceed with action
            pass
    """

    def __init__(
        self,
        store: ApprovalStore | None = None,
    ) -> None:
        """Initialize approval manager.

        Args:
            store: Storage backend for approval requests
        """
        self._store = store or InMemoryApprovalStore()
        self._policies: list[ApprovalPolicy] = []
        self._on_request_callbacks: list[ApprovalCallback] = []
        self._on_response_callbacks: list[ApprovalCallback] = []
        self._pending_futures: dict[str, asyncio.Future[ApprovalResponse]] = {}

    def add_policy(self, policy: ApprovalPolicy) -> None:
        """Add an approval policy.

        Args:
            policy: The policy to add
        """
        self._policies.append(policy)

    def remove_policy(self, name: str) -> None:
        """Remove a policy by name.

        Args:
            name: Policy name to remove
        """
        self._policies = [p for p in self._policies if p.name != name]

    def on_request(self, callback: ApprovalCallback) -> None:
        """Register callback for new approval requests.

        Args:
            callback: Async function to call with request
        """
        self._on_request_callbacks.append(callback)

    def on_response(self, callback: ApprovalCallback) -> None:
        """Register callback for approval responses.

        Args:
            callback: Async function to call with request
        """
        self._on_response_callbacks.append(callback)

    def requires_approval(
        self,
        action: str,
        resource: str,
        context: dict[str, Any] | None = None,
    ) -> ApprovalPolicy | None:
        """Check if an action requires approval.

        Args:
            action: The action being performed
            resource: The resource being acted upon
            context: Additional context

        Returns:
            The matching policy or None
        """
        for policy in self._policies:
            if policy.requires_approval(action, resource, context):
                return policy
        return None

    async def request_approval(
        self,
        action: str,
        resource: str,
        description: str,
        *,
        run_id: str | None = None,
        requester_id: str | None = None,
        approvers: list[str] | None = None,
        data: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> ApprovalRequest:
        """Create an approval request.

        Args:
            action: Action requiring approval
            resource: Resource being acted upon
            description: Human-readable description
            run_id: Associated run ID
            requester_id: Who is requesting
            approvers: Who can approve (uses policy default if None)
            data: Additional context
            timeout_seconds: Custom timeout (uses policy default if None)

        Returns:
            The created approval request
        """
        # Find matching policy for defaults
        policy = self.requires_approval(action, resource, data)

        # Determine approvers
        if not approvers:
            approvers = policy.approvers if policy else []

        # Determine timeout
        if timeout_seconds is None:
            timeout_seconds = policy.timeout_seconds if policy else 3600

        expires_at = time.time() + timeout_seconds

        request = ApprovalRequest(
            action=action,
            resource=resource,
            description=description,
            run_id=run_id,
            requester_id=requester_id,
            approvers=approvers,
            data=data or {},
            expires_at=expires_at,
        )

        await self._store.save(request)

        # Notify callbacks
        for callback in self._on_request_callbacks:
            try:
                await callback(request)
            except Exception:
                # Don't fail on callback errors
                pass

        return request

    async def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID.

        Args:
            request_id: Request identifier

        Returns:
            The request or None
        """
        return await self._store.get(request_id)

    async def list_pending(
        self,
        approver_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """List pending approval requests.

        Args:
            approver_id: Filter by approver

        Returns:
            List of pending requests
        """
        return await self._store.list_pending(approver_id)

    async def approve(
        self,
        request_id: str,
        reviewer_id: str,
        notes: str = "",
    ) -> ApprovalResponse:
        """Approve a request.

        Args:
            request_id: Request to approve
            reviewer_id: Who is approving
            notes: Optional notes

        Returns:
            Approval response
        """
        return await self._respond(
            request_id,
            ApprovalStatus.APPROVED,
            reviewer_id,
            notes,
        )

    async def reject(
        self,
        request_id: str,
        reviewer_id: str,
        notes: str = "",
    ) -> ApprovalResponse:
        """Reject a request.

        Args:
            request_id: Request to reject
            reviewer_id: Who is rejecting
            notes: Optional notes

        Returns:
            Approval response
        """
        return await self._respond(
            request_id,
            ApprovalStatus.REJECTED,
            reviewer_id,
            notes,
        )

    async def cancel(
        self,
        request_id: str,
        notes: str = "",
    ) -> ApprovalResponse:
        """Cancel a request.

        Args:
            request_id: Request to cancel
            notes: Optional notes

        Returns:
            Approval response
        """
        return await self._respond(
            request_id,
            ApprovalStatus.CANCELLED,
            None,
            notes,
        )

    async def _respond(
        self,
        request_id: str,
        status: ApprovalStatus,
        reviewer_id: str | None,
        notes: str,
    ) -> ApprovalResponse:
        """Process a response to an approval request."""
        request = await self._store.get(request_id)
        if request is None:
            raise ValueError(f"Request not found: {request_id}")

        if not request.is_pending:
            raise ValueError(f"Request is not pending: {request_id}")

        # Update request
        request.status = status
        request.reviewed_by = reviewer_id
        request.reviewed_at = time.time()
        request.review_notes = notes

        await self._store.update(request)

        response = ApprovalResponse(
            request_id=request_id,
            status=status,
            reviewer_id=reviewer_id,
            notes=notes,
        )

        # Resolve any waiting futures
        if request_id in self._pending_futures:
            future = self._pending_futures.pop(request_id)
            if not future.done():
                future.set_result(response)

        # Notify callbacks
        for callback in self._on_response_callbacks:
            try:
                await callback(request)
            except Exception:
                pass

        return response

    async def wait_for_approval(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> ApprovalResponse:
        """Wait for an approval response.

        Args:
            request_id: Request to wait for
            timeout: Maximum seconds to wait

        Returns:
            Approval response

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            ValueError: If request not found
        """
        request = await self._store.get(request_id)
        if request is None:
            raise ValueError(f"Request not found: {request_id}")

        # If already resolved, return immediately
        if not request.is_pending:
            return ApprovalResponse(
                request_id=request_id,
                status=request.status,
                reviewer_id=request.reviewed_by,
                notes=request.review_notes,
            )

        # Create future for waiting
        if request_id not in self._pending_futures:
            self._pending_futures[request_id] = asyncio.get_event_loop().create_future()

        future = self._pending_futures[request_id]

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            # Check if auto-approve is enabled
            policy = self.requires_approval(
                request.action,
                request.resource,
                request.data,
            )
            if policy and policy.auto_approve:
                return await self.approve(
                    request_id,
                    reviewer_id="system:auto-approve",
                    notes="Auto-approved due to timeout",
                )

            # Mark as expired
            request.status = ApprovalStatus.EXPIRED
            await self._store.update(request)

            raise
