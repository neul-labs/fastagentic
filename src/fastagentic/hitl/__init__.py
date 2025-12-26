"""Human-in-the-loop (HITL) actions for FastAgentic.

Provides approval workflows, confirmation dialogs,
escalation handling, and human review for sensitive actions.
"""

from fastagentic.hitl.approval import (
    ApprovalManager,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from fastagentic.hitl.confirmation import (
    ConfirmationRequest,
    ConfirmationResponse,
    ConfirmationType,
    require_confirmation,
)
from fastagentic.hitl.escalation import (
    EscalationHandler,
    EscalationLevel,
    EscalationManager,
    EscalationTrigger,
)

__all__ = [
    # Approval
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    "ApprovalPolicy",
    "ApprovalManager",
    # Confirmation
    "ConfirmationRequest",
    "ConfirmationResponse",
    "ConfirmationType",
    "require_confirmation",
    # Escalation
    "EscalationTrigger",
    "EscalationLevel",
    "EscalationHandler",
    "EscalationManager",
]
