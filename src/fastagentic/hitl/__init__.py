"""Human-in-the-loop (HITL) actions for FastAgentic.

Provides approval workflows, confirmation dialogs,
escalation handling, and human review for sensitive actions.
"""

from fastagentic.hitl.approval import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    ApprovalPolicy,
    ApprovalManager,
)
from fastagentic.hitl.confirmation import (
    ConfirmationRequest,
    ConfirmationResponse,
    ConfirmationType,
    require_confirmation,
)
from fastagentic.hitl.escalation import (
    EscalationTrigger,
    EscalationLevel,
    EscalationHandler,
    EscalationManager,
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
