# Human-in-the-Loop (HITL)

FastAgentic provides built-in support for human oversight, approval workflows, and escalation handling.

## Overview

- **Approval Manager** - Request and manage human approvals
- **Confirmation Dialogs** - Require user confirmation for sensitive actions
- **Escalation Manager** - Automatic escalation based on triggers

## Quick Start

```python
from fastagentic import (
    ApprovalManager,
    ApprovalPolicy,
    ApprovalStatus,
    EscalationManager,
    EscalationTrigger,
    EscalationLevel,
    require_confirmation,
)

# Create approval manager
approvals = ApprovalManager()

# Add policies
approvals.add_policy(ApprovalPolicy(
    name="high_value",
    condition=lambda ctx: ctx.get("amount", 0) > 10000,
    approvers=["finance-team"],
))

# Request approval
request = await approvals.request_approval(
    action="transfer_funds",
    resource="bank-account-123",
    description="Transfer $15,000 to vendor",
    context={"amount": 15000},
    requester_id="user-123",
)

# Wait for approval
result = await approvals.wait_for_approval(request.id, timeout=3600)
if result.status == ApprovalStatus.APPROVED:
    # Proceed with action
    pass
```

## Approval Manager

### Creating Policies

```python
from fastagentic import ApprovalPolicy

# Simple policy - always requires approval
policy = ApprovalPolicy(
    name="all_actions",
    approvers=["admin-team"],
)

# Conditional policy
policy = ApprovalPolicy(
    name="high_risk",
    condition=lambda ctx: ctx.get("risk_score", 0) > 0.8,
    approvers=["security-team", "manager"],
    require_all=False,  # Any one approver is sufficient
)

# Time-based policy
policy = ApprovalPolicy(
    name="after_hours",
    condition=lambda ctx: not is_business_hours(),
    approvers=["on-call-team"],
    auto_expire_hours=4,
)
```

### Approval Workflow

```python
# Request approval
request = await approvals.request_approval(
    action="delete_data",
    resource="customer-records",
    description="Bulk delete inactive customer records",
    context={"record_count": 5000},
    requester_id="user-123",
)

print(f"Request ID: {request.id}")
print(f"Status: {request.status}")  # PENDING

# Approve (by an approver)
await approvals.approve(
    request_id=request.id,
    reviewer_id="admin-456",
    notes="Verified backup exists",
)

# Or reject
await approvals.reject(
    request_id=request.id,
    reviewer_id="admin-456",
    reason="Need to verify backup first",
)

# Check status
request = await approvals.get_request(request.id)
print(f"Status: {request.status}")  # APPROVED or REJECTED
```

### Waiting for Approval

```python
# Blocking wait with timeout
try:
    result = await approvals.wait_for_approval(
        request_id=request.id,
        timeout=3600,  # 1 hour
    )
    if result.status == ApprovalStatus.APPROVED:
        print("Approved! Proceeding...")
    elif result.status == ApprovalStatus.REJECTED:
        print(f"Rejected: {result.rejection_reason}")
except TimeoutError:
    print("Approval timed out")
    await approvals.cancel(request.id)
```

### Approval Status

```python
class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
```

## Confirmation Decorator

Require confirmation for sensitive operations:

```python
from fastagentic import require_confirmation

@require_confirmation(
    message="This will delete all user data. Are you sure?",
    timeout=60,
)
async def delete_all_data(user_id: str):
    # Only executes after confirmation
    await db.delete_user_data(user_id)
```

### Confirmation Options

```python
@require_confirmation(
    message="Deploy to production?",
    confirm_value="DEPLOY",  # User must type exact value
    timeout=120,
    on_cancel=lambda: print("Deployment cancelled"),
)
async def deploy_production():
    ...
```

## Escalation Manager

Automatic escalation based on conditions:

```python
from fastagentic import (
    EscalationManager,
    EscalationTrigger,
    EscalationTriggerType,
    EscalationLevel,
)

escalation = EscalationManager()

# Add triggers
escalation.add_trigger(EscalationTrigger(
    name="low_confidence",
    trigger_type=EscalationTriggerType.CONFIDENCE_LOW,
    threshold=0.5,
    level=EscalationLevel.HUMAN_REVIEW,
))

escalation.add_trigger(EscalationTrigger(
    name="error_count",
    trigger_type=EscalationTriggerType.ERROR_COUNT,
    threshold=3,
    level=EscalationLevel.SUPERVISOR,
))

escalation.add_trigger(EscalationTrigger(
    name="cost_limit",
    trigger_type=EscalationTriggerType.COST_EXCEEDED,
    threshold=100.0,
    level=EscalationLevel.MANAGER,
))
```

### Escalation Levels

```python
class EscalationLevel(str, Enum):
    HUMAN_REVIEW = "human_review"  # Basic human oversight
    SUPERVISOR = "supervisor"       # Team lead review
    MANAGER = "manager"            # Management approval
    EXECUTIVE = "executive"        # Executive decision
    EMERGENCY = "emergency"        # Immediate action required
```

### Trigger Types

```python
class EscalationTriggerType(str, Enum):
    CONFIDENCE_LOW = "confidence_low"      # Model confidence below threshold
    ERROR_COUNT = "error_count"            # Too many errors
    COST_EXCEEDED = "cost_exceeded"        # Cost limit reached
    TIME_EXCEEDED = "time_exceeded"        # Task taking too long
    SENSITIVE_DATA = "sensitive_data"      # PII or sensitive content detected
    POLICY_VIOLATION = "policy_violation"  # Policy rule triggered
    CUSTOM = "custom"                      # Custom condition
```

### Checking Escalation

```python
# During agent execution
context = {
    "confidence": 0.4,
    "error_count": 2,
    "cost": 50.0,
}

escalation_result = await escalation.check_escalation(
    run_id="run-123",
    context=context,
)

if escalation_result.should_escalate:
    print(f"Escalating to: {escalation_result.level}")
    print(f"Reason: {escalation_result.reason}")

    # Create escalation
    esc = await escalation.escalate(
        run_id="run-123",
        level=escalation_result.level,
        reason=escalation_result.reason,
    )

    # Wait for resolution
    resolution = await escalation.wait_for_resolution(
        escalation_id=esc.id,
        timeout=3600,
    )
```

### Resolving Escalations

```python
# Resolve an escalation
await escalation.resolve(
    escalation_id=esc.id,
    resolution="Reviewed and approved",
    resolved_by="manager-789",
    action="continue",  # continue, abort, modify
)

# Get escalation history
history = await escalation.get_history(run_id="run-123")
for e in history:
    print(f"{e.level}: {e.reason} -> {e.resolution}")
```

## Integration with App

```python
from fastagentic import App, agent_endpoint
from fastagentic.hitl import ApprovalManager, EscalationManager

app = App(title="My Agent")

# Shared managers
approvals = ApprovalManager()
escalation = EscalationManager()

@agent_endpoint(path="/process")
async def process(data: dict):
    # Check if approval needed
    if data.get("amount", 0) > 10000:
        request = await approvals.request_approval(
            action="high_value_process",
            resource="transaction",
            context=data,
        )
        result = await approvals.wait_for_approval(request.id)
        if result.status != ApprovalStatus.APPROVED:
            raise ValueError("Approval denied")

    # Process with escalation checking
    result = await agent.run(data)

    if result.confidence < 0.5:
        await escalation.escalate(
            run_id=result.run_id,
            level=EscalationLevel.HUMAN_REVIEW,
            reason="Low confidence result",
        )

    return result
```

## Best Practices

1. **Set reasonable timeouts** - Don't block indefinitely; have fallback behavior
2. **Clear descriptions** - Provide context so approvers can make informed decisions
3. **Multiple approvers** - Use `require_all=False` for availability
4. **Audit trail** - All approvals and escalations are logged
5. **Escalation levels** - Match severity to appropriate level
6. **Auto-expire** - Set expiration for pending approvals
7. **Notification integration** - Connect to Slack/Email for approver notifications
