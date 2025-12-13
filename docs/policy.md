# Policy Engine

FastAgentic includes a comprehensive policy engine for authorization, access control, and budget enforcement.

## Overview

The policy engine provides:

- **RBAC (Role-Based Access Control)** - Assign roles with specific permissions
- **Scope-Based Access** - OAuth2-style scopes for fine-grained control
- **Budget Policies** - Enforce cost, token, and request limits
- **Policy Engine** - Combine multiple policies with priority ordering

## Quick Start

```python
from fastagentic import (
    PolicyEngine,
    RBACPolicy,
    ScopePolicy,
    BudgetPolicy,
    Role,
    Permission,
    Scope,
    Budget,
    BudgetPeriod,
)

# Create policies
rbac = RBACPolicy()
rbac.add_role(Role(
    name="analyst",
    permissions=[
        Permission(resource="reports", actions=["read", "create"]),
        Permission(resource="data", actions=["read"]),
    ],
))

scopes = ScopePolicy()
scopes.add_scope(Scope(name="tools:read", description="Read tool definitions"))
scopes.add_scope(Scope(name="tools:invoke", description="Invoke tools"))

budgets = BudgetPolicy()
budgets.set_user_budget("user-123", Budget(
    max_cost=10.0,
    max_requests=1000,
    period=BudgetPeriod.DAY,
))

# Combine in engine
engine = PolicyEngine()
engine.add_policy(rbac, priority=100)
engine.add_policy(scopes, priority=90)
engine.add_policy(budgets, priority=80)

# Evaluate
decision = await engine.evaluate(context)
if decision.allowed:
    # Proceed with request
    pass
```

## RBAC Policy

Role-Based Access Control assigns permissions through roles:

```python
from fastagentic import RBACPolicy, Role, Permission

policy = RBACPolicy()

# Define roles with permissions
admin_role = Role(
    name="admin",
    permissions=[
        Permission(resource="*", actions=["*"]),  # Full access
    ],
)

analyst_role = Role(
    name="analyst",
    permissions=[
        Permission(resource="reports", actions=["read", "create"]),
        Permission(resource="dashboards", actions=["read"]),
    ],
)

policy.add_role(admin_role)
policy.add_role(analyst_role)

# Assign roles to users
policy.assign_role("user-123", "analyst")
policy.assign_role("user-456", "admin")
```

### Permission Patterns

```python
# Specific resource and action
Permission(resource="reports", actions=["read"])

# Multiple actions
Permission(resource="reports", actions=["read", "create", "delete"])

# Wildcard actions
Permission(resource="reports", actions=["*"])

# Wildcard resource (admin)
Permission(resource="*", actions=["*"])

# With conditions
Permission(
    resource="reports",
    actions=["read"],
    conditions={"department": "sales"},
)
```

## Scope-Based Access

OAuth2-style scopes for API access control:

```python
from fastagentic import ScopePolicy, Scope

policy = ScopePolicy()

# Define scopes with hierarchy
policy.add_scope(Scope(
    name="tools:read",
    description="Read tool definitions",
))
policy.add_scope(Scope(
    name="tools:invoke",
    description="Invoke tools",
    implies=["tools:read"],  # Invoking implies reading
))
policy.add_scope(Scope(
    name="admin",
    description="Full admin access",
    implies=["tools:read", "tools:invoke", "resources:*"],
))

# Check if user has required scopes
has_access = policy.check_scopes(
    user_scopes=["tools:invoke"],
    required_scopes=["tools:read"],  # True - implied by invoke
)
```

### Common Scopes

```python
COMMON_SCOPES = {
    "tools:read": "Read tool definitions",
    "tools:invoke": "Invoke tools",
    "resources:read": "Read resources",
    "resources:write": "Write resources",
    "prompts:read": "Read prompts",
    "prompts:execute": "Execute prompts",
    "admin": "Full administrative access",
}
```

## Budget Policy

Enforce cost, token, and request limits:

```python
from fastagentic import BudgetPolicy, Budget, BudgetPeriod

policy = BudgetPolicy()

# Global budget for all users
policy.set_global_budget(Budget(
    max_cost=1000.0,
    max_tokens=10_000_000,
    max_requests=100_000,
    period=BudgetPeriod.MONTH,
    soft_limit_percent=80,  # Warn at 80%
))

# Per-user budgets
policy.set_user_budget("user-123", Budget(
    max_cost=50.0,
    max_requests=1000,
    period=BudgetPeriod.DAY,
))

# Per-tenant budgets
policy.set_tenant_budget("tenant-abc", Budget(
    max_cost=500.0,
    period=BudgetPeriod.WEEK,
))
```

### Budget Periods

```python
class BudgetPeriod(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
```

### Budget Enforcement

The budget policy tracks usage and enforces limits:

```python
# Record usage
await policy.record_usage(
    user_id="user-123",
    cost=0.05,
    tokens=1500,
    requests=1,
)

# Check remaining budget
remaining = await policy.get_remaining("user-123")
print(f"Cost remaining: ${remaining.cost}")
print(f"Tokens remaining: {remaining.tokens}")

# Budget exceeded returns denied policy result
result = await policy.evaluate(context)
if not result.allowed:
    print(f"Budget exceeded: {result.reason}")
```

## Policy Engine

Combine multiple policies with priority-based evaluation:

```python
from fastagentic import PolicyEngine, PolicyAction

engine = PolicyEngine()

# Add policies with priorities (higher = evaluated first)
engine.add_policy(rbac_policy, priority=100)
engine.add_policy(scope_policy, priority=90)
engine.add_policy(budget_policy, priority=80)

# Evaluate all policies
decision = await engine.evaluate(context)

print(f"Allowed: {decision.allowed}")
print(f"Action: {decision.action}")  # ALLOW, DENY, or REQUIRE_APPROVAL
print(f"Reasons: {decision.reasons}")
print(f"Metadata: {decision.metadata}")
```

### Policy Actions

```python
class PolicyAction(str, Enum):
    ALLOW = "allow"           # Request is allowed
    DENY = "deny"             # Request is denied
    REQUIRE_APPROVAL = "require_approval"  # Needs human approval
```

### Policy Context

Policies receive context about the request:

```python
from fastagentic import PolicyContext

context = PolicyContext(
    user_id="user-123",
    tenant_id="tenant-abc",
    resource="reports",
    action="create",
    scopes=["tools:invoke", "resources:write"],
    metadata={
        "ip_address": "192.168.1.1",
        "user_agent": "Mozilla/5.0...",
    },
)
```

## Cost Tracking

Track and analyze LLM costs:

```python
from fastagentic import CostTracker, ModelPricing

tracker = CostTracker()

# Record usage
await tracker.record(
    run_id="run-123",
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    cached_tokens=200,
)

# Get cost for a run
cost = await tracker.get_run_cost("run-123")
print(f"Run cost: ${cost.total_cost:.4f}")

# Aggregate by period
daily = await tracker.aggregate(
    period=AggregationPeriod.DAY,
    group_by="model",
)
```

### Model Pricing

Default pricing for common models:

```python
DEFAULT_PRICING = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.010,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
    ),
    "claude-3-5-sonnet-20241022": ModelPricing(
        model="claude-3-5-sonnet-20241022",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        cached_input_cost_per_1k=0.0003,
    ),
}
```

## Audit Logging

Log security and access events:

```python
from fastagentic import AuditLogger, AuditEventType, AuditSeverity

logger = AuditLogger()

# Log access events
await logger.log_access(
    user_id="user-123",
    endpoint="/api/reports",
    action="create",
    outcome="success",
)

# Log security events
await logger.log_security(
    event_type=AuditEventType.PROMPT_INJECTION,
    severity=AuditSeverity.HIGH,
    details={"input": "...", "detection_score": 0.95},
)

# Query audit logs
events = await logger.query(
    user_id="user-123",
    event_types=[AuditEventType.ACCESS_GRANTED, AuditEventType.ACCESS_DENIED],
    start_time=datetime.now() - timedelta(days=7),
)
```

### Event Types

```python
class AuditEventType(str, Enum):
    # Authentication
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"

    # Access Control
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"

    # Request Lifecycle
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    REQUEST_ERROR = "request_error"

    # Tool Usage
    TOOL_INVOKE = "tool_invoke"
    TOOL_SUCCESS = "tool_success"
    TOOL_FAILURE = "tool_failure"

    # Security
    SECURITY_ALERT = "security_alert"
    PROMPT_INJECTION = "prompt_injection"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"

    # And more...
```

## Integration with App

```python
from fastagentic import App, PolicyEngine, RBACPolicy

# Create policy engine
engine = PolicyEngine()
engine.add_policy(RBACPolicy())

# Add to app
app = App(
    title="My Agent",
    policy_engine=engine,
)

# Policies are automatically evaluated on each request
```

## Best Practices

1. **Layer policies** - Use RBAC for coarse access, scopes for API permissions, budgets for cost control
2. **Set soft limits** - Warn users before they hit hard budget limits
3. **Audit everything** - Log access decisions for compliance and debugging
4. **Use tenant isolation** - Separate budgets and policies per tenant
5. **Review regularly** - Audit role assignments and budget usage periodically
