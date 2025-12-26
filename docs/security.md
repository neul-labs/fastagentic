# Security Guide

This document outlines security best practices for deploying and operating FastAgentic applications.

## Overview

FastAgentic provides multiple security layers:

- **Authentication** - OIDC/JWT integration
- **Authorization** - RBAC with scopes and permissions
- **Input Validation** - Pydantic models for all inputs
- **PII Detection** - Automatic detection and masking
- **Audit Logging** - Comprehensive activity logging
- **Rate Limiting** - Protection against abuse

## Authentication

### OIDC Integration

FastAgentic supports OpenID Connect for authentication:

```python
from fastagentic import App

app = App(
    title="Secure Agent",
    oidc_issuer="https://auth.example.com",
    oidc_audience="your-api-audience",
)
```

### JWT Validation

All incoming requests are validated against the OIDC provider:

- Token signature verification
- Expiration checking
- Audience validation
- Issuer validation

### API Keys

For service-to-service communication:

```python
app = App(
    api_keys=["sk-prod-key-1", "sk-prod-key-2"],
)
```

**Best Practices:**
- Rotate API keys regularly
- Use different keys per environment
- Never commit keys to version control
- Use environment variables or secret managers

## Authorization

### Role-Based Access Control (RBAC)

Define roles and permissions:

```python
from fastagentic import Role, Permission, RBACPolicy

admin_role = Role(
    name="admin",
    permissions=[
        Permission(resource="*", actions=["*"]),
    ],
)

user_role = Role(
    name="user",
    permissions=[
        Permission(resource="chat", actions=["read", "write"]),
    ],
)

policy = RBACPolicy(roles=[admin_role, user_role])
```

### Scope-Based Authorization

Limit access using OAuth scopes:

```python
from fastagentic import ScopePolicy, Scope

policy = ScopePolicy(
    required_scopes=[
        Scope("chat:read"),
        Scope("chat:write"),
    ],
)
```

### Endpoint Protection

```python
@agent_endpoint(
    path="/admin/config",
    scopes=["admin:config"],
    roles=["admin"],
)
async def admin_config():
    pass
```

## Input Validation

### Pydantic Models

All inputs are validated using Pydantic:

```python
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)

    @validator("message")
    def sanitize_message(cls, v):
        # Remove potentially dangerous content
        return v.strip()
```

### Request Size Limits

Configure maximum request sizes:

```python
app = App(
    max_request_size=1_000_000,  # 1MB
)
```

## PII Detection and Protection

### Automatic Detection

FastAgentic can detect PII in inputs and outputs:

```python
from fastagentic import PIIDetector, PIIType

detector = PIIDetector(
    types=[
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.SSN,
        PIIType.CREDIT_CARD,
    ],
)
```

### Automatic Masking

```python
from fastagentic import PIIMasker, PIIConfig

config = PIIConfig(
    detect_types=[PIIType.EMAIL, PIIType.PHONE],
    mask_character="*",
    log_detections=True,
)

app = App(
    pii_config=config,
)
```

### Compliance Hooks

```python
from fastagentic import PIIDetectionHook, PIIMaskingHook

app.add_hook(PIIDetectionHook(config))
app.add_hook(PIIMaskingHook(config))
```

## Rate Limiting

### Configure Limits

```python
from fastagentic import RateLimit

app = App(
    rate_limit=RateLimit(
        rpm=60,          # Requests per minute
        tpm=100000,      # Tokens per minute
        by="user",       # Rate limit by user
    ),
)
```

### Budget Controls

```python
from fastagentic import BudgetPolicy, Budget, BudgetPeriod

policy = BudgetPolicy(
    budgets=[
        Budget(
            name="daily",
            limit_usd=100.0,
            period=BudgetPeriod.DAILY,
        ),
    ],
)
```

## Audit Logging

### Enable Comprehensive Logging

```python
from fastagentic import AuditLogger, AuditEventType

logger = AuditLogger(
    events=[
        AuditEventType.AUTH_SUCCESS,
        AuditEventType.AUTH_FAILURE,
        AuditEventType.TOOL_CALL,
        AuditEventType.PII_DETECTED,
    ],
    destination="file://logs/audit.jsonl",
)

app = App(audit_logger=logger)
```

### Log Format

Audit logs include:
- Timestamp
- Event type
- User ID
- Request ID
- Action details
- Outcome

## Secure Deployment

### Environment Variables

Store sensitive configuration in environment variables:

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Authentication
OIDC_ISSUER=https://auth.example.com
OIDC_AUDIENCE=your-api

# Database
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://...
```

### Secrets Management

Use a secrets manager in production:

```python
import os
from fastagentic import App

# Load from environment (injected by secrets manager)
app = App(
    oidc_issuer=os.environ["OIDC_ISSUER"],
    durable_store=os.environ["REDIS_URL"],
)
```

### HTTPS

Always use HTTPS in production:

```bash
# Run with TLS
fastagentic run --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Network Security

- Use private networks for internal services
- Implement network policies in Kubernetes
- Use service mesh for mTLS

## Security Checklist

### Development
- [ ] No hardcoded secrets in code
- [ ] Input validation on all endpoints
- [ ] Type hints and Pydantic models
- [ ] Security tests in CI

### Staging
- [ ] OIDC authentication enabled
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] PII detection enabled

### Production
- [ ] HTTPS only
- [ ] Secrets in secret manager
- [ ] Network policies configured
- [ ] Monitoring and alerting
- [ ] Incident response plan
- [ ] Regular security audits

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** open a public issue
2. Email security@fastagentic.dev
3. Include detailed reproduction steps
4. Allow time for a fix before disclosure

## Security Updates

- Subscribe to security advisories
- Keep dependencies updated
- Monitor CVE databases
- Apply patches promptly

## Additional Resources

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [FastAgentic Policy Documentation](policy.md)
- [FastAgentic Compliance Documentation](compliance.md)
