# Compliance Guide

Configure FastAgentic to meet SOC2, GDPR, HIPAA, and other compliance requirements.

## Audit Logging

### Enable Audit Logs

```bash
export FASTAGENTIC_AUDIT_ENABLED=true
export FASTAGENTIC_AUDIT_DESTINATION=stdout  # or file:///var/log/audit.json
```

### Audit Log Format

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "event_type": "run.start",
  "event_id": "evt-abc123",
  "actor": {
    "user_id": "user-123",
    "tenant_id": "tenant-456",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0..."
  },
  "resource": {
    "type": "agent_endpoint",
    "path": "/chat",
    "run_id": "run-789"
  },
  "context": {
    "scopes": ["chat:run"],
    "model": "gpt-4",
    "estimated_cost": 0.02
  },
  "outcome": {
    "status": "success",
    "duration_ms": 1500
  },
  "trace_id": "trace-xyz"
}
```

### Event Types

| Event | Description | Logged Fields |
|-------|-------------|---------------|
| `auth.success` | Successful authentication | user, method, scopes |
| `auth.failure` | Failed authentication | ip, reason |
| `run.start` | Agent run initiated | user, endpoint, input_hash |
| `run.complete` | Agent run finished | run_id, duration, cost |
| `run.error` | Agent run failed | run_id, error |
| `policy.allow` | Policy allowed action | user, action, resource |
| `policy.deny` | Policy denied action | user, action, reason |
| `checkpoint.write` | Checkpoint saved | run_id, size |
| `config.change` | Configuration modified | user, change |

## SOC2 Compliance

### Control Mappings

| SOC2 Control | FastAgentic Feature | Configuration |
|--------------|---------------------|---------------|
| CC6.1 Logical Access | OIDC authentication | `FASTAGENTIC_OIDC_ISSUER` |
| CC6.2 Access Revocation | Scope-based auth | `scopes` parameter |
| CC6.3 Access Review | Audit logging | `FASTAGENTIC_AUDIT_ENABLED=true` |
| CC7.2 System Monitoring | OTEL telemetry | `FASTAGENTIC_OTEL_ENDPOINT` |
| CC7.3 Anomaly Detection | Cost alerts, rate limits | Policy configuration |
| CC8.1 Change Management | Contract testing | `fastagentic test contract` |

### Evidence Collection

```bash
# Export audit logs for review period
fastagentic audit export \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --format csv \
  --output q1-2024-audit.csv

# Generate access report
fastagentic audit report \
  --type access \
  --period Q1-2024 \
  --output access-report.pdf

# Generate cost allocation report
fastagentic audit report \
  --type cost \
  --period Q1-2024 \
  --by tenant \
  --output cost-report.pdf
```

## GDPR Compliance

### PII Handling

```bash
# Enable PII redaction in logs
export FASTAGENTIC_PII_REDACTION=true
```

Redacted fields:
- Email addresses → `[EMAIL]`
- Phone numbers → `[PHONE]`
- IP addresses → `[IP]`
- Names (configurable) → `[NAME]`

### Data Subject Rights

#### Right to Access (Article 15)

```bash
# Export user's data
fastagentic data export \
  --user user-123 \
  --format json \
  --output user-123-data.json
```

#### Right to Erasure (Article 17)

```bash
# Delete user's data
fastagentic data delete \
  --user user-123 \
  --confirm
```

#### Right to Portability (Article 20)

```bash
# Export in portable format
fastagentic data export \
  --user user-123 \
  --format portable \
  --output user-123-portable.zip
```

### Data Processing Records (Article 30)

Template for processing records:

| Field | Value |
|-------|-------|
| Controller | Your Company |
| Purpose | AI agent services |
| Data Categories | Queries, responses, usage metadata |
| Recipients | LLM providers (list specific) |
| Transfers | US (LLM providers) |
| Retention | 30 days (configurable) |
| Security | Encryption in transit/rest, access control |

### Retention Configuration

```yaml
# config/settings.yaml
data_retention:
  runs: 30d
  checkpoints: 7d
  audit_logs: 365d
  pii_fields:
    - user_input
    - agent_output
```

## HIPAA Compliance

### Required Configurations

```bash
# Encryption
export FASTAGENTIC_TLS_REQUIRED=true

# Audit logging
export FASTAGENTIC_AUDIT_ENABLED=true
export FASTAGENTIC_AUDIT_PHI=true

# Access controls
export FASTAGENTIC_AUTH_REQUIRED=true
export FASTAGENTIC_SESSION_TIMEOUT=900  # 15 minutes
```

### PHI Handling

```yaml
# config/settings.yaml
hipaa:
  enabled: true
  phi_fields:
    - patient_name
    - medical_record_number
    - diagnosis
  encryption:
    at_rest: true
    algorithm: AES-256-GCM
```

### BAA Requirements

When using LLM providers with PHI:
1. Execute BAA with provider
2. Configure dedicated endpoints
3. Disable training data collection
4. Document in compliance records

## Encryption

### In Transit

```bash
# TLS configuration
export FASTAGENTIC_TLS_CERT=/path/to/cert.pem
export FASTAGENTIC_TLS_KEY=/path/to/key.pem
export FASTAGENTIC_TLS_MIN_VERSION=1.2
```

### At Rest

Durable store encryption:

```bash
# PostgreSQL with encryption
export FASTAGENTIC_DURABLE_STORE=postgresql://user:pass@host:5432/db?sslmode=require

# Redis with TLS
export FASTAGENTIC_DURABLE_STORE=rediss://user:pass@host:6379
```

## Access Control

### Role-Based Access

```yaml
# config/settings.yaml
authorization:
  role_mappings:
    admin:
      - "*:*"
    operator:
      - "runs:read"
      - "runs:resume"
      - "metrics:read"
    user:
      - "chat:run"
      - "runs:read:own"
```

### Scope Requirements

```python
@agent_endpoint(
    path="/sensitive",
    scopes=["sensitive:run", "phi:access"],  # Require both scopes
)
async def sensitive_agent(input: Input) -> Output:
    pass
```

## Compliance Reporting

### Generate Reports

```bash
# SOC2 evidence package
fastagentic compliance report --standard soc2 --period Q1-2024

# GDPR compliance summary
fastagentic compliance report --standard gdpr --period 2024

# HIPAA audit
fastagentic compliance report --standard hipaa --period Q1-2024
```

### Report Contents

- Access control configuration
- Audit log summary
- Data retention policies
- Encryption status
- Incident history
- Policy configurations

## Checklist

### SOC2

- [ ] OIDC authentication enabled
- [ ] Audit logging enabled
- [ ] OTEL telemetry configured
- [ ] Rate limits and quotas set
- [ ] Contract testing in CI/CD
- [ ] Access review process documented

### GDPR

- [ ] PII redaction enabled
- [ ] Data retention configured
- [ ] Data export capability tested
- [ ] Data deletion capability tested
- [ ] Processing records documented
- [ ] DPA with LLM providers

### HIPAA

- [ ] BAA executed with LLM providers
- [ ] TLS required
- [ ] PHI field encryption
- [ ] Audit logging for PHI access
- [ ] Session timeouts configured
- [ ] Access controls documented

## Next Steps

- [Security Hardening](hardening.md) - Security configuration
- [Environment Variables](../configuration/environment-vars.md) - All settings
- [Audit Logging](../observability/metrics.md) - Monitoring
