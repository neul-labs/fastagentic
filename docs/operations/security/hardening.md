# Security Hardening Guide

Production security configuration for FastAgentic deployments.

## Authentication

### Enforce OIDC Authentication

```bash
# Required in production
export FASTAGENTIC_AUTH_REQUIRED=true
export FASTAGENTIC_OIDC_ISSUER=https://auth.company.com
export FASTAGENTIC_OIDC_AUDIENCE=my-agent
```

### Token Validation

```yaml
# config/settings.yaml
auth:
  oidc:
    issuer: https://auth.company.com
    audience: my-agent
    algorithms: [RS256, RS384]
    clock_skew: 30  # seconds
    required_claims:
      - sub
      - exp
      - iat
```

### Session Management

```bash
# Session timeout (seconds)
export FASTAGENTIC_SESSION_TIMEOUT=3600

# Require re-authentication for sensitive operations
export FASTAGENTIC_STEP_UP_AUTH_ENABLED=true
```

## Network Security

### TLS Configuration

```bash
# Enforce TLS
export FASTAGENTIC_TLS_CERT=/etc/ssl/certs/server.crt
export FASTAGENTIC_TLS_KEY=/etc/ssl/private/server.key
export FASTAGENTIC_TLS_MIN_VERSION=1.2
export FASTAGENTIC_TLS_CIPHERS="ECDHE+AESGCM:DHE+AESGCM"
```

### CORS Configuration

```bash
# Restrict origins in production
export FASTAGENTIC_CORS_ORIGINS=https://app.company.com,https://admin.company.com
export FASTAGENTIC_CORS_METHODS=GET,POST,OPTIONS
export FASTAGENTIC_CORS_HEADERS=Authorization,Content-Type
export FASTAGENTIC_CORS_CREDENTIALS=true
export FASTAGENTIC_CORS_MAX_AGE=3600
```

### Trusted Proxies

```bash
# Only trust specific proxy IPs
export FASTAGENTIC_TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12
```

### Rate Limiting

```bash
# Global rate limits
export FASTAGENTIC_RATE_LIMIT_REQUESTS=100
export FASTAGENTIC_RATE_LIMIT_BURST=20

# Per-endpoint limits in config
```

```yaml
# config/settings.yaml
rate_limits:
  default: 100/minute
  endpoints:
    /expensive: 10/minute
    /chat: 60/minute
```

## Input Validation

### Request Size Limits

```bash
export FASTAGENTIC_MAX_REQUEST_SIZE=10485760  # 10MB
export FASTAGENTIC_MAX_INPUT_LENGTH=100000    # characters
```

### Content Validation

```yaml
# config/settings.yaml
validation:
  max_input_length: 100000
  max_output_length: 500000
  blocked_patterns:
    - "(?i)password\\s*[:=]"
    - "(?i)api[_-]?key\\s*[:=]"
  sanitize_html: true
```

## Secrets Management

### Environment Variables

**Never commit secrets to code.** Use:

```bash
# Kubernetes secrets
kubectl create secret generic agent-secrets \
  --from-literal=openai-key=sk-... \
  --from-literal=db-password=...

# Reference in deployment
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: agent-secrets
        key: openai-key
```

### External Secrets

```yaml
# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agent-secrets
spec:
  secretStoreRef:
    name: vault
    kind: ClusterSecretStore
  target:
    name: agent-secrets
  data:
    - secretKey: openai-api-key
      remoteRef:
        key: fastagentic/openai
        property: api_key
```

## Container Security

### Non-Root User

```dockerfile
# Run as non-root
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
```

### Read-Only Filesystem

```yaml
# Kubernetes
securityContext:
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

### Resource Limits

```yaml
resources:
  limits:
    cpu: "2"
    memory: 2Gi
  requests:
    cpu: "500m"
    memory: 512Mi
```

## Logging Security

### PII Redaction

```bash
export FASTAGENTIC_PII_REDACTION=true
```

Redacted patterns:
- Email addresses
- Phone numbers
- Credit card numbers
- Social security numbers
- IP addresses (optional)

### Log Sanitization

```yaml
# config/settings.yaml
logging:
  pii_redaction: true
  redact_fields:
    - user_input
    - api_keys
    - passwords
  redact_patterns:
    - "sk-[a-zA-Z0-9]+"
    - "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
```

## LLM Security

### Prompt Injection Protection

```yaml
# config/settings.yaml
llm_security:
  input_filtering: true
  output_filtering: true
  blocked_instructions:
    - "ignore previous instructions"
    - "disregard above"
    - "system prompt"
```

### Output Validation

```python
@agent_endpoint(
    path="/chat",
    output_validators=[
        no_pii_validator,
        no_harmful_content_validator,
    ],
)
async def chat(message: str) -> str:
    pass
```

### Model Access Control

```yaml
# config/settings.yaml
llm_access:
  allowed_models:
    - openai:gpt-4
    - openai:gpt-4o
    - anthropic:claude-3-opus
  denied_models:
    - "*:*-preview"  # No preview models
```

## Dependency Security

### Vulnerability Scanning

```bash
# Scan Python dependencies
pip-audit

# Scan container
docker scan myimage:latest

# In CI/CD
trivy image myimage:latest
```

### Dependency Pinning

```txt
# requirements.txt - pin versions
fastagentic==0.2.0
pydantic==2.5.0
openai==1.0.0
```

## Security Headers

FastAgentic sets secure headers by default:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

## Audit Trail

```bash
# Enable comprehensive audit logging
export FASTAGENTIC_AUDIT_ENABLED=true
export FASTAGENTIC_AUDIT_DESTINATION=stdout
```

Logged events:
- Authentication attempts
- Authorization decisions
- Run lifecycle events
- Configuration changes
- Error conditions

## Security Checklist

### Before Production

- [ ] OIDC authentication configured
- [ ] TLS enabled with modern ciphers
- [ ] CORS restricted to known origins
- [ ] Rate limiting configured
- [ ] Secrets in secret manager (not env vars)
- [ ] Container runs as non-root
- [ ] Resource limits set
- [ ] PII redaction enabled
- [ ] Audit logging enabled
- [ ] Dependencies scanned

### Ongoing

- [ ] Regular dependency updates
- [ ] Security patch monitoring
- [ ] Access review quarterly
- [ ] Penetration testing annually
- [ ] Incident response plan documented

## Next Steps

- [Compliance Guide](compliance.md) - SOC2, GDPR, HIPAA
- [Environment Variables](../configuration/environment-vars.md) - All settings
- [Alerting](../observability/alerting.md) - Security alerts
