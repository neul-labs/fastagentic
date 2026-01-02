# Secrets Management

Secure handling of API keys, credentials, and sensitive configuration.

## Principles

1. **Never commit secrets** to version control
2. **Use secret managers** in production
3. **Rotate regularly** and audit access
4. **Least privilege** - only grant what's needed

## Development

### .env File

For local development only:

```bash
# .env (add to .gitignore!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FASTAGENTIC_DURABLE_STORE=redis://localhost:6379
```

Load automatically:
```bash
fastagentic run  # Loads .env by default
```

### .env.example

Commit a template without values:

```bash
# .env.example
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
FASTAGENTIC_DURABLE_STORE=redis://localhost:6379
```

## Kubernetes Secrets

### Create Secrets

```bash
# From literal values
kubectl create secret generic agent-secrets \
  --from-literal=openai-api-key=sk-... \
  --from-literal=anthropic-api-key=sk-ant-... \
  --from-literal=database-url=postgresql://...

# From file
kubectl create secret generic agent-secrets \
  --from-file=./secrets/
```

### Use in Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: agent
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: openai-api-key
            - name: FASTAGENTIC_DURABLE_STORE
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: database-url
```

### Sealed Secrets

For GitOps workflows:

```bash
# Install kubeseal
brew install kubeseal

# Seal secrets
kubeseal --format yaml < secret.yaml > sealed-secret.yaml

# Commit sealed-secret.yaml (safe!)
```

## HashiCorp Vault

### Setup

```bash
# Enable KV secrets engine
vault secrets enable -path=fastagentic kv-v2

# Store secrets
vault kv put fastagentic/llm \
  openai_api_key=sk-... \
  anthropic_api_key=sk-ant-...

vault kv put fastagentic/database \
  url=postgresql://user:pass@host:5432/db
```

### Kubernetes Integration

```yaml
# Vault Agent Injector
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "fastagentic"
        vault.hashicorp.com/agent-inject-secret-llm: "fastagentic/llm"
        vault.hashicorp.com/agent-inject-template-llm: |
          {{- with secret "fastagentic/llm" -}}
          export OPENAI_API_KEY="{{ .Data.data.openai_api_key }}"
          export ANTHROPIC_API_KEY="{{ .Data.data.anthropic_api_key }}"
          {{- end }}
```

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agent-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: agent-secrets
    creationPolicy: Owner
  data:
    - secretKey: openai-api-key
      remoteRef:
        key: fastagentic/llm
        property: openai_api_key
    - secretKey: database-url
      remoteRef:
        key: fastagentic/database
        property: url
```

## AWS Secrets Manager

### Store Secrets

```bash
aws secretsmanager create-secret \
  --name fastagentic/llm \
  --secret-string '{"openai_api_key":"sk-...","anthropic_api_key":"sk-ant-..."}'
```

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agent-secrets
spec:
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: agent-secrets
  data:
    - secretKey: openai-api-key
      remoteRef:
        key: fastagentic/llm
        property: openai_api_key
```

### Lambda with Secrets

```python
import boto3
import json

def get_secrets():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='fastagentic/llm')
    return json.loads(response['SecretString'])
```

## Google Secret Manager

### Store Secrets

```bash
echo -n "sk-..." | gcloud secrets create openai-api-key --data-file=-
```

### Cloud Run

```bash
gcloud run deploy my-agent \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest"
```

### External Secrets

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: gcp-secrets
spec:
  provider:
    gcpsm:
      projectID: my-project

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agent-secrets
spec:
  secretStoreRef:
    name: gcp-secrets
    kind: SecretStore
  target:
    name: agent-secrets
  data:
    - secretKey: openai-api-key
      remoteRef:
        key: openai-api-key
```

## Secret Rotation

### Automated Rotation

```yaml
# AWS Secrets Manager rotation
aws secretsmanager rotate-secret \
  --secret-id fastagentic/llm \
  --rotation-lambda-arn arn:aws:lambda:...
```

### Manual Rotation Process

1. Generate new credentials
2. Update secret in manager
3. Verify application uses new credentials
4. Revoke old credentials
5. Document rotation in audit log

### Rotation Schedule

| Secret Type | Rotation Frequency |
|-------------|-------------------|
| LLM API keys | 90 days |
| Database passwords | 30 days |
| Service accounts | 90 days |
| TLS certificates | Before expiry |

## Audit

### Access Logging

Enable audit logging for secret access:

```bash
# Vault
vault audit enable file file_path=/var/log/vault-audit.log

# AWS
aws cloudtrail create-trail --name secrets-audit ...
```

### Access Review

Quarterly review checklist:
- [ ] Who has access to secrets?
- [ ] Are there unused credentials?
- [ ] Are rotation schedules followed?
- [ ] Are access patterns normal?

## Best Practices

1. **Use different secrets per environment**
   ```
   fastagentic/dev/llm
   fastagentic/staging/llm
   fastagentic/prod/llm
   ```

2. **Scope API keys narrowly**
   - Use project-specific keys when possible
   - Limit permissions to what's needed

3. **Monitor for leaked secrets**
   - GitHub secret scanning
   - GitGuardian or similar tools

4. **Have incident response plan**
   - How to rotate compromised secrets quickly
   - Who to notify
   - How to audit impact

## Next Steps

- [Environment Variables](environment-vars.md) - Configuration reference
- [Security Hardening](../security/hardening.md) - Security guide
- [Compliance](../security/compliance.md) - Audit requirements
