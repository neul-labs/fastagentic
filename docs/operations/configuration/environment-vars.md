# Environment Variables Reference

Complete reference for all FastAgentic configuration options.

## Configuration Precedence

1. **Command-line flags** (highest priority)
2. **Environment variables** (`FASTAGENTIC_*`)
3. **Config file** (`config/settings.yaml`)
4. **`.env` file** (local development)
5. **Defaults** (lowest priority)

## Core Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_HOST` | Bind address | `127.0.0.1` | No |
| `FASTAGENTIC_PORT` | HTTP port | `8000` | No |
| `FASTAGENTIC_WORKERS` | Worker processes | `1` | No |
| `FASTAGENTIC_RELOAD` | Auto-reload on changes | `false` | No |
| `FASTAGENTIC_LOG_LEVEL` | Logging verbosity | `INFO` | No |
| `FASTAGENTIC_LOG_FORMAT` | Log format (`json` or `text`) | `json` | No |
| `FASTAGENTIC_ENV` | Environment name | `production` | No |

### Example

```bash
export FASTAGENTIC_HOST=0.0.0.0
export FASTAGENTIC_PORT=8000
export FASTAGENTIC_WORKERS=4
export FASTAGENTIC_LOG_LEVEL=INFO
export FASTAGENTIC_LOG_FORMAT=json
```

## Authentication

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_OIDC_ISSUER` | OIDC issuer URL | - | Prod: Yes |
| `FASTAGENTIC_OIDC_AUDIENCE` | Expected audience claim | - | Prod: Yes |
| `FASTAGENTIC_OIDC_JWKS_URL` | JWKS endpoint | Auto from issuer | No |
| `FASTAGENTIC_OIDC_ALGORITHMS` | Allowed JWT algorithms | `RS256` | No |
| `FASTAGENTIC_AUTH_REQUIRED` | Enforce authentication | `true` | No |
| `FASTAGENTIC_AUTH_HEADER` | Auth header name | `Authorization` | No |

### Example

```bash
export FASTAGENTIC_OIDC_ISSUER=https://auth.company.com
export FASTAGENTIC_OIDC_AUDIENCE=my-agent
export FASTAGENTIC_AUTH_REQUIRED=true
```

## Durability

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_DURABLE_STORE` | Connection URL | `memory://` | Prod: Yes |
| `FASTAGENTIC_CHECKPOINT_INTERVAL` | Checkpoint frequency (seconds) | `30` | No |
| `FASTAGENTIC_RUN_TTL` | Run retention period | `7d` | No |
| `FASTAGENTIC_CHECKPOINT_TTL` | Checkpoint retention | `30d` | No |
| `FASTAGENTIC_MAX_CHECKPOINTS` | Max checkpoints per run | `100` | No |

### Connection URLs

```bash
# Redis
export FASTAGENTIC_DURABLE_STORE=redis://localhost:6379/0

# Redis with auth
export FASTAGENTIC_DURABLE_STORE=redis://:password@localhost:6379/0

# PostgreSQL
export FASTAGENTIC_DURABLE_STORE=postgresql://user:pass@localhost:5432/fastagentic

# S3 (archival)
export FASTAGENTIC_DURABLE_STORE=s3://bucket-name/prefix?region=us-east-1
```

## Telemetry

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_OTEL_ENDPOINT` | OTLP collector endpoint | - | Recommended |
| `FASTAGENTIC_OTEL_SERVICE_NAME` | Service identifier | `fastagentic` | No |
| `FASTAGENTIC_OTEL_SERVICE_VERSION` | Service version | `unknown` | No |
| `FASTAGENTIC_METRICS_ENABLED` | Enable metrics export | `true` | No |
| `FASTAGENTIC_TRACES_ENABLED` | Enable trace export | `true` | No |
| `FASTAGENTIC_LOGS_ENABLED` | Enable log export | `false` | No |
| `FASTAGENTIC_OTEL_SAMPLE_RATE` | Trace sampling rate (0-1) | `1.0` | No |

### Example

```bash
export FASTAGENTIC_OTEL_ENDPOINT=http://otel-collector:4318
export FASTAGENTIC_OTEL_SERVICE_NAME=my-agent
export FASTAGENTIC_OTEL_SERVICE_VERSION=1.0.0
export FASTAGENTIC_OTEL_SAMPLE_RATE=0.1  # 10% sampling in production
```

## Policy & Governance

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_RATE_LIMIT_REQUESTS` | Requests per minute | `100` | No |
| `FASTAGENTIC_RATE_LIMIT_BURST` | Burst allowance | `20` | No |
| `FASTAGENTIC_QUOTA_RUNS_DAILY` | Max runs per user/day | `unlimited` | No |
| `FASTAGENTIC_QUOTA_TOKENS_DAILY` | Token limit per user/day | `unlimited` | No |
| `FASTAGENTIC_COST_LIMIT_DAILY` | USD limit per user/day | `unlimited` | No |
| `FASTAGENTIC_COST_LIMIT_PER_RUN` | USD limit per run | `unlimited` | No |
| `FASTAGENTIC_AUDIT_ENABLED` | Enable audit logging | `true` | No |
| `FASTAGENTIC_AUDIT_DESTINATION` | Audit log destination | `stdout` | No |

### Example

```bash
export FASTAGENTIC_RATE_LIMIT_REQUESTS=60
export FASTAGENTIC_QUOTA_TOKENS_DAILY=100000
export FASTAGENTIC_COST_LIMIT_DAILY=10.00
export FASTAGENTIC_COST_LIMIT_PER_RUN=1.00
export FASTAGENTIC_AUDIT_ENABLED=true
```

## Security

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_CORS_ORIGINS` | Allowed CORS origins | `*` | Prod: Set |
| `FASTAGENTIC_CORS_METHODS` | Allowed methods | `GET,POST,OPTIONS` | No |
| `FASTAGENTIC_CORS_HEADERS` | Allowed headers | `*` | No |
| `FASTAGENTIC_TLS_CERT` | TLS certificate path | - | No |
| `FASTAGENTIC_TLS_KEY` | TLS private key path | - | No |
| `FASTAGENTIC_PII_REDACTION` | Redact PII in logs | `true` | No |
| `FASTAGENTIC_TRUSTED_PROXIES` | Trusted proxy IPs | - | No |

### Example

```bash
export FASTAGENTIC_CORS_ORIGINS=https://app.company.com,https://admin.company.com
export FASTAGENTIC_PII_REDACTION=true
export FASTAGENTIC_TRUSTED_PROXIES=10.0.0.0/8
```

## MCP Protocol

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_MCP_ENABLED` | Enable MCP endpoints | `true` | No |
| `FASTAGENTIC_MCP_STDIO` | Enable stdio transport | `true` | No |
| `FASTAGENTIC_MCP_HTTP` | Enable HTTP transport | `true` | No |
| `FASTAGENTIC_MCP_PATH` | MCP endpoint prefix | `/mcp` | No |

## Streaming

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_SSE_ENABLED` | Enable SSE streaming | `true` | No |
| `FASTAGENTIC_WS_ENABLED` | Enable WebSocket | `true` | No |
| `FASTAGENTIC_STREAM_TIMEOUT` | Max stream duration | `300s` | No |
| `FASTAGENTIC_STREAM_KEEPALIVE` | Keepalive interval | `15s` | No |

## Shutdown

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FASTAGENTIC_SHUTDOWN_TIMEOUT` | Graceful shutdown timeout | `30s` | No |
| `FASTAGENTIC_DRAIN_TIMEOUT` | Request drain timeout | `10s` | No |

## LLM Providers

Set API keys for LLM providers (used by underlying agent frameworks):

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google
export GOOGLE_API_KEY=...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://....openai.azure.com/

# AWS Bedrock
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

## Configuration File

### config/settings.yaml

```yaml
app:
  title: My Agent
  version: 1.0.0

auth:
  oidc_issuer: https://auth.company.com
  oidc_audience: my-agent

durability:
  backend: postgres
  connection_string: ${DATABASE_URL}  # Reference env var
  checkpoint_interval: 30
  run_ttl: 7d

telemetry:
  enabled: true
  service_name: my-agent
  exporter_endpoint: ${OTEL_ENDPOINT}

policy:
  rate_limits:
    default: 100/minute
    endpoints:
      /expensive: 10/minute
  quotas:
    tokens_per_user_per_day: 100000
  cost_guards:
    max_per_run: 1.00
    max_per_user_per_day: 10.00

security:
  cors_origins:
    - https://app.company.com
  pii_redaction: true
```

## .env File (Development)

```bash
# .env
FASTAGENTIC_LOG_LEVEL=DEBUG
FASTAGENTIC_RELOAD=true
FASTAGENTIC_DURABLE_STORE=redis://localhost:6379
FASTAGENTIC_AUTH_REQUIRED=false

OPENAI_API_KEY=sk-...
```

## Validation

Check configuration:

```bash
fastagentic config validate
```

Show resolved configuration:

```bash
fastagentic config show
```

## Next Steps

- [Secrets Management](secrets-management.md) - Secure credential handling
- [Docker Deployment](../deployment/docker.md) - Container deployment
- [Kubernetes Deployment](../deployment/kubernetes.md) - K8s deployment
