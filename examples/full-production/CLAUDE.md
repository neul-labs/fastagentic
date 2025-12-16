# Full Production Example - Claude Code Guide

This is a complete production-ready FastAgentic application demonstrating all features.

## Project Structure

```
full-production/
├── CLAUDE.md              # This file
├── app.py                 # Main application with all features
├── agent.py               # Agent definition
├── config.py              # Configuration management
├── models.py              # API models
├── hooks.py               # Custom lifecycle hooks
├── policies.py            # RBAC and policy definitions
├── docker-compose.yml     # Local infrastructure
├── Dockerfile             # Production image
├── kubernetes/            # K8s manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── pyproject.toml
├── .env.example
└── README.md
```

## Key Features Demonstrated

1. **Authentication** - OIDC/JWT integration
2. **Authorization** - RBAC with scopes
3. **Durability** - Redis checkpointing
4. **Observability** - OTEL traces and metrics
5. **Policy** - Rate limits, budgets
6. **Compliance** - PII detection
7. **HITL** - Approval workflows
8. **Streaming** - SSE responses

## Key Commands

```bash
# Start infrastructure
docker-compose up -d

# Run application
uv run fastagentic run

# Check production readiness
uv run fastagentic ops readiness

# Deploy to Kubernetes
kubectl apply -f kubernetes/
```

## Configuration

All settings via environment variables or `config.py`:

```python
# Required
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379
OIDC_ISSUER=https://auth.example.com

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# Policy
RATE_LIMIT_RPM=60
BUDGET_DAILY_USD=100
```

## Production Checklist

- [ ] Set `FASTAGENTIC_ENV=production`
- [ ] Configure OIDC provider
- [ ] Enable Redis for durability
- [ ] Set up OTEL collector
- [ ] Configure rate limits
- [ ] Enable PII detection
- [ ] Set up monitoring dashboards
- [ ] Configure alerting

## Common Tasks

### Enable authentication

```python
app = App(
    oidc_issuer="https://auth.example.com",
    oidc_audience="your-api-audience",
)
```

### Add rate limiting

```python
from fastagentic import RateLimit

app = App(
    rate_limit=RateLimit(rpm=60, tpm=100000, by="user"),
)
```

### Enable observability

```python
from fastagentic.integrations import LangfuseIntegration

app = App(
    integrations=[
        LangfuseIntegration(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        ),
    ],
)
```
