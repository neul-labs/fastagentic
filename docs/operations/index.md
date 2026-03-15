# Operations Guide

FastAgentic provides a unified operations layer for AI agents. This guide covers deployment, configuration, observability, security, and operational procedures.

## Why DevOps Teams Choose FastAgentic

### The Problem: AI Agent Deployment Chaos

Without FastAgentic, deploying AI agents means:
- Each agent framework has different deployment patterns
- Custom instrumentation required for observability
- No unified governance across agent types
- Compliance gaps (audit trails, cost tracking)
- Operational complexity multiplies with each new framework

### The Solution: Unified Agent Operations

**1. Deploy Once, Run Any Framework**
```bash
# Same deployment for PydanticAI, LangGraph, CrewAI, LangChain
docker run -e FASTAGENTIC_OIDC_ISSUER=https://auth.company.com \
           -e FASTAGENTIC_DURABLE_STORE=postgres://... \
           fastagentic/runtime:latest
```

**2. Built-in Governance Without Custom Code**
- Cost tracking per run, user, tenant (out of the box)
- Audit trails satisfy SOC2/GDPR requirements
- Policy enforcement (quotas, rate limits) at runtime
- Role-based access with OAuth2/OIDC integration

**3. Production Reliability by Default**
- Checkpoint durability (Redis/Postgres/S3)
- Automatic resume on failure
- Idempotent execution with run IDs
- Graceful degradation under load

**4. Zero-Instrumentation Observability**
- OpenTelemetry traces span all agent operations
- Standard RED metrics per endpoint
- Token usage and cost metrics
- Correlated logs with trace IDs

## Quick Production Deployment

### Docker

```bash
docker run -d \
  --name fastagentic \
  -p 8000:8000 \
  -e FASTAGENTIC_OIDC_ISSUER=https://auth.company.com \
  -e FASTAGENTIC_DURABLE_STORE=redis://redis:6379 \
  -e FASTAGENTIC_OTEL_ENDPOINT=http://otel-collector:4318 \
  fastagentic/runtime:latest
```

### Kubernetes

```bash
helm repo add fastagentic https://charts.fastagentic.dev
helm install my-agents fastagentic/runtime \
  --set auth.oidcIssuer=https://auth.company.com \
  --set durability.backend=postgres \
  --set durability.connectionString=postgresql://... \
  --set telemetry.enabled=true
```

## Operational Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │  (TLS, routing) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │FastAgentic│  │FastAgentic│  │FastAgentic│
        │ Instance 1│  │ Instance 2│  │ Instance N│
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐        ┌──────────┐       ┌──────────┐
   │  Durable │        │   OTEL   │       │   LLM    │
   │  Store   │        │ Collector│       │ Providers│
   │(Postgres)│        │          │       │          │
   └──────────┘        └──────────┘       └──────────┘
```

## Documentation Sections

### Deployment

- [Docker Deployment](deployment/docker.md) - Container deployment guide
- [Kubernetes Deployment](deployment/kubernetes.md) - Production K8s with Helm
- [Serverless Deployment](deployment/serverless.md) - AWS Lambda, Cloud Run patterns

### Configuration

- [Environment Variables](configuration/environment-vars.md) - Complete reference
- [Secrets Management](configuration/secrets-management.md) - Secure credential handling

### Observability

- [Metrics Reference](observability/metrics.md) - Available metrics and dashboards
- [Distributed Tracing](observability/tracing.md) - OpenTelemetry setup
- [Alerting Guide](observability/alerting.md) - Alert rules and runbooks

### Security

- [Security Hardening](security/hardening.md) - Production security guide
- [Compliance Guide](security/compliance.md) - SOC2, GDPR, audit trails

### Runbooks

- [Troubleshooting](runbook/troubleshooting.md) - Common issues and solutions

## Key Operational Features

### Health Checks

FastAgentic exposes standard health endpoints:

| Endpoint | Purpose | Returns |
|----------|---------|---------|
| `/health/live` | Liveness probe | `200` if process alive |
| `/health/ready` | Readiness probe | `200` if accepting traffic |
| `/health/startup` | Startup probe | `200` when initialized |

### Graceful Shutdown

On SIGTERM:
1. Stop accepting new requests
2. Wait for in-flight runs to checkpoint
3. Drain background workers
4. Close database connections
5. Exit cleanly

Configure drain timeout: `FASTAGENTIC_SHUTDOWN_TIMEOUT=30`

### Multi-Tenancy

FastAgentic supports tenant isolation:

```yaml
# config/settings.yaml
tenancy:
  enabled: true
  isolation: database  # or "schema" or "row"
  header: X-Tenant-ID
```

Each tenant gets:
- Separate durable stores (or schemas)
- Independent quotas and rate limits
- Isolated audit trails
- Per-tenant cost tracking

### Horizontal Scaling

FastAgentic instances are stateless. Scale horizontally behind a load balancer:

- Sticky sessions not required
- All state in durable store (Redis/Postgres)
- Background workers share centralized queue
- No instance-to-instance communication needed

## Production Checklist

Before going to production:

- [ ] Configure OIDC authentication
- [ ] Set up durable store (Postgres recommended)
- [ ] Enable OpenTelemetry exports
- [ ] Configure rate limits and quotas
- [ ] Set cost guardrails
- [ ] Enable audit logging
- [ ] Configure TLS termination
- [ ] Set resource limits (CPU, memory)
- [ ] Configure health check probes
- [ ] Set up alerting rules
- [ ] Document runbook procedures
- [ ] Test disaster recovery

## Next Steps

- [Docker Deployment](deployment/docker.md) - Start with containers
- [Environment Variables](configuration/environment-vars.md) - Configuration reference
- [Metrics Reference](observability/metrics.md) - Set up dashboards
