# Full Production Example

A complete production-ready FastAgentic deployment demonstrating all features.

## Features

- Authentication (OIDC/JWT)
- Authorization (RBAC)
- Durability (Redis)
- Observability (OTEL)
- Policy (Rate limits, budgets)
- Compliance (PII detection)
- Kubernetes deployment

## Quick Start

```bash
# Start infrastructure
docker-compose up -d

# Install and run
uv sync
cp .env.example .env
uv run fastagentic run

# Check production readiness
uv run fastagentic ops readiness
```

## Deploy to Kubernetes

```bash
# Build image
docker build -t fastagentic-agent:latest .

# Deploy
kubectl apply -f kubernetes/
```

## Configuration

See `.env.example` for all configuration options.

## Architecture

See `CLAUDE.md` for detailed documentation.
