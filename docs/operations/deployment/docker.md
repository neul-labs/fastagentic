# Docker Deployment

Deploy FastAgentic applications using Docker containers.

## Quick Start

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run FastAgentic
EXPOSE 8000
CMD ["fastagentic", "run", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
fastagentic[pydanticai,langgraph,otel]>=0.2.0
```

### Build and Run

```bash
# Build
docker build -t my-agent .

# Run with minimal config
docker run -p 8000:8000 my-agent

# Run with full production config
docker run -d \
  --name my-agent \
  -p 8000:8000 \
  -e FASTAGENTIC_OIDC_ISSUER=https://auth.company.com \
  -e FASTAGENTIC_DURABLE_STORE=redis://redis:6379 \
  -e FASTAGENTIC_OTEL_ENDPOINT=http://otel:4318 \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  my-agent
```

## Production Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy application
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health/ready || exit 1

# Run
EXPOSE 8000
CMD ["fastagentic", "run", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose

### Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FASTAGENTIC_LOG_LEVEL=DEBUG
      - FASTAGENTIC_DURABLE_STORE=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
    volumes:
      - .:/app  # Hot reload

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  agent:
    image: my-agent:${VERSION:-latest}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        max_attempts: 3
    ports:
      - "8000:8000"
    environment:
      - FASTAGENTIC_OIDC_ISSUER=${OIDC_ISSUER}
      - FASTAGENTIC_DURABLE_STORE=postgres://${DB_USER}:${DB_PASS}@postgres:5432/fastagentic
      - FASTAGENTIC_OTEL_ENDPOINT=http://otel-collector:4318
      - FASTAGENTIC_OTEL_SERVICE_NAME=my-agent
    depends_on:
      postgres:
        condition: service_healthy
      otel-collector:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASS}
      POSTGRES_DB: fastagentic
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  otel-collector:
    image: otel/opentelemetry-collector:latest
    command: ["--config", "/etc/otel-config.yaml"]
    volumes:
      - ./otel-config.yaml:/etc/otel-config.yaml

volumes:
  postgres_data:
```

## Environment Variables

Essential variables for Docker deployment:

```bash
# Core
FASTAGENTIC_HOST=0.0.0.0
FASTAGENTIC_PORT=8000
FASTAGENTIC_WORKERS=4  # Match CPU cores

# Authentication
FASTAGENTIC_OIDC_ISSUER=https://auth.company.com
FASTAGENTIC_OIDC_AUDIENCE=my-agent

# Durability
FASTAGENTIC_DURABLE_STORE=postgres://user:pass@host:5432/db

# Observability
FASTAGENTIC_OTEL_ENDPOINT=http://otel-collector:4318
FASTAGENTIC_OTEL_SERVICE_NAME=my-agent
FASTAGENTIC_LOG_FORMAT=json

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Multi-Architecture Builds

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myregistry/my-agent:latest \
  --push .
```

## Security Best Practices

1. **Non-root user**: Always run as non-root
2. **Read-only filesystem**: Mount app as read-only where possible
3. **No secrets in image**: Use environment variables or secrets management
4. **Minimal base image**: Use slim/alpine images
5. **Scan for vulnerabilities**: Use `docker scan` or Trivy

```dockerfile
# Security-hardened runtime
FROM python:3.11-slim AS runtime

# Remove unnecessary packages
RUN apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# Read-only app directory
COPY --from=builder --chown=root:root /app /app
RUN chmod -R 555 /app

USER appuser
```

## Logging

Configure JSON logging for container environments:

```bash
docker run \
  -e FASTAGENTIC_LOG_FORMAT=json \
  -e FASTAGENTIC_LOG_LEVEL=INFO \
  my-agent
```

Log output:
```json
{"timestamp":"2024-01-15T10:30:00Z","level":"INFO","message":"Request completed","trace_id":"abc123","run_id":"run-456","duration_ms":150}
```

## Health Checks

FastAgentic exposes health endpoints:

```bash
# Liveness - is the process alive?
curl http://localhost:8000/health/live

# Readiness - can it accept traffic?
curl http://localhost:8000/health/ready

# Startup - has it finished initializing?
curl http://localhost:8000/health/startup
```

## Resource Limits

Recommended starting points:

| Workload | CPU | Memory | Workers |
|----------|-----|--------|---------|
| Development | 0.5 | 512MB | 1 |
| Light production | 1 | 1GB | 2 |
| Standard production | 2 | 2GB | 4 |
| High throughput | 4 | 4GB | 8 |

## Next Steps

- [Kubernetes Deployment](kubernetes.md) - Scale with K8s
- [Environment Variables](../configuration/environment-vars.md) - Full reference
- [Observability](../observability/metrics.md) - Monitoring setup
