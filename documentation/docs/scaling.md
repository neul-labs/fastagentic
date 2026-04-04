# Scaling FastAgentic

Guide to deploying FastAgentic at scale with horizontal scaling, connection pooling, and production configurations.

## Quick Start

```bash
# Development (single worker, auto-reload)
fastagentic run --reload

# Production (Gunicorn, 4 workers)
fastagentic run --server gunicorn --workers 4

# Production with backpressure
fastagentic run --server gunicorn --workers 4 --max-concurrent 100
```

## Server Options

### Uvicorn (Development)

Default server, great for development:

```bash
fastagentic run --reload
```

### Gunicorn (Production)

Production-grade process manager with Uvicorn workers:

```bash
fastagentic run --server gunicorn --workers 4
```

Benefits:
- Pre-fork worker model for parallel processing
- Graceful restarts and zero-downtime deployments
- Worker recycling to prevent memory leaks
- Signal handling for proper shutdown

**Note:** Requires `gunicorn` to be installed:
```bash
pip install gunicorn
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--server` | uvicorn | Server type: `uvicorn` or `gunicorn` |
| `--host` | 127.0.0.1 | Host to bind to |
| `--port` | 8000 | Port to bind to |
| `--workers` | 1 | Number of worker processes |
| `--reload` | false | Enable auto-reload (dev only) |
| `--max-concurrent` | unlimited | Max concurrent requests per worker |
| `--instance-id` | auto | Instance ID for metrics |
| `--redis-pool-size` | 10 | Redis connection pool size |
| `--db-pool-size` | 5 | Database connection pool size |
| `--db-max-overflow` | 10 | Database pool overflow connections |
| `--timeout-graceful` | 30 | Graceful shutdown timeout (seconds) |

## Connection Pooling

### Redis Pool

Configure Redis connection pool for high throughput:

```bash
fastagentic run --server gunicorn --workers 4 --redis-pool-size 20
```

Or via environment:
```bash
export FASTAGENTIC_REDIS_POOL_SIZE=20
export FASTAGENTIC_REDIS_POOL_TIMEOUT=5.0
```

Pool per worker:
- Each Gunicorn worker gets its own pool
- Total connections = workers × pool_size
- Example: 4 workers × 20 pool = 80 max connections

### Database Pool

Configure PostgreSQL connection pool:

```bash
fastagentic run --server gunicorn --workers 4 --db-pool-size 10 --db-max-overflow 20
```

Or via environment:
```bash
export FASTAGENTIC_DB_POOL_SIZE=10
export FASTAGENTIC_DB_MAX_OVERFLOW=20
```

Pool settings:
- `pool_size`: Base connections per worker
- `max_overflow`: Additional connections under load
- `pool_recycle`: Recycle connections after 30 minutes
- `pool_pre_ping`: Test connections before use

## Concurrency Limits

Prevent server overload with request limits:

```bash
fastagentic run --server gunicorn --workers 4 --max-concurrent 100
```

Behavior:
- Limits concurrent requests per worker
- Returns HTTP 503 when limit reached
- Includes `Retry-After` header
- Health checks bypass the limit

Example response when limit reached:
```json
{
  "error": "Service temporarily unavailable",
  "message": "Too many concurrent requests",
  "retry_after": 1
}
```

## Cluster Deployments

### Instance Identification

Track metrics per instance in cluster deployments:

```bash
# Manual instance ID
fastagentic run --server gunicorn --instance-id worker-1

# Auto-generated (hostname-pid)
fastagentic run --server gunicorn
```

Instance ID appears in:
- `/health` endpoint response
- `/ready` endpoint response
- `/metrics` endpoint response
- `X-FastAgentic-Instance` response header
- Structured logs

### Environment Variables

Configure via environment for container deployments:

```bash
# Server configuration
FASTAGENTIC_SERVER=gunicorn
FASTAGENTIC_HOST=0.0.0.0
FASTAGENTIC_PORT=8000
FASTAGENTIC_WORKERS=4
FASTAGENTIC_MAX_CONCURRENT=100
FASTAGENTIC_INSTANCE_ID=worker-1

# Pool configuration
FASTAGENTIC_REDIS_POOL_SIZE=20
FASTAGENTIC_REDIS_POOL_TIMEOUT=5.0
FASTAGENTIC_DB_POOL_SIZE=10
FASTAGENTIC_DB_MAX_OVERFLOW=20

# Timeouts
FASTAGENTIC_TIMEOUT_KEEP_ALIVE=5
FASTAGENTIC_TIMEOUT_GRACEFUL_SHUTDOWN=30
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastagentic-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastagentic-agent
  template:
    metadata:
      labels:
        app: fastagentic-agent
    spec:
      containers:
      - name: agent
        image: your-agent:latest
        command: ["fastagentic", "run"]
        args:
          - "--server=gunicorn"
          - "--workers=2"
          - "--host=0.0.0.0"
          - "--max-concurrent=50"
        env:
        - name: FASTAGENTIC_INSTANCE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: FASTAGENTIC_REDIS_POOL_SIZE
          value: "10"
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastagentic-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fastagentic-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync

# Copy application
COPY . .

# Production settings
ENV FASTAGENTIC_HOST=0.0.0.0
ENV FASTAGENTIC_PORT=8000

# Run with Gunicorn
CMD ["fastagentic", "run", "--server", "gunicorn", "--workers", "4"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  agent:
    build: .
    command: >
      fastagentic run
      --server gunicorn
      --workers 4
      --max-concurrent 100
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - FASTAGENTIC_REDIS_POOL_SIZE=20
    depends_on:
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 512M

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## Monitoring

### Health Endpoints

```bash
# Liveness check
curl http://localhost:8000/health
# {"status": "healthy", "version": "1.0.0", "title": "My Agent", "instance_id": "worker-1"}

# Readiness check
curl http://localhost:8000/ready
# {"ready": true, "checks": {"app": true, "durable_store": true}, "instance_id": "worker-1"}

# Metrics
curl http://localhost:8000/metrics
# {"instance_id": "worker-1", "version": "1.0.0", "config": {...}}
```

### Response Headers

Every response includes:
```
X-FastAgentic-Instance: worker-1
```

### Structured Logging

Logs include instance ID:
```json
{
  "event": "request_completed",
  "instance_id": "worker-1",
  "path": "/chat",
  "duration_ms": 150
}
```

## Recommended Settings

### Development
```bash
fastagentic run --reload
```

### Staging
```bash
fastagentic run --server gunicorn --workers 2 --max-concurrent 50
```

### Production (Small)
```bash
fastagentic run \
  --server gunicorn \
  --workers 4 \
  --max-concurrent 100 \
  --redis-pool-size 20
```

### Production (Large)
```bash
fastagentic run \
  --server gunicorn \
  --workers 8 \
  --max-concurrent 200 \
  --redis-pool-size 30 \
  --db-pool-size 15 \
  --db-max-overflow 30 \
  --timeout-graceful 60
```

## Troubleshooting

### "Too many concurrent requests"

Increase `--max-concurrent` or add more workers/replicas.

### Connection pool exhaustion

```bash
# Increase Redis pool
fastagentic run --redis-pool-size 50

# Increase DB pool
fastagentic run --db-pool-size 20 --db-max-overflow 40
```

### Slow graceful shutdown

Increase timeout:
```bash
fastagentic run --timeout-graceful 60
```

### Workers keep dying

Check memory limits and consider:
- Reducing worker count
- Increasing memory limits
- Adding `--max-concurrent` to prevent overload
