# Production Checklist

This checklist ensures your FastAgentic deployment is ready for production traffic. Each section explains **why** it matters and **how** to implement it.

---

## Quick Assessment

Rate your readiness (0-3 per category):

| Category | 0 | 1 | 2 | 3 |
|----------|---|---|---|---|
| **Observability** | Nothing | Logs only | Basic tracing | Full analytics |
| **Security** | None | Auth only | + Guardrails | + PII handling |
| **Reliability** | No handling | Retry only | + Circuit breaker | + Fallbacks |
| **Operations** | Manual | Docker | Kubernetes | + CI/CD |
| **Data** | None | Checkpoints | + Encryption | + Backup |

**Score interpretation:**
- 0-5: Development only
- 6-10: Staging / internal
- 11-15: Production ready

---

## Checklist

### 1. Observability

**Why it matters**: You can't fix what you can't see. In production, debugging becomes critical.

#### Essential

- [ ] **Tracing enabled** — Every request can be traced end-to-end
  ```python
  from fastagentic.integrations.langfuse import LangfuseHook

  app = App(hooks=[LangfuseHook()])
  ```

- [ ] **Structured logging** — JSON logs with request IDs
  ```python
  app = App(
      log_format="json",
      log_level="INFO",
  )
  ```

- [ ] **Health endpoint** — Kubernetes/load balancer can check status
  ```bash
  curl http://localhost:8000/health
  # {"status": "healthy", "version": "0.2.0"}
  ```

#### Recommended

- [ ] **Cost tracking** — Know how much each request costs
  ```python
  LangfuseHook(
      track_cost=True,
      cost_per_user=True,
  )
  ```

- [ ] **Metrics exported** — Prometheus can scrape metrics
  ```bash
  curl http://localhost:8000/metrics
  # fastagentic_requests_total{endpoint="/chat"} 1234
  ```

- [ ] **Alerting configured** — Get notified on errors
  ```yaml
  # alerts.yaml
  - alert: HighErrorRate
    expr: rate(fastagentic_errors_total[5m]) > 0.1
  ```

**Tools**: [Langfuse](../integrations/langfuse.md), [Logfire](../integrations/logfire.md), [Datadog](../integrations/datadog.md)

---

### 2. Security

**Why it matters**: AI agents are attack surfaces. Prompt injection is the #1 OWASP risk for LLM apps.

#### Essential

- [ ] **Authentication enabled** — Know who's calling
  ```python
  app = App(
      auth=OIDCAuth(
          issuer="https://auth.example.com",
          audience="my-agent",
      ),
  )
  ```

- [ ] **Prompt injection protection** — Block malicious inputs
  ```python
  from fastagentic.integrations.lakera import LakeraHook

  app = App(hooks=[LakeraHook(on_detection="reject")])
  ```

- [ ] **HTTPS only** — Never expose HTTP in production
  ```bash
  # In production, always terminate TLS
  ```

#### Recommended

- [ ] **Rate limiting** — Prevent abuse
  ```python
  app = App(
      rate_limit=RateLimit(
          rpm=60,
          by="user",
      ),
  )
  ```

- [ ] **PII handling** — Detect and handle sensitive data
  ```python
  LakeraHook(categories=["pii"])
  ```

- [ ] **Audit logging** — Record who did what
  ```python
  @hook("on_request")
  async def audit(ctx: HookContext):
      await audit_log.record(
          user=ctx.user.id,
          action=ctx.endpoint,
          input=ctx.request,
      )
  ```

- [ ] **Secrets management** — No hardcoded API keys
  ```python
  # Use environment variables
  app = App()  # Reads from OPENAI_API_KEY, etc.
  ```

**Tools**: [Lakera](../integrations/lakera.md), [Guardrails AI](../integrations/guardrails-ai.md)

---

### 3. Reliability

**Why it matters**: LLM APIs fail. Rate limits hit. Networks timeout. Your agent should handle this gracefully.

#### Essential

- [ ] **Retry policy** — Handle transient failures
  ```python
  @agent_endpoint(
      retry=RetryPolicy(
          max_attempts=3,
          backoff="exponential",
          retry_on=["rate_limit", "timeout"],
      ),
  )
  ```

- [ ] **Timeouts** — Don't hang forever
  ```python
  @agent_endpoint(
      timeout=Timeout(
          total_ms=60000,      # 1 minute max
          llm_call_ms=30000,   # 30 seconds per LLM call
      ),
  )
  ```

- [ ] **Graceful degradation** — Fail helpfully
  ```python
  @hook("on_error")
  async def handle_error(ctx: HookContext):
      return {"error": "Service temporarily unavailable"}
  ```

#### Recommended

- [ ] **Circuit breaker** — Stop calling failing services
  ```python
  @agent_endpoint(
      circuit_breaker=CircuitBreaker(
          failure_threshold=5,
          reset_timeout_ms=30000,
      ),
  )
  ```

- [ ] **Model fallbacks** — Switch models when primary fails
  ```python
  from fastagentic.integrations.portkey import PortkeyGateway

  app = App(
      llm_gateway=PortkeyGateway(
          config={
              "strategy": {"mode": "fallback"},
              "targets": [
                  {"virtual_key": "openai"},
                  {"virtual_key": "anthropic"},
              ],
          },
      ),
  )
  ```

- [ ] **Idempotency** — Safe to retry requests
  ```python
  @agent_endpoint(
      idempotency=True,  # Same request ID = same result
  )
  ```

**Reference**: [Reliability Patterns](../reliability.md)

---

### 4. Durability

**Why it matters**: Long-running agent workflows should survive crashes and restarts.

#### Essential

- [ ] **Durable store configured** — Checkpoints persist
  ```python
  app = App(
      durable_store="redis://localhost:6379",
  )
  ```

- [ ] **Checkpointing enabled** — Resume from failure
  ```python
  @agent_endpoint(
      durable=True,  # Auto-checkpoint
  )
  ```

#### Recommended

- [ ] **Backup strategy** — Don't lose data
  ```bash
  # Redis: Enable RDB or AOF persistence
  # Postgres: Regular pg_dump
  ```

- [ ] **Encryption at rest** — Protect checkpoint data
  ```python
  app = App(
      durable_store="redis://localhost:6379",
      encryption_key=os.getenv("CHECKPOINT_ENCRYPTION_KEY"),
  )
  ```

- [ ] **TTL configured** — Clean up old checkpoints
  ```python
  app = App(
      checkpoint_ttl_hours=168,  # 7 days
  )
  ```

**Reference**: [Architecture - Durability](../architecture.md)

---

### 5. Operations

**Why it matters**: Deploying, monitoring, and updating your agent should be automated and safe.

#### Essential

- [ ] **Containerized** — Consistent deployment
  ```dockerfile
  FROM python:3.11-slim
  COPY . /app
  RUN pip install fastagentic
  CMD ["fastagentic", "run", "--host", "0.0.0.0"]
  ```

- [ ] **Resource limits** — Prevent runaway consumption
  ```yaml
  resources:
    requests:
      memory: "256Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  ```

- [ ] **Readiness probe** — Don't route traffic until ready
  ```yaml
  readinessProbe:
    httpGet:
      path: /health
      port: 8000
    initialDelaySeconds: 5
  ```

#### Recommended

- [ ] **Horizontal scaling** — Handle traffic spikes
  ```yaml
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  spec:
    minReplicas: 2
    maxReplicas: 10
    metrics:
      - type: Resource
        resource:
          name: cpu
          targetAverageUtilization: 70
  ```

- [ ] **Rolling updates** — Zero-downtime deployments
  ```yaml
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  ```

- [ ] **CI/CD pipeline** — Automated testing and deployment
  ```yaml
  # .github/workflows/deploy.yml
  - run: fastagentic test contract
  - run: fastagentic test integration
  - run: kubectl apply -f k8s/
  ```

**Reference**: [Docker](../operations/deployment/docker.md), [Kubernetes](../operations/deployment/kubernetes.md)

---

### 6. Testing

**Why it matters**: Untested agents break in production. Contract tests catch regressions.

#### Essential

- [ ] **Contract tests** — API contracts are validated
  ```bash
  fastagentic test contract
  # ✓ /chat - input schema valid
  # ✓ /chat - output schema valid
  ```

- [ ] **Integration tests** — End-to-end flows work
  ```python
  async def test_chat_endpoint():
      response = await client.post("/chat", json={"message": "hello"})
      assert response.status_code == 200
  ```

#### Recommended

- [ ] **Load tests** — Know your limits
  ```bash
  locust -f loadtest.py --host http://localhost:8000
  # Requests/s, p99 latency, error rate
  ```

- [ ] **Evaluation baseline** — Track quality over time
  ```python
  BraintrustHook(
      project="my-agent",
      baseline_experiment="v1.0",  # Compare against baseline
  )
  ```

---

### 7. Documentation

**Why it matters**: Your team needs to operate and debug the system.

#### Essential

- [ ] **API documented** — Others can use your agent
  ```bash
  curl http://localhost:8000/openapi.json
  # or http://localhost:8000/docs (Swagger UI)
  ```

- [ ] **Runbook exists** — How to handle common issues
  - What to do when error rate spikes
  - How to restart stuck runs
  - How to roll back deployments

#### Recommended

- [ ] **Architecture diagram** — Visual overview
- [ ] **Incident playbook** — Step-by-step for outages
- [ ] **On-call guide** — Who to contact, escalation paths

---

## Quick Start: Minimum Viable Production

Copy this configuration as your production baseline:

```python
from fastagentic import App, agent_endpoint
from fastagentic.integrations.langfuse import LangfuseHook
from fastagentic.integrations.lakera import LakeraHook
from fastagentic.reliability import RetryPolicy, Timeout

app = App(
    title="My Production Agent",
    version="1.0.0",

    # Observability
    hooks=[
        LangfuseHook(),
        LakeraHook(on_detection="reject"),
    ],

    # Durability
    durable_store="redis://localhost:6379",

    # Auth
    auth=OIDCAuth(
        issuer=os.getenv("AUTH_ISSUER"),
        audience=os.getenv("AUTH_AUDIENCE"),
    ),

    # Rate limiting
    rate_limit=RateLimit(rpm=60, by="user"),
)

@agent_endpoint(
    path="/chat",
    runnable=...,
    durable=True,
    retry=RetryPolicy(max_attempts=3, backoff="exponential"),
    timeout=Timeout(total_ms=60000),
)
async def chat(message: str) -> str:
    pass
```

---

## Common Production Issues

### "My agent is slow"

1. Check trace latency in Langfuse
2. Is it the LLM call or tool execution?
3. Consider caching with Portkey
4. Review timeout configuration

### "I'm getting rate limited"

1. Check RPM/TPM usage in observability
2. Implement request queuing
3. Add model fallbacks via Portkey
4. Consider caching similar requests

### "Users are injecting prompts"

1. Enable Lakera with `on_detection="reject"`
2. Review flagged inputs in logs
3. Tune detection categories
4. Consider fail-closed (`on_failure="reject"`)

### "Checkpoints are filling up disk"

1. Configure TTL: `checkpoint_ttl_hours=168`
2. Set up periodic cleanup job
3. Archive old checkpoints to S3

---

## Next Steps

- [Deployment Guide](../operations/deployment/docker.md)
- [Reliability Patterns](../reliability.md)
- [Troubleshooting](../operations/runbook/troubleshooting.md)
