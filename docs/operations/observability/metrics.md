# Metrics Reference

FastAgentic exports Prometheus-compatible metrics for monitoring agent performance, costs, and reliability.

## Metrics Endpoint

```bash
# Prometheus format
curl http://localhost:8000/metrics

# JSON format
curl http://localhost:8000/metrics?format=json
```

## Available Metrics

### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_requests_total` | Counter | `method`, `path`, `status` | Total HTTP requests |
| `fastagentic_request_duration_seconds` | Histogram | `method`, `path` | Request latency |
| `fastagentic_requests_in_progress` | Gauge | `method`, `path` | Active requests |

### Run Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_runs_total` | Counter | `endpoint`, `status`, `adapter` | Total agent runs |
| `fastagentic_runs_active` | Gauge | `endpoint`, `adapter` | Currently active runs |
| `fastagentic_run_duration_seconds` | Histogram | `endpoint`, `adapter` | Run execution time |
| `fastagentic_runs_resumed_total` | Counter | `endpoint` | Runs resumed from checkpoint |

### Token Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_tokens_total` | Counter | `endpoint`, `type`, `model` | Total tokens used |
| `fastagentic_tokens_prompt` | Counter | `endpoint`, `model` | Input tokens |
| `fastagentic_tokens_completion` | Counter | `endpoint`, `model` | Output tokens |

### Cost Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_cost_usd_total` | Counter | `endpoint`, `model`, `tenant` | Total cost in USD |
| `fastagentic_cost_per_run_usd` | Histogram | `endpoint`, `model` | Cost per run |

### Checkpoint Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_checkpoints_total` | Counter | `endpoint`, `backend` | Checkpoints created |
| `fastagentic_checkpoint_size_bytes` | Histogram | `endpoint` | Checkpoint size |
| `fastagentic_checkpoint_duration_seconds` | Histogram | `backend` | Checkpoint write time |

### Streaming Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_stream_connections` | Gauge | `transport` | Active stream connections |
| `fastagentic_stream_events_total` | Counter | `endpoint`, `event_type` | Events streamed |
| `fastagentic_stream_duration_seconds` | Histogram | `endpoint` | Stream duration |

### Policy Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_rate_limit_hits_total` | Counter | `endpoint`, `user` | Rate limit triggers |
| `fastagentic_quota_usage_ratio` | Gauge | `endpoint`, `user`, `quota_type` | Quota utilization |
| `fastagentic_policy_decisions_total` | Counter | `endpoint`, `decision` | Allow/deny/downgrade |

### Health Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastagentic_durable_store_up` | Gauge | `backend` | Store connectivity |
| `fastagentic_durable_store_latency_seconds` | Histogram | `backend`, `operation` | Store latency |

## Prometheus Configuration

### prometheus.yml

```yaml
scrape_configs:
  - job_name: 'fastagentic'
    scrape_interval: 15s
    static_configs:
      - targets: ['my-agent:8000']
    metrics_path: /metrics
```

### ServiceMonitor (Kubernetes)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fastagentic
spec:
  selector:
    matchLabels:
      app: my-agent
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

## Grafana Dashboards

### Overview Dashboard

```json
{
  "panels": [
    {
      "title": "Requests per Second",
      "targets": [{
        "expr": "rate(fastagentic_requests_total[5m])"
      }]
    },
    {
      "title": "Error Rate",
      "targets": [{
        "expr": "rate(fastagentic_requests_total{status=~\"5..\"}[5m]) / rate(fastagentic_requests_total[5m])"
      }]
    },
    {
      "title": "P99 Latency",
      "targets": [{
        "expr": "histogram_quantile(0.99, rate(fastagentic_request_duration_seconds_bucket[5m]))"
      }]
    },
    {
      "title": "Active Runs",
      "targets": [{
        "expr": "sum(fastagentic_runs_active)"
      }]
    }
  ]
}
```

### Cost Dashboard

```json
{
  "panels": [
    {
      "title": "Daily Cost by Model",
      "targets": [{
        "expr": "sum(increase(fastagentic_cost_usd_total[24h])) by (model)"
      }]
    },
    {
      "title": "Cost per User",
      "targets": [{
        "expr": "sum(increase(fastagentic_cost_usd_total[24h])) by (tenant)"
      }]
    },
    {
      "title": "Token Usage",
      "targets": [{
        "expr": "sum(rate(fastagentic_tokens_total[1h])) by (type)"
      }]
    }
  ]
}
```

## Key Queries

### RED Metrics

```promql
# Rate
rate(fastagentic_requests_total[5m])

# Errors
rate(fastagentic_requests_total{status=~"5.."}[5m])

# Duration (P50, P95, P99)
histogram_quantile(0.50, rate(fastagentic_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(fastagentic_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(fastagentic_request_duration_seconds_bucket[5m]))
```

### Cost Analysis

```promql
# Total daily cost
sum(increase(fastagentic_cost_usd_total[24h]))

# Cost by endpoint
sum(increase(fastagentic_cost_usd_total[24h])) by (endpoint)

# Average cost per run
sum(rate(fastagentic_cost_usd_total[1h])) / sum(rate(fastagentic_runs_total[1h]))
```

### Capacity Planning

```promql
# Runs per minute
sum(rate(fastagentic_runs_total[5m])) * 60

# Average run duration
rate(fastagentic_run_duration_seconds_sum[5m]) / rate(fastagentic_run_duration_seconds_count[5m])

# Concurrent capacity
sum(fastagentic_runs_active) / count(fastagentic_runs_active)
```

## Custom Metrics

Add custom metrics in your application:

```python
from fastagentic.telemetry import metrics

# Counter
my_counter = metrics.counter(
    "my_custom_events_total",
    description="Custom event counter",
    labels=["event_type"]
)
my_counter.add(1, {"event_type": "success"})

# Histogram
my_histogram = metrics.histogram(
    "my_custom_duration_seconds",
    description="Custom duration",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0]
)
my_histogram.record(1.5)
```

## Next Steps

- [Tracing](tracing.md) - Distributed tracing setup
- [Alerting](alerting.md) - Alert rules and runbooks
- [Environment Variables](../configuration/environment-vars.md) - Configuration
