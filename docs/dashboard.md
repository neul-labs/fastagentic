# Dashboard & Metrics

FastAgentic provides built-in observability with stats collection, Prometheus metrics, and a dashboard API.

## Quick Start

```python
from fastagentic import DashboardAPI

# Create dashboard
dashboard = DashboardAPI()

# Record run
await dashboard.record_run_start("run-123", "/chat", input_tokens=50)
await dashboard.record_run_end("run-123", "completed", output_tokens=100)

# Get stats
summary = dashboard.get_summary()
print(summary)
```

## Stats Collection

### Basic Usage

```python
from fastagentic import StatsCollector

collector = StatsCollector()

# Start a run
run = await collector.start_run(
    run_id="run-123",
    endpoint="/chat",
    input_tokens=50,
)

# Complete the run
completed = await collector.complete_run(
    run_id="run-123",
    status="completed",
    output_tokens=100,
    cost=0.05,
)

# Get recent runs
recent = collector.get_recent_runs(limit=10)
```

### Run Statistics

```python
from fastagentic import RunStats

# Create run stats
stats = RunStats(run_id="run-123", endpoint="/chat")

# Complete with details
stats.complete(
    status="completed",
    output_tokens=100,
    cost=0.05,
)

# Access stats
print(stats.duration_ms)
print(stats.status)
print(stats.to_dict())
```

### Endpoint Statistics

```python
from fastagentic import EndpointStats

stats = collector.get_endpoint_stats("/chat")

print(f"Total requests: {stats.total_requests}")
print(f"Successful: {stats.successful_requests}")
print(f"Error rate: {stats.error_rate:.2%}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Total cost: ${stats.total_cost:.4f}")
print(f"P50 latency: {stats.p50_duration_ms}ms")
print(f"P95 latency: {stats.p95_duration_ms}ms")
```

### System Statistics

```python
from fastagentic import SystemStats

stats = collector.get_system_stats()

print(f"Uptime: {stats.uptime_seconds}s")
print(f"Total requests: {stats.total_requests}")
print(f"Active requests: {stats.active_requests}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Total cost: ${stats.total_cost:.4f}")
```

### Summary

```python
summary = collector.get_summary()

# Summary includes:
# - system: System-wide stats
# - endpoints: Per-endpoint stats
# - recent_runs: Last N runs
```

## Prometheus Metrics

### MetricsRegistry

```python
from fastagentic import MetricsRegistry

# Create registry with prefix
registry = MetricsRegistry(prefix="fastagentic")

# Create metrics
requests = registry.counter("requests_total", "Total requests")
active = registry.gauge("active_requests", "Active requests")
duration = registry.histogram("request_duration", "Request duration")
```

### Counter

```python
from fastagentic import Counter

# Basic counter
counter = Counter("requests", "Total requests")
counter.inc()
counter.add(5)
print(counter.get())  # 6

# With labels
counter = Counter("requests", "Total requests", ["endpoint", "status"])
counter.inc(endpoint="/chat", status="success")
counter.inc(endpoint="/chat", status="error")

print(counter.get(endpoint="/chat", status="success"))  # 1
```

### Gauge

```python
from fastagentic import Gauge

gauge = Gauge("active_connections", "Active connections")

gauge.set(5)
gauge.inc()
gauge.dec()

print(gauge.get())  # 5
```

### Histogram

```python
from fastagentic import Histogram

# Default buckets
hist = Histogram("duration", "Request duration in seconds")

# Custom buckets
hist = Histogram(
    "duration",
    "Request duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# Record observations
hist.observe(0.05)
hist.observe(0.12)
hist.observe(0.8)

print(hist.get_count())  # 3
print(hist.get_sum())    # 0.97
```

### Prometheus Export

```python
from fastagentic import MetricsRegistry, PrometheusExporter

registry = MetricsRegistry(prefix="myapp")

# Create and update metrics
requests = registry.counter("requests_total", "Total requests")
requests.inc()

# Export in Prometheus format
exporter = PrometheusExporter(registry)
output = exporter.export()

print(output)
# # HELP myapp_requests_total Total requests
# # TYPE myapp_requests_total counter
# myapp_requests_total 1
```

## Dashboard API

### Configuration

```python
from fastagentic import DashboardConfig, DashboardAPI

config = DashboardConfig(
    enabled=True,
    path_prefix="/dashboard",
    require_auth=True,
    allowed_origins=["http://localhost:3000"],
)

dashboard = DashboardAPI(config=config)
```

### Recording Runs

```python
dashboard = DashboardAPI()

# Start run
await dashboard.record_run_start(
    run_id="run-123",
    endpoint="/chat",
    input_tokens=50,
)

# End run
await dashboard.record_run_end(
    run_id="run-123",
    status="completed",
    output_tokens=100,
    cost=0.05,
)
```

### Getting Data

```python
# Get specific run
run = dashboard.get_run("run-123")

# Get recent runs
runs = dashboard.get_recent_runs(limit=20)

# Get summary
summary = dashboard.get_summary()

# Get system stats
stats = dashboard.get_system_stats()

# Get Prometheus metrics
metrics = dashboard.get_metrics()

# Health check
health = dashboard.get_health()
```

### Routes

```python
routes = dashboard.get_routes()

# Returns route definitions for integration:
# [
#     {"path": "/health", "method": "GET", "handler": ...},
#     {"path": "/metrics", "method": "GET", "handler": ...},
#     {"path": "/stats", "method": "GET", "handler": ...},
#     {"path": "/runs", "method": "GET", "handler": ...},
#     {"path": "/runs/{run_id}", "method": "GET", "handler": ...},
# ]
```

## Integration with FastAPI

```python
from fastapi import FastAPI
from fastagentic import DashboardAPI, DashboardConfig

app = FastAPI()
dashboard = DashboardAPI(DashboardConfig(path_prefix="/api/dashboard"))

# Add dashboard routes
for route in dashboard.get_routes():
    app.add_api_route(
        route["path"],
        route["handler"],
        methods=[route["method"]],
    )

# Use in endpoints
@app.post("/chat")
async def chat(message: str):
    run_id = generate_run_id()

    await dashboard.record_run_start(run_id, "/chat")

    try:
        result = await process_message(message)
        await dashboard.record_run_end(run_id, "completed")
        return result
    except Exception as e:
        await dashboard.record_run_end(run_id, "failed", error=str(e))
        raise
```

## Time Series Data

```python
from fastagentic import TimeSeriesPoint

# Time series data for charting
point = TimeSeriesPoint(
    timestamp=datetime.now(),
    value=42.5,
    labels={"endpoint": "/chat"},
)
```

## Health Endpoint

```python
health = dashboard.get_health()

# Returns:
# {
#     "status": "healthy",
#     "uptime_seconds": 3600,
#     "total_requests": 1000,
#     "active_requests": 5,
# }
```

## Metrics Endpoint

```python
metrics = dashboard.get_metrics()

# Returns Prometheus-formatted metrics:
# # HELP fastagentic_requests_total Total requests
# # TYPE fastagentic_requests_total counter
# fastagentic_requests_total 1000
# ...
```

## Best Practices

1. **Use labels wisely**: Too many label combinations can cause cardinality issues
2. **Set appropriate buckets**: Histogram buckets should match your latency distribution
3. **Record both starts and ends**: Track active requests accurately
4. **Include costs**: Track token usage and costs per endpoint
5. **Export to Prometheus**: Use the exporter for production monitoring
