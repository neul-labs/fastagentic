# Distributed Tracing

Configure OpenTelemetry tracing for end-to-end visibility into agent execution.

## Quick Setup

```bash
export FASTAGENTIC_OTEL_ENDPOINT=http://otel-collector:4318
export FASTAGENTIC_OTEL_SERVICE_NAME=my-agent
export FASTAGENTIC_TRACES_ENABLED=true
```

## What Gets Traced

FastAgentic automatically instruments:

| Component | Span Name | Attributes |
|-----------|-----------|------------|
| HTTP Request | `HTTP {method} {path}` | method, path, status |
| Agent Run | `agent.run {endpoint}` | run_id, adapter, model |
| LLM Call | `llm.call {model}` | model, tokens, cost |
| Tool Call | `tool.call {name}` | tool, duration |
| Checkpoint | `checkpoint.write` | backend, size |
| Streaming | `stream.emit {type}` | event_type, run_id |

## Trace Structure

```
HTTP POST /chat
├── auth.validate
├── policy.check
└── agent.run (run_id=run-123)
    ├── llm.call (model=gpt-4)
    │   ├── tokens.prompt: 150
    │   └── tokens.completion: 200
    ├── tool.call (name=search)
    │   └── duration_ms: 250
    ├── checkpoint.write
    └── stream.emit (count=50)
```

## Configuration

### Environment Variables

```bash
# Required
FASTAGENTIC_OTEL_ENDPOINT=http://otel-collector:4318

# Optional
FASTAGENTIC_OTEL_SERVICE_NAME=my-agent
FASTAGENTIC_OTEL_SERVICE_VERSION=1.0.0
FASTAGENTIC_TRACES_ENABLED=true
FASTAGENTIC_OTEL_SAMPLE_RATE=1.0  # 1.0 = 100%
FASTAGENTIC_OTEL_PROPAGATORS=tracecontext,baggage
```

### Sampling

```bash
# Production: Sample 10% of traces
export FASTAGENTIC_OTEL_SAMPLE_RATE=0.1

# Always trace errors
export FASTAGENTIC_OTEL_SAMPLE_ERRORS=true

# Always trace slow requests (>5s)
export FASTAGENTIC_OTEL_SAMPLE_SLOW_THRESHOLD=5000
```

## OTEL Collector Setup

### otel-config.yaml

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

  attributes:
    actions:
      - key: environment
        value: production
        action: insert

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  # Or other backends
  otlp/honeycomb:
    endpoint: api.honeycomb.io:443
    headers:
      x-honeycomb-team: ${HONEYCOMB_API_KEY}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, attributes]
      exporters: [jaeger]
```

### Docker Compose

```yaml
services:
  otel-collector:
    image: otel/opentelemetry-collector:latest
    command: ["--config", "/etc/otel-config.yaml"]
    volumes:
      - ./otel-config.yaml:/etc/otel-config.yaml
    ports:
      - "4317:4317"  # gRPC
      - "4318:4318"  # HTTP

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14250:14250"  # gRPC
```

## Custom Spans

Add custom spans in your code:

```python
from fastagentic.telemetry import tracer

@agent_endpoint(path="/process")
async def process(input: Input) -> Output:
    with tracer.start_span("custom.preprocessing") as span:
        span.set_attribute("input_size", len(input.text))
        preprocessed = await preprocess(input)

    # Agent runs automatically traced
    result = await run_agent(preprocessed)

    with tracer.start_span("custom.postprocessing") as span:
        span.set_attribute("result_type", type(result).__name__)
        return postprocess(result)
```

## Trace Context Propagation

### Incoming Requests

FastAgentic extracts trace context from:
- `traceparent` header (W3C)
- `x-b3-*` headers (Zipkin B3)
- `uber-trace-id` header (Jaeger)

### Outgoing Requests

Trace context is automatically propagated to:
- LLM provider calls
- Tool HTTP requests
- Database queries

## Correlation IDs

Every request includes correlated IDs:

```json
{
  "trace_id": "abc123...",
  "span_id": "def456...",
  "run_id": "run-789",
  "user_id": "user-012"
}
```

Access in logs and metrics for cross-cutting analysis.

## Backend-Specific Setup

### Jaeger

```bash
export FASTAGENTIC_OTEL_ENDPOINT=http://jaeger:4318
```

### Honeycomb

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=YOUR_API_KEY"
```

### Datadog

```bash
export FASTAGENTIC_OTEL_ENDPOINT=http://datadog-agent:4318
```

### AWS X-Ray

```bash
export FASTAGENTIC_OTEL_ENDPOINT=http://aws-otel-collector:4318
```

## Debugging

### Verify Traces

```bash
# Check OTEL collector is receiving
curl http://localhost:4318/v1/traces -X POST -H "Content-Type: application/json" -d '{}'

# View in Jaeger UI
open http://localhost:16686
```

### Verbose Logging

```bash
export FASTAGENTIC_LOG_LEVEL=DEBUG
export OTEL_LOG_LEVEL=DEBUG
```

## Performance Impact

| Sampling Rate | Overhead | Use Case |
|---------------|----------|----------|
| 100% | ~5ms/request | Development, debugging |
| 10% | ~0.5ms/request | Production monitoring |
| 1% | Negligible | High-traffic production |

## Next Steps

- [Metrics Reference](metrics.md) - Prometheus metrics
- [Alerting](alerting.md) - Alert configuration
- [Environment Variables](../configuration/environment-vars.md) - All settings
