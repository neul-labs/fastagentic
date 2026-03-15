# Platform Services

FastAgentic ships operational primitives that keep REST, MCP, and agent workflows aligned. This guide covers authentication, policy enforcement, observability, durability, and streaming.

## What Agent Frameworks Don't Ship

Agent frameworks (PydanticAI, LangGraph, CrewAI, LangChain) excel at orchestrating LLM interactions. But deploying them to production requires infrastructure they don't provide:

| Capability | Raw Framework | FastAgentic |
|------------|---------------|-------------|
| **Authentication** | DIY JWT validation | Built-in OAuth2/OIDC with MCP alignment |
| **Rate Limiting** | External proxy required | Native per-user/tenant/endpoint limits |
| **Cost Controls** | Manual tracking | Automatic guards with downgrade/reject |
| **Audit Logging** | Custom implementation | Structured append-only audit trails |
| **Checkpointing** | Framework-specific | Unified durable store (Redis/Postgres/S3) |
| **Multi-Protocol** | REST only | REST + SSE + WebSocket + MCP from one definition |
| **Observability** | Manual instrumentation | OTEL spans, metrics, structured logs included |

FastAgentic adds these capabilities without changing your agent code. Wrap your existing agent with an adapter and deploy with production-grade governance.

## Authentication and Authorization

> **Without FastAgentic:** You write JWT validation middleware, manage JWKS refresh, handle token expiry, and duplicate this logic for every transport (REST, WebSocket, MCP stdio).

FastAgentic provides unified authentication across all transports with a single configuration.

### Supported Modes

- **OAuth2/OIDC bearer tokens** using JWT access tokens
- **Cookie sessions** for browser-based flows (optional)
- **MCP HTTP auth** with `WWW-Authenticate` headers aligned to OAuth scopes
- **Stdio credentials** discovered through environment variables or runtime configuration

### Configuring OIDC

```python
from fastagentic import App
from fastagentic.auth import configure_oidc

app = App(
    title="Support Triage",
    version="1.0.0",
    oidc_issuer="https://auth.mycompany.com",
)

configure_oidc(
    app,
    audience="support-triage",
    jwks_url="https://auth.mycompany.com/.well-known/jwks.json",
    required_scopes=["openid", "profile"],
)
```

`current_user()` returns a `UserContext` object containing user, tenant, and scope metadata that can be injected into routes and agent endpoints.

## Policy Enforcement

> **Without FastAgentic:** You build rate limiting with Redis scripts, implement quota tracking with custom database tables, and hope your cost controls catch runaway API calls before the bill arrives.

FastAgentic applies policy decisions consistently across transports.

| Policy            | Description                                        |
| ----------------- | -------------------------------------------------- |
| Quotas            | Limit runs or tokens per user/tenant within a SLA  |
| Rate limits       | Apply burst/sustained limits per endpoint          |
| Role mapping      | Translate scopes to application-level roles        |
| Cost guards       | Downgrade or reject runs after budget exhaustion   |
| Tenant isolation  | Assign separate durable stores per tenant          |
| Audit trails      | Persist append-only logs of significant events     |

Policies can hook into lifecycle events (`before_run`, `after_run`, `on_checkpoint`) to enforce or log decisions.

## Observability

> **Without FastAgentic:** You manually instrument LLM calls with OpenTelemetry, build custom span hierarchies for agent nodes, and correlate logs across async boundaries. Token usage? That's another custom integration.

FastAgentic provides comprehensive observability out of the box.

### Tracing

OpenTelemetry instrumentation provides spans for:

- FastAPI routes and dependencies
- WebSocket events
- Agent adapter node execution
- Database queries and durable store operations

```python
from fastagentic.telemetry import configure_otel

configure_otel(
    app,
    service_name="support-triage",
    exporter_endpoint="http://localhost:4318",
)
```

### Metrics

Default metrics include:

- RED metrics (rate, errors, duration) per endpoint
- Token usage and cost per agent run
- Connection pool statistics
- Background worker queue depth

Metrics are exported via OTEL or Prometheus-compatible endpoints.

### Logging

- Structured JSON with `trace_id`, `span_id`, `user_id`, and `run_id`
- PII redaction filters configurable per field
- Log sampling policies for high-volume streaming events

## Durability

> **Without FastAgentic:** LangGraph has its own checkpointer. CrewAI has task persistence. PydanticAI has none. You end up with three different checkpoint formats, three different storage backends, and no unified way to inspect or resume runs.

FastAgentic provides a unified durable store that works with any adapter. Durable runs persist execution state to support resumable workflows and post-mortem analysis.

| Backend   | Use Case                                      |
| --------- | ---------------------------------------------- |
| Redis     | Fast local development, short-lived checkpoints |
| Postgres  | Transactional durability with relational queries |
| S3        | Long-term archival of checkpoints and artifacts |

Runs are identified by a unique `run_id`. Each checkpoint captures:

- Adapter-specific state (e.g., LangGraph node state)
- Emitted events up to the checkpoint boundary
- Cost and token counters

Idempotency keys ensure replayed requests do not create duplicate runs.

## Streaming

> **Without FastAgentic:** PydanticAI streams tokens. LangGraph streams node events. CrewAI streams task updates. You write three different SSE implementations, each with different event schemas, and none of them work over MCP.

FastAgentic unifies streaming semantics across transports:

| Event Type    | Payload Fields                                                |
| ------------- | ------------------------------------------------------------- |
| `token`       | `{"run_id": "...", "content": "...", "delta": {...}}`         |
| `node_start`  | `{"run_id": "...", "node": "...", "timestamp": ...}`          |
| `node_end`    | `{"run_id": "...", "node": "...", "result": {...}}`           |
| `tool_call`   | `{"run_id": "...", "tool": "...", "args": {...}}`             |
| `tool_result` | `{"run_id": "...", "tool": "...", "output": {...}}`           |
| `checkpoint`  | `{"run_id": "...", "checkpoint_id": "...", "metadata": {...}}`|
| `cost`        | `{"run_id": "...", "tokens": {...}, "amount": {...}}`         |

### Transports

- **SSE**: Default for REST clients. Each event is serialized as an SSE frame with `event` and `data`.
- **WebSocket**: Supports bidirectional messaging. Clients can submit control events (e.g., pause, resume) with guaranteed acknowledgment frames.
- **MCP streaming**: Mirrors SSE payloads over the MCP event protocol for stdio or HTTP agents.

### Ordering Guarantees

- Events are emitted in the order produced by the underlying runnable.
- Each transport maintains per-run ordering.
- Checkpoint events include the offset to resume streaming from a specific point in history.

## Cost Tracking

> **Without FastAgentic:** You parse provider responses for token counts, maintain a pricing table for each model, aggregate costs across retries and tool calls, and build alerts before your CFO asks why the API bill tripled.

Cost collectors aggregate token usage and billable metrics per run, user, and tenant. Integrations:

- Model pricing catalogs configurable per deployment
- Exporters to billing systems or FinOps dashboards
- Alerts when thresholds are exceeded

## Audit Logging

> **Without FastAgentic:** You build custom logging for every decision point, ensure immutability for compliance, correlate entries across services, and pray the audit trail survives the SOC2 review.

Audit entries capture:

- Actor (user, tenant, service account)
- Action (`run.start`, `policy.block`, `checkpoint.write`, etc.)
- Context (scopes, model, tool name)
- Decision metadata (allow, deny, downgrade, reason)

Audit logs feed compliance reports and anomaly detection pipelines.

## Summary: The Deployment Layer Value

These platform services represent what you'd otherwise build yourself:

```
┌─────────────────────────────────────────────────────────┐
│                    Your Agent Code                       │
│         (PydanticAI / LangGraph / CrewAI / etc.)        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAgentic Platform                    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │   Auth   │  │  Policy  │  │Durability│  │ Observe │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │Streaming │  │  Costs   │  │  Audit   │  │Multi-   │ │
│  │ Fabric   │  │ Tracking │  │ Logging  │  │Protocol │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────┴─────────────┐
              │    REST    │    MCP       │
              │    SSE     │   WebSocket  │
              └─────────────┴─────────────┘
```

**Build agents with anything. Ship them with FastAgentic.**

## Next Steps

- [Adapters Guide](adapters/index.md) - Wrap your framework
- [Operations Guide](operations/index.md) - Deploy to production
- [Getting Started](getting-started.md) - Quick start tutorial

