# FastAgentic Architecture

FastAgentic expands FastAPI into a unified runtime for REST, agent, MCP, and A2A workloads. The architecture layers declarative API surfaces, protocol bridges, and operational services on top of an asynchronous core.

## FastAgentic as a Deployment Layer

FastAgentic is the deployment layer for agentic applications. It transforms agents built with any framework into production-ready services.

**Key insight:** Agent frameworks (PydanticAI, LangChain, LangGraph, CrewAI) excel at orchestrating LLMs. FastAgentic adds what they don't ship: multi-protocol hosting (REST, SSE/WebSocket, MCP, A2A), authentication and policy middleware, observability, durable checkpoints, and unified governance.

**Build agents with anything. Ship them with FastAgentic.**

| Framework | What It Does | FastAgentic Adds |
|-----------|--------------|------------------|
| PydanticAI | Type-safe agent orchestration | REST + MCP + A2A hosting, governance, durability |
| LangGraph | Stateful graph workflows | Production deployment, node-level checkpoints |
| CrewAI | Multi-agent collaboration | Per-agent observability, task checkpointing |
| LangChain | Chains and runnables | MCP/A2A protocol, auth, policy enforcement |

## Protocol Stack

FastAgentic implements two complementary interoperability protocols:

```
┌─────────────────────────────────────────────────────────┐
│                    External Clients                      │
│         (LLM Hosts, Other Agents, Applications)         │
└─────────────────────────────────────────────────────────┘
          │ MCP (Tools/Resources)    │ A2A (Agent Tasks)
          ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Protocol Layer                          │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   MCP 2025-11-25    │  │      A2A v0.3           │  │
│  │   - Tools           │  │   - Agent Cards         │  │
│  │   - Resources       │  │   - Task Management     │  │
│  │   - Prompts         │  │   - Streaming           │  │
│  │   - Tasks           │  │   - Push Notifications  │  │
│  │   - Extensions      │  │   - gRPC Support        │  │
│  └─────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent Registry                         │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │  │External│ │
│  │(Pydantic)│  │(LangGraph│  │ (CrewAI) │  │ Agents │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬───┘ │
│       └─────────────┼─────────────┴──────────────┘     │
│                     ▼                                   │
│           Internal A2A Delegation                       │
└─────────────────────────────────────────────────────────┘
```

**MCP** (Model Context Protocol): Connects LLM applications to agent capabilities. Tools, resources, and prompts exposed to AI models.

**A2A** (Agent-to-Agent Protocol): Enables agent collaboration. Internal registry for local agents, external connectivity for cross-platform interoperability.

## Layered View

```
+----------------------------------------------+
|               Developer Surface              |
|----------------------------------------------|
|  Decorators (@tool, @resource, @prompt,      |
|             @agent_endpoint)                 |
|  CLI (fastagentic new/run/tail/test)         |
|  Project templates (pydanticai, langgraph,   |
|                     crewai, langchain)       |
+----------------------------------------------+
|             Protocol Layer                   |
|----------------------------------------------|
|  MCP 2025-11-25 (Tools, Resources, Tasks)    |
|  A2A v0.3 (Agent Cards, Task Delegation)     |
|  Agent Registry (internal + external)        |
|  Schema fusion (Pydantic -> OpenAPI/MCP/A2A) |
+----------------------------------------------+
|              Hooks & Integrations            |
|----------------------------------------------|
|  Lifecycle hooks (on_request, on_response,   |
|    on_llm_start, on_llm_end, on_tool_call)   |
|  Observability: Langfuse, Logfire, Datadog   |
|  Guardrails: Lakera, Guardrails AI, NeMo     |
|  Memory: Mem0, Zep, Redis                    |
|  Eval: Braintrust, LangSmith                 |
+----------------------------------------------+
|           Application Composition            |
|----------------------------------------------|
|  App (FastAPI extension)                     |
|  Auth + policy middleware (OAuth2/OIDC)      |
|  Rate limiting & cost guardrails             |
|  Streaming fabric (SSE, WebSocket, MCP)      |
+----------------------------------------------+
|             Adapter Integration              |
|----------------------------------------------|
|  PydanticAIAdapter                           |
|  LangGraphAdapter                            |
|  CrewAIAdapter                               |
|  LangChainAdapter                            |
|  BaseAdapter interface for custom frameworks |
+----------------------------------------------+
|               Persistence & IO               |
|----------------------------------------------|
|  AsyncSQLAlchemy (metadata & state)          |
|  Durable stores (Redis/Postgres/S3)          |
|  Background job runners                      |
|  LLM providers & external APIs               |
+----------------------------------------------+
```

## App Lifecycle

1. **Configuration:** `App` accepts metadata (title, version, tenant info) and service toggles (telemetry, durable store, auth providers).
2. **Initialization:** During startup, the app wires database connections, telemetry exporters, MCP transport, and job schedulers inside FastAPI lifespan events.
3. **Registration:** Decorators register routes, tools, prompts, and resources. Pydantic models are introspected once to produce OpenAPI paths and MCP schema entries.
4. **Execution:** Requests and agent runs share middlewares for authentication, policy enforcement, tracing, and logging.
5. **Shutdown:** Connections, checkpoint stores, and background workers close gracefully to preserve in-flight runs.

## Protocol Fusion

FastAgentic generates compatible schemas for all protocols from a single definition:

- **OpenAPI 3.1**: Every decorator contributes operations, security definitions, and examples to the REST API schema.
- **MCP 2025-11-25**: Tools, resources, and prompts exposed under `/mcp/schema` and stdio transport. Supports Tasks for long-running operations and Extensions for custom capabilities.
- **A2A v0.3**: Agent Cards auto-generated with skills, security schemes, and protocol interfaces. Available at `/.well-known/agent.json`.
- **Prompt Catalog**: Prompt decorators generate entries for both MCP prompts and A2A skill descriptions.

The fusion engine ensures schema parity by caching model signatures and reconciling differences on application startup.

| Source | OpenAPI | MCP | A2A |
|--------|---------|-----|-----|
| `@tool` | POST endpoint | Tool definition | - |
| `@resource` | GET endpoint | Resource definition | - |
| `@prompt` | - | Prompt template | - |
| `@agent_endpoint` | POST + SSE endpoints | Tool (optional) | Skill definition |

## Execution Modes

| Mode      | Description                                               |
| --------- | --------------------------------------------------------- |
| `asgi()`  | Runs FastAgentic as a FastAPI-compatible ASGI app.        |
| `stdio()` | Spins up an MCP-compliant stdio server for local agents.  |
| `hybrid`  | Default command (`fastagentic run`) serving both modes.   |

Each mode shares the same dependency graph, ensuring policy, auth, and telemetry configurations remain consistent.

## Streaming Fabric

FastAgentic delivers unified streaming semantics across transports:

- **Server-Sent Events** for HTTP clients, with event types such as `token`, `checkpoint`, and `tool_call`.
- **WebSocket** channels for bidirectional updates and human-in-the-loop interactions.
- **MCP streaming** events with identical payload schemas to SSE for tool and agent runs.

A stream multiplexer fans out events from agent adapters to all active transports while preserving ordering guarantees per `run_id`.

## Durable Run Management

Durable runs are tracked through:

- Unique `run_id` per agent invocation
- Checkpoint snapshots stored in Redis, PostgreSQL, or S3
- Idempotency keys to guard against duplicate submissions
- Resume endpoint (`POST /runs/{run_id}/resume`) backed by the same adapter interface

Background workers reconcile checkpoints with the runnable interface so agent frameworks can continue execution from the last committed node.

## Policy and Governance

Policy controls are enforced early in the request lifecycle:

- Rate limits and quotas keyed by user, tenant, and endpoint
- Scope-to-role mapping integrated with OAuth2/JWT claims
- Cost guardrails that throttle or downgrade models after thresholds
- Audit logging of run metadata, costs, and decision points

Policies apply uniformly whether the invocation arrives via REST, MCP, or background job replay.

## Hooks and Integrations

FastAgentic provides fine-grained hooks for integrating with external tools. Rather than building everything, we provide the integration points.

### Hook Lifecycle

```
on_request → on_llm_start → on_tool_call → on_tool_result → on_llm_end → on_response
                  │                                              │
                  ▼                                              ▼
           [Guardrail Hooks]                              [Eval Hooks]
            (Lakera, etc.)                               (Braintrust)
```

### Hook Types

| Hook | When | Integration Examples |
|------|------|---------------------|
| `on_request` | Request received | Auth enrichment, rate limiting |
| `on_llm_start` | Before LLM call | Lakera (prompt injection) |
| `on_llm_end` | After LLM call | Langfuse (tracing), cost tracking |
| `on_tool_call` | Before tool | Tool authorization |
| `on_tool_result` | After tool | Result caching |
| `on_response` | Response ready | Guardrails AI (output validation) |
| `on_checkpoint` | Checkpoint created | Durability logging |
| `on_node_enter/exit` | LangGraph nodes | Node-level tracing |

### Integration Philosophy

| Category | FastAgentic Builds | Integrates With |
|----------|-------------------|-----------------|
| Observability | OTEL spans (basic) | Langfuse, Logfire, Datadog |
| Guardrails | Hook interface | Lakera, Guardrails AI, NeMo |
| Memory | Redis (basic) | Mem0, Zep |
| Eval | Hook interface | Braintrust, LangSmith |
| LLM Gateway | Simple limiter | Portkey, LiteLLM |

See [Hooks Architecture](hooks.md) for detailed documentation.

## Observability

OpenTelemetry instrumentation spans:

- Request lifecycle (FastAPI routes, WebSocket messages, SSE emits)
- Database queries and queue interactions
- Agent framework node execution
- Token usage and cost metrics

Structured logs correlate with trace and span IDs, allowing cross-cutting analysis of agent decisions, policy enforcement, and resource consumption.

For advanced observability, integrate with:
- **Langfuse** — LLM-specific tracing, prompt analytics
- **Logfire** — PydanticAI native, structured logging
- **Datadog** — Full APM, dashboards, alerting

See [Integrations Guide](integrations/index.md) for setup instructions.
