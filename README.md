# FastAgentic

> **Build agents with anything. Ship them with FastAgentic.**

FastAgentic is the **deployment layer** for agentic applications. It transforms agents built with PydanticAI, LangChain, LangGraph, or CrewAI into production-ready services with REST, MCP, and streaming interfaces—plus authentication, policy, observability, and durability baked in.

**FastAgentic is not an agent framework. It deploys them.**

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   Your Agent Logic      │     │      FastAgentic        │     REST, MCP, SSE, WebSocket
│   ─────────────────     │ ──► │   (Deployment Layer)    │ ──► Auth, Policy, Telemetry
│   PydanticAI            │     │                         │     Durability, Cost Control
│   LangGraph             │     │   One decorator =       │
│   CrewAI                │     │   Production service    │
│   LangChain             │     │                         │
└─────────────────────────┘     └─────────────────────────┘
```

## Supported Adapters

| Framework | Adapter | Best For |
|-----------|---------|----------|
| **PydanticAI** | `PydanticAIAdapter` | Type-safe agents, structured outputs |
| **LangGraph** | `LangGraphAdapter` | Stateful graph workflows, cycles |
| **CrewAI** | `CrewAIAdapter` | Multi-agent collaboration |
| **LangChain** | `LangChainAdapter` | Chains, LCEL runnables |
| **Custom** | `BaseAdapter` | Your own framework |

## What You Get

| Without FastAgentic | With FastAgentic |
|---------------------|------------------|
| Write REST endpoints manually | `@tool`, `@agent_endpoint` decorators |
| Build MCP server from scratch | Automatic MCP schema fusion |
| Implement auth middleware | OAuth2/OIDC built-in |
| Add streaming yourself | SSE/WebSocket/MCP streaming |
| Build checkpoint system | Redis/Postgres/S3 durability |
| Instrument observability | OTEL traces and metrics |
| Track costs manually | Automatic cost logging |
| Write deployment scripts | `fastagentic run` |

## Should You Use FastAgentic?

**Yes, if you:**
- Need to expose agents via REST and/or MCP protocols
- Require production governance (auth, policy, cost control, audit)
- Want to use multiple agent frameworks behind a unified interface
- Care about durability, streaming, and observability

**Not yet, if you:**
- Are experimenting with agent logic locally
- Have a single internal consumer with no governance needs
- Need an agent framework first (use PydanticAI, LangChain, etc.)

## Quick Start

```python
from fastagentic import App, agent_endpoint, prompt, resource, tool
from fastagentic.adapters.langgraph import LangGraphAdapter
from models import TicketIn, TicketOut, triage_graph

app = App(
    title="Support Triage",
    version="1.0.0",
    oidc_issuer="https://auth.mycompany.com",
    telemetry=True,
    durable_store="redis://localhost:6379",
)


@tool(
    name="summarize_text",
    description="Summarize ticket text into key points",
    scopes=["summaries:run"],
)
async def summarize(text: str) -> str:
    ...


@resource(name="run-status", uri="/runs/{run_id}", cache_ttl=60)
async def fetch_run(run_id: str) -> dict:
    ...


@prompt(name="triage_prompt", description="System prompt for support ticket triage")
def triage_prompt() -> str:
    return """
    You are a support triage assistant.
    Ask clarifying questions when urgency or impact is ambiguous.
    """


@agent_endpoint(
    path="/triage",
    runnable=LangGraphAdapter(triage_graph),
    input_model=TicketIn,
    output_model=TicketOut,
    stream=True,
    durable=True,
    mcp_tool="triage_ticket",      # Expose as MCP tool
    a2a_skill="support-triage",    # Expose as A2A skill
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

Run the framework with both ASGI and MCP entry points:

```bash
fastagentic run
```

The command boots the FastAPI application, registers MCP discovery metadata, and exposes streaming endpoints for agent workflows.

## How It Fits Together

```
+-----------------------------------------------------------+
|                       FastAgentic                         |
|-----------------------------------------------------------|
|  @tool, @agent_endpoint, @resource, @prompt decorators     |
|  Schema fusion (Pydantic -> OpenAPI + MCP + A2A)          |
|  Unified Auth (OIDC/JWT -> MCP + A2A Auth Bridge)         |
|  Observability (OTEL traces, metrics, cost logs)          |
|  Policy (rate limits, quotas, roles, tenancy)             |
|  Streaming (SSE, WebSocket, MCP, gRPC)                    |
|  Durable jobs (Redis/Postgres checkpoints)                |
|  Agent Registry (internal + external A2A agents)          |
+-----------------------------------------------------------+
|    Adapter Layer (PydanticAI, LangGraph, CrewAI, etc.)    |
+-----------------------------------------------------------+
|          Core Stack (FastAPI, AsyncSQLAlchemy, OTEL SDK)  |
+-----------------------------------------------------------+
```

## Protocol Alignment

- MCP specification (2025-11-25) with Tasks, Extensions, and OAuth support
- A2A protocol (v0.3) for agent-to-agent collaboration and discovery
- OpenAPI 3.1 JSON schemas derived from Pydantic models
- OAuth2/OIDC bearer authentication with scoped policies
- Streaming surfaces: Server-Sent Events, WebSocket, and MCP event streaming

## Developer Tooling

| Command                     | Description                                     |
| --------------------------- | ----------------------------------------------- |
| `fastagentic new`           | Scaffold a new application with sample modules  |
| `fastagentic add endpoint`  | Generate a tool or agent endpoint stub          |
| `fastagentic run`           | Start the ASGI server and MCP stdio transport   |
| `fastagentic tail`          | Stream live traces, events, and cost telemetry  |
| `fastagentic test contract` | Verify OpenAPI and MCP schema parity            |

## Roadmap Highlights

- **v0.1:** Core decorators, MCP schema fusion, LangChain adapter, SSE streaming
- **v0.2:** LangGraph & CrewAI adapters, resilient run storage
- **v0.3:** Policy engine, cost tracking, audit logging
- **v0.4:** Advanced prompt management, human-in-the-loop actions
- **v0.5:** Cluster orchestration, distributed checkpointing
- **v1.0:** Full MCP compliance, SDK, and operational dashboard

## Learn More

**Getting Started**
- [Getting Started Guide](docs/getting-started.md)
- [Why FastAgentic?](docs/why-fastagentic.md)
- [Comparison with Alternatives](docs/comparison.md)

**Templates**
- [Templates Overview](docs/templates/index.md)
- [PydanticAI Template](docs/templates/pydanticai.md)
- [LangGraph Template](docs/templates/langgraph.md)
- [CrewAI Template](docs/templates/crewai.md)
- [Contributing Templates](docs/templates/contributing.md)

**Adapters**
- [Adapters Overview](docs/adapters/index.md)
- [PydanticAI Adapter](docs/adapters/pydanticai.md)
- [LangGraph Adapter](docs/adapters/langgraph.md)
- [CrewAI Adapter](docs/adapters/crewai.md)
- [LangChain Adapter](docs/adapters/langchain.md)
- [Custom Adapters](docs/adapters/custom.md)

**Protocols**
- [Protocol Overview](docs/protocols/index.md)
- [MCP Implementation](docs/protocols/mcp.md)
- [A2A Integration](docs/protocols/a2a.md)

**Operations**
- [Operations Guide](docs/operations/index.md)
- [Deployment (Docker, K8s, Serverless)](docs/operations/deployment/)
- [Configuration Reference](docs/operations/configuration/)
- [Observability](docs/operations/observability/)
- [Security & Compliance](docs/operations/security/)

**Reference**
- [Runtime Architecture](docs/architecture.md)
- [Platform Services](docs/platform-services.md)
- [Roadmap](docs/roadmap.md)
