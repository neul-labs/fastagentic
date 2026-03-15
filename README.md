# FastAgentic

> **Build agents with anything. Ship them with FastAgentic.**

[![Tests](https://img.shields.io/badge/tests-608%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

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

## Installation

```bash
# Core installation
pip install fastagentic

# Or with uv (recommended)
uv add fastagentic

# With specific adapter support
uv add "fastagentic[pydanticai]"
uv add "fastagentic[langgraph]"
uv add "fastagentic[crewai]"

# With integrations
uv add "fastagentic[langfuse]"    # Observability
uv add "fastagentic[portkey]"    # AI Gateway
uv add "fastagentic[lakera]"     # Security guardrails

# Everything
uv add "fastagentic[all]"
```

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
# Start the HTTP server
fastagentic run

# Or run as MCP server (for Claude Desktop, etc.)
fastagentic mcp serve app:app
```

The command boots the FastAPI application, registers MCP discovery metadata, and exposes streaming endpoints for agent workflows.

## First-Class Integrations

FastAgentic integrates with best-of-breed tools for specialized concerns:

```python
from fastagentic import App
from fastagentic.integrations import (
    LangfuseIntegration,    # Observability & tracing
    PortkeyIntegration,     # AI Gateway (200+ LLMs)
    LakeraIntegration,      # Security guardrails
    Mem0Integration,        # Intelligent memory
)

app = App(
    title="My Agent",
    integrations=[
        LangfuseIntegration(
            public_key="pk-...",
            secret_key="sk-...",
        ),
        LakeraIntegration(
            api_key="lak-...",
            block_on_detect=True,  # Block prompt injection
        ),
        Mem0Integration(
            api_key="m0-...",
            auto_search=True,      # Auto-inject relevant memories
        ),
    ]
)
```

| Integration | Purpose | Features |
|-------------|---------|----------|
| **Langfuse** | Observability | Tracing, analytics, prompt management |
| **Portkey** | AI Gateway | 200+ LLMs, fallbacks, caching |
| **Lakera** | Security | Prompt injection, PII detection |
| **Mem0** | Memory | Semantic memory, auto-extraction |

## Reliability Patterns

Built-in patterns for production resilience:

```python
from fastagentic import App, RetryPolicy, CircuitBreaker, RateLimit

app = App(
    title="Resilient Agent",
    retry_policy=RetryPolicy(
        max_attempts=3,
        backoff="exponential",
        retry_on=["rate_limit", "timeout"],
    ),
    rate_limit=RateLimit(
        rpm=60,
        tpm=100000,
        by="user",
    ),
)
```

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
| `fastagentic run`           | Start the ASGI server                          |
| `fastagentic new`           | Scaffold a new application with sample modules  |
| `fastagentic info`          | Show information about the current app          |
| `fastagentic agent chat`    | Interactive CLI for agent testing               |
| `fastagentic agent query`   | Send single queries (scriptable)                |
| `fastagentic mcp serve`     | Run as MCP stdio server (for Claude Desktop)    |
| `fastagentic mcp schema`    | Print MCP schema (tools, resources, prompts)    |
| `fastagentic a2a card`      | Print A2A Agent Card                           |
| `fastagentic test contract` | Verify OpenAPI and MCP schema parity            |

## Roadmap Highlights

- **v0.1:** Core decorators, MCP schema fusion, adapters, SSE streaming
- **v0.2:** Reliability patterns, MCP stdio, first-class integrations (Langfuse, Portkey, Lakera, Mem0)
- **v0.3:** Policy engine, cost tracking, audit logging
- **v0.4:** Advanced prompt management, human-in-the-loop actions
- **v0.5:** Cluster orchestration, distributed checkpointing
- **v1.0:** Python SDK, PII detection, dashboard & metrics, production readiness checker
- **v1.1:** New adapters (Semantic Kernel, AutoGen, LlamaIndex, DSPy), template ecosystem
- **v1.2:** Interactive Agent CLI for testing and development

## Contributing

```bash
# Clone the repository
git clone https://github.com/fastagentic/fastagentic.git
cd fastagentic

# Install dependencies with uv
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check src/
```

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

**Developer Tools**
- [Agent CLI](docs/cli-agent.md) - Interactive testing and development

**Reference**
- [Runtime Architecture](docs/architecture.md)
- [Platform Services](docs/platform-services.md)
- [Roadmap](docs/roadmap.md)
