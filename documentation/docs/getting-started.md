# Getting Started with FastAgentic

FastAgentic is the deployment layer for agentic applications. This guide walks through installing the framework, choosing an adapter for your agent framework, and deploying your first agent as a production-ready service.

## Prerequisites

- Python 3.10 or later
- An existing agent (PydanticAI, LangGraph, CrewAI, or LangChain)
- Redis or PostgreSQL for durable storage (Redis recommended for development)
- Optional: OIDC provider for authentication
- Optional: OpenTelemetry collector for observability

## Installation

```bash
pip install fastagentic
```

Install with your preferred adapter:

```bash
# PydanticAI (recommended for new projects)
pip install fastagentic[pydanticai]

# LangGraph (for stateful workflows)
pip install fastagentic[langgraph]

# CrewAI (for multi-agent collaboration)
pip install fastagentic[crewai]

# LangChain (for existing chains)
pip install fastagentic[langchain]

# All adapters + observability
pip install fastagentic[all,otel]
```

## Choose Your Adapter

| If you have... | Install | Import |
|----------------|---------|--------|
| PydanticAI `Agent` | `fastagentic[pydanticai]` | `from fastagentic.adapters.pydanticai import PydanticAIAdapter` |
| LangGraph `StateGraph` | `fastagentic[langgraph]` | `from fastagentic.adapters.langgraph import LangGraphAdapter` |
| CrewAI `Crew` | `fastagentic[crewai]` | `from fastagentic.adapters.crewai import CrewAIAdapter` |
| LangChain `Runnable` | `fastagentic[langchain]` | `from fastagentic.adapters.langchain import LangChainAdapter` |

See the [Adapters Guide](adapters/index.md) for detailed documentation on each adapter.

## Scaffold a New Project

```bash
fastagentic new support-triage
cd support-triage
```

The scaffold generates:

- `app.py` with a preconfigured `App`
- Sample models in `models/`
- Example decorators in `endpoints/`
- `config/settings.yaml` for environment-specific settings
- `tests/test_contracts.py` for schema parity checks

## Anatomy of an App

```python
from fastagentic import App
from fastagentic.auth import configure_oidc
from fastagentic.telemetry import configure_otel

app = App(
    title="Support Triage",
    version="1.0.0",
    oidc_issuer="https://auth.mycompany.com",
    telemetry=True,
    durable_store="redis://localhost:6379",
)

configure_oidc(app)
configure_otel(app)
```

The application manages lifespan events for database connections, durable run stores, telemetry exporters, and MCP discovery endpoints. When the ASGI server starts, MCP metadata becomes available at `/mcp/schema`, `/mcp/discovery`, and `/mcp/health`.

## Define Endpoints and Tools

```python
from fastagentic import agent_endpoint, prompt, resource, tool
from fastagentic.adapters.langgraph import LangGraphAdapter
from models import TicketIn, TicketOut
from workflows import triage_graph


@tool(name="summarize_text", description="Summarize text into key points")
async def summarize(text: str) -> str:
    ...


@resource(name="run-status", uri="runs/{run_id}", cache_ttl=60)
async def fetch_run(run_id: str) -> dict:
    ...


@prompt(name="triage_prompt", description="System prompt for support ticket triage")
def triage_prompt() -> str:
    return """
    You are a support triage assistant.
    Ask clarifying questions before prioritizing a ticket.
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

Each decorator registers both REST and MCP surfaces, reusing the same Pydantic models for schema validation and documentation.

## Run the Application

```bash
fastagentic run --reload
```

The command launches:

- ASGI server: REST routes, SSE streaming, WebSocket endpoints
- MCP stdio transport: tool, prompt, and resource discovery
- Background runners for durable workflows

Browse `http://localhost:8000/docs` for the OpenAPI explorer, `http://localhost:8000/mcp/schema` for MCP metadata.

## Test Your Agent

Use the interactive Agent CLI to test your agent:

```bash
# Start interactive chat
fastagentic agent chat --endpoint /triage

# Or send a single query
fastagentic agent query "Classify this ticket: User cannot login"
```

The Agent CLI provides:
- Streaming responses with real-time display
- Tool call visualization
- Conversation history
- Multiple output formats (markdown, plain, JSON)

See the [Agent CLI Guide](cli-agent.md) for full documentation.

## Validate Schema Parity

Contract tests ensure that OpenAPI and MCP descriptions stay synchronized:

```bash
fastagentic test contract
```

The command compares generated schemas, required scopes, and parameter definitions across both protocols.

## Next Steps

**Learn More:**
- [Why FastAgentic?](why-fastagentic.md) - Understand the deployment layer concept
- [Adapters Guide](adapters/index.md) - Deep dive into each adapter
- [Comparison](comparison.md) - Compare with alternatives

**Go Deeper:**
- [Architecture](architecture.md) - How FastAgentic works
- [Decorators](decorators.md) - API reference
- [Platform Services](platform-services.md) - Auth, policy, observability

**Deploy to Production:**
- [Operations Guide](operations/index.md) - Production deployment
- [Docker](operations/deployment/docker.md) - Container deployment
- [Kubernetes](operations/deployment/kubernetes.md) - K8s deployment

