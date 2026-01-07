# FastAgentic

> **Build agents with anything. Ship them with FastAgentic.**

[![Tests](https://img.shields.io/badge/tests-608%20passed-brightgreen)]()[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()[![License](https://img.shields.io/badge/license-MIT-green)]()

FastAgentic is the deployment layer for agentic applications. Deploy PydanticAI, LangGraph, CrewAI, or LangChain agents with REST, MCP, and streaming interfaces—plus auth, policy, and observability.

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   Your Agent Logic      │     │      FastAgentic        │     REST, MCP, SSE
│   ─────────────────     │ ──► │   (Deployment Layer)    │ ──► Auth, Policy, Telemetry
│   PydanticAI            │     │   One decorator =       │     Durability, Cost Control
│   LangGraph             │     │   Production service    │
│   CrewAI                │     │                         │
│   LangChain             │     │                         │
└─────────────────────────┘     └─────────────────────────┘
```

## Adapters

| Framework | Adapter |
|-----------|---------|
| PydanticAI | `PydanticAIAdapter` |
| LangGraph | `LangGraphAdapter` |
| CrewAI | `CrewAIAdapter` |
| LangChain | `LangChainAdapter` |

## Installation

```bash
uv add fastagentic

# With adapter support
uv add "fastagentic[pydanticai]" "fastagentic[langgraph]" "fastagentic[crewai]"

# With integrations
uv add "fastagentic[langfuse]" "fastagentic[portkey]"
```

## Quick Start

```python
from fastagentic import App, agent_endpoint, tool
from fastagentic.adapters.langgraph import LangGraphAdapter
from models import TicketIn, TicketOut, triage_graph

app = App(
    title="Support Triage",
    oidc_issuer="https://auth.mycompany.com",
)

@tool(name="summarize", description="Summarize ticket text")
async def summarize(text: str) -> str:
    ...

@agent_endpoint(
    path="/triage",
    runnable=LangGraphAdapter(triage_graph),
    input_model=TicketIn,
    output_model=TicketOut,
    stream=True,
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

Run:

```bash
fastagentic run          # HTTP server
fastagentic mcp serve    # MCP stdio server
```

## Integrations

```python
from fastagentic import App
from fastagentic.integrations import LangfuseIntegration, LakeraIntegration

app = App(
    integrations=[
        LangfuseIntegration(public_key="pk-...", secret_key="sk-..."),
        LakeraIntegration(api_key="lak-...", block_on_detect=True),
    ]
)
```

## Reliability

```python
from fastagentic import App, RetryPolicy, RateLimit

app = App(
    retry_policy=RetryPolicy(max_attempts=3, backoff="exponential"),
    rate_limit=RateLimit(rpm=60, tpm=100000),
)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `fastagentic run` | Start ASGI server |
| `fastagentic new` | Scaffold new application |
| `fastagentic agent chat` | Interactive agent testing |
| `fastagentic mcp serve` | Run as MCP stdio server |
| `fastagentic mcp schema` | Print MCP schema |

## Protocol Support

- **MCP** (2025-11-25) with Tasks, Extensions, OAuth
- **A2A** (v0.3) for agent-to-agent collaboration
- **OpenAPI 3.1** schemas from Pydantic models
- **OAuth2/OIDC** with scoped policies
- **Streaming**: SSE, WebSocket, MCP events

## Contributing

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/
```

## Learn More

- [Getting Started](docs/getting-started.md)
- [Adapters](docs/adapters/index.md)
- [MCP Protocol](docs/protocols/mcp.md)
- [A2A Protocol](docs/protocols/a2a.md)
- [Deployment](docs/operations/deployment/)
