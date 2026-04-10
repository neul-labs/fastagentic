# FastAgentic

> **Build agents with anything. Ship them with FastAgentic.**

[![Tests](https://img.shields.io/badge/tests-899%20passed-brightgreen)]()[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()[![License](https://img.shields.io/badge/license-MIT-green)]()

FastAgentic makes agents **resumable, observable, and production-safe**—without rewriting them.

```
Your Agent Code
    ↓
FastAgentic Runtime
    ├── Checkpointing     → Resume after crashes
    ├── Step Tracking     → See exactly what's happening
    ├── Cost Control      → Token budgets per phase
    └── Reliability       → Retry, circuit breaker, rate limits
    ↓
Production-Safe Execution
```

## The Problem

Agent builders already know this pain:
- Research agents run **30–90 minutes**
- They crash halfway and lose intermediate state
- Can't be resumed, observed, or reasoned about
- DevOps can't monitor progress or control costs

## The Solution

FastAgentic wraps execution, not reasoning. **Zero code changes** to your agent:

```python
from local_deep_research import quick_summary  # Any existing agent
from fastagentic import run_opaque

# One line wraps ANY agent
result = await run_opaque(
    quick_summary,           # Unchanged agent
    query="quantum computing",
    run_id="research-001",   # For resume capability
)

# Resume returns cached result instantly
result = await run_opaque(
    quick_summary,
    query="quantum computing",
    run_id="research-001",   # Same run_id = cached result
)
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

## Demo: Wrap Any Agent

See FastAgentic wrap an existing agent with **zero code changes**:

```bash
python examples/deep-research/demo.py "quantum computing"
```

```
┌────────────────────── FastAgentic Demo: Wrap Any Agent ──────────────────────┐
│ Agent: local_deep_research.quick_summary                                     │
│ Query: quantum computing                                                     │
│ Run ID: research-7f3a2b                                                      │
│                                                                              │
│ This is an UNMODIFIED existing agent.                                        │
│ FastAgentic wraps it with zero code changes.                                 │
└──────────────────────────────────────────────────────────────────────────────┘

● Running quick_summary...
✓ Complete! Cached for resume. Duration: 45.2s
```

Run again with the same run ID:

```bash
python examples/deep-research/demo.py "quantum computing" --run-id research-7f3a2b
```

```
✓ Found cached result for research-7f3a2b - skipping execution
```

**Instant.** No re-execution. The result was cached.

## Two Patterns

| Pattern | Use Case | Resume Behavior |
|---------|----------|-----------------|
| **Opaque** (`run_opaque`) | Wrap existing agents, zero changes | Skip if complete |
| **Step-aware** (`run` + `StepTracker`) | New multi-step workflows | Resume from exact step |

```python
# Pattern 1: Opaque - zero code changes
from fastagentic import run_opaque
result = await run_opaque(existing_agent, query="...")

# Pattern 2: Step-aware - fine-grained checkpointing
from fastagentic import run, StepTracker

async def my_agent(input: str, tracker: StepTracker):
    async with tracker.step("search"):
        results = await search(input)
    async with tracker.step("analyze"):
        return await analyze(results)

result = await run(my_agent, "quantum computing")
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
| `fastagentic runs list` | List recent runs with status |
| `fastagentic runs show <id>` | Show execution graph for a run |
| `fastagentic runs delete <id>` | Delete checkpoints for a run |

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
- [Deep Research Demo](examples/deep-research/) - Wrap any agent with zero code changes
- [Adapters](docs/adapters/index.md)
- [Checkpointing](docs/checkpoint.md)
- [MCP Protocol](docs/protocols/mcp.md)
- [A2A Protocol](docs/protocols/a2a.md)
- [Deployment](docs/operations/deployment/)
