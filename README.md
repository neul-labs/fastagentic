<p align="center">
  <img src="https://docs.neullabs.com/fastagentic/logo.svg" alt="FastAgentic" width="120" />
</p>

<h1 align="center">FastAgentic</h1>

<p align="center">
  <strong>Build agents with anything. Ship them with FastAgentic.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/fastagentic/"><img src="https://img.shields.io/pypi/v/fastagentic?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/fastagentic/"><img src="https://img.shields.io/pypi/pyversions/fastagentic" alt="Python"></a>
  <a href="https://github.com/neullabs/fastagentic/actions"><img src="https://img.shields.io/badge/tests-899%20passed-brightgreen" alt="Tests"></a>
  <a href="https://docs.neullabs.com/fastagentic"><img src="https://img.shields.io/badge/docs-neullabs.com-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <a href="https://docs.neullabs.com/fastagentic">Documentation</a> •
  <a href="https://docs.neullabs.com/fastagentic/quickstart">Quickstart</a> •
  <a href="https://docs.neullabs.com/fastagentic/examples">Examples</a>
</p>

---

## The Deployment Layer for AI Agents

Your agent works locally. Now make it **production-ready in minutes, not weeks**.

FastAgentic wraps any agent framework—**PydanticAI, LangGraph, CrewAI, LangChain**—and adds everything you need for production:

| You Get | Without Rewriting Your Agent |
|---------|------------------------------|
| **Checkpointing** | Resume 90-minute research agents after crashes |
| **Observability** | See every step, token, and decision |
| **Cost Control** | Token budgets, rate limits, circuit breakers |
| **Security** | OAuth, RBAC, PII detection out of the box |
| **Protocols** | MCP + A2A support for tool sharing and collaboration |

```
Your Agent (any framework)  →  FastAgentic  →  Production-Ready API
```

## Install

```bash
pip install fastagentic
```

## 30-Second Example

Wrap any existing agent with **zero code changes**:

```python
from your_agent import research_agent  # Any agent, any framework
from fastagentic import run_opaque

# One line. That's it.
result = await run_opaque(
    research_agent,
    query="AI trends 2025",
    run_id="research-001",  # Enables resume on crash
)
```

Re-run with the same `run_id`? **Instant cached result.** No re-execution.

## Full Application

```python
from fastagentic import App, tool, agent_endpoint
from fastagentic.adapters import LangGraphAdapter

app = App(title="My Agent API")

@tool
async def search(query: str) -> list[str]:
    """Search the knowledge base."""
    return await kb.search(query)

@agent_endpoint("/chat", runnable=LangGraphAdapter(my_graph))
async def chat(message: str) -> str:
    ...
```

```bash
fastagentic run              # HTTP API server
fastagentic mcp serve        # MCP stdio server
fastagentic agent chat       # Interactive testing
```

## Framework Adapters

| Framework | Adapter | Install |
|-----------|---------|---------|
| PydanticAI | `PydanticAIAdapter` | `pip install fastagentic[pydanticai]` |
| LangGraph | `LangGraphAdapter` | `pip install fastagentic[langgraph]` |
| CrewAI | `CrewAIAdapter` | `pip install fastagentic[crewai]` |
| LangChain | `LangChainAdapter` | `pip install fastagentic[langchain]` |

## Why FastAgentic?

**The problem:** You've built an agent. It works. But shipping it means adding:
- Crash recovery for long-running tasks
- Monitoring and cost tracking
- Authentication and authorization
- Rate limiting and retries
- API endpoints and protocols

**The solution:** FastAgentic handles all of this. You keep your agent code unchanged.

## Documentation

**[docs.neullabs.com/fastagentic](https://docs.neullabs.com/fastagentic)**

- [Getting Started](https://docs.neullabs.com/fastagentic/quickstart) — Up and running in 5 minutes
- [Adapters](https://docs.neullabs.com/fastagentic/adapters) — Connect any framework
- [Checkpointing](https://docs.neullabs.com/fastagentic/checkpoint) — Resume long-running agents
- [MCP Protocol](https://docs.neullabs.com/fastagentic/protocols/mcp) — Tool sharing standard
- [Deployment](https://docs.neullabs.com/fastagentic/deployment) — Docker, Kubernetes, cloud

## Contributing

```bash
git clone https://github.com/neullabs/fastagentic
cd fastagentic
uv sync --extra dev
uv run pytest tests/ -v
```

## License

MIT
