# Why FastAgentic?

## The Deployment Layer for Agentic Applications

FastAgentic solves a specific problem: **the gap between building an agent and running it in production**.

Agent frameworks like PydanticAI, LangChain, LangGraph, and CrewAI excel at orchestrating LLM interactions. But deploying those agents with proper governance—authentication, rate limiting, cost tracking, multi-protocol support—requires writing the same boilerplate for every project.

FastAgentic provides that deployment layer. One decorator gives you REST endpoints, MCP tools, SSE streaming, WebSocket support, OIDC auth, OpenTelemetry traces, and checkpoint durability. Your agent becomes a production service in minutes, not weeks.

## What FastAgentic Is (and Isn't)

| FastAgentic IS | FastAgentic IS NOT |
|----------------|---------------------|
| A deployment layer | An agent framework |
| Protocol multiplexer (REST + MCP + A2A + streaming) | Replacement for PydanticAI or LangChain |
| Production runtime with governance | Local development tool only |
| Framework-agnostic adapter host | Locked to one agent framework |
| The "FastAPI for agents" | Yet another LLM orchestration library |

## The Problem: Deployment Gap

Building an agent is straightforward:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4', system_prompt="You are a helpful assistant")
result = await agent.run("Hello!")
```

But deploying it for production use requires:

- **REST API**: Routes, request validation, error handling
- **MCP Protocol**: Tool registration, schema generation, transport handling
- **A2A Protocol**: Agent Cards, task delegation, agent discovery
- **Authentication**: OAuth2/OIDC integration, token validation, scope enforcement
- **Authorization**: Rate limits, quotas, role-based access
- **Streaming**: SSE, WebSocket, backpressure handling
- **Durability**: Checkpoints, resume capability, idempotency
- **Observability**: Traces, metrics, structured logging
- **Cost Control**: Token tracking, budget enforcement, alerts

Each of these is 100-500 lines of code. For every agent. Every project.

## The Solution: One Decorator

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic_ai import Agent

agent = Agent('openai:gpt-4', system_prompt="You are a helpful assistant")

app = App(
    title="My Assistant",
    version="1.0.0",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost:6379",
)

@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(agent),
    stream=True,
    durable=True,
    scopes=["chat:run"],
    mcp_tool="chat",
    a2a_skill="assistant-chat",
)
async def chat(message: str) -> str:
    pass
```

**What you get:**
- `POST /chat` - REST endpoint with OpenAPI docs
- `POST /chat/stream` - SSE streaming endpoint
- MCP tool registration at `/mcp/schema`
- A2A skill registration at `/.well-known/agent.json`
- OAuth2 authentication with scope enforcement
- Rate limiting and quota enforcement
- OpenTelemetry traces and metrics
- Redis checkpoints with resume capability
- Cost tracking per run, user, and tenant

## How FastAgentic Fits Your Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ PydanticAI  │  │  LangGraph  │  │   CrewAI    │     │
│  │   Agent     │  │    Graph    │  │    Crew     │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         ▼                ▼                ▼             │
│  ┌─────────────────────────────────────────────────┐   │
│  │              FastAgentic Adapters               │   │
│  │  PydanticAIAdapter | LangGraphAdapter | ...     │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │           FastAgentic Runtime Layer             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │   │
│  │  │  REST   │ │   MCP   │ │Streaming│ │ Auth  │ │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────┘ │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │   │
│  │  │ Policy  │ │Telemetry│ │Durability│ │ Cost │ │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────┘ │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Production Traffic   │
              │  REST | MCP | WebSocket│
              └───────────────────────┘
```

## Key Principles

### 1. Framework Agnostic
Use any agent framework. FastAgentic adapters wrap:
- PydanticAI agents
- LangGraph state graphs
- CrewAI crews
- LangChain runnables
- Your custom implementation

### 2. Protocol Unified
Define once, expose everywhere:
- REST endpoints with OpenAPI 3.1 schemas
- MCP tools with identical schemas (2025-11-25)
- A2A skills for agent-to-agent collaboration (v0.3)
- SSE, WebSocket, and MCP streaming
- Same auth and policy across all protocols

### 3. Production First
Every feature is designed for production:
- Durable checkpoints survive restarts
- Cost tracking prevents budget overruns
- Audit logs satisfy compliance requirements
- Observability built in, not bolted on

### 4. Developer Familiar
If you know FastAPI, you know FastAgentic:
- Decorator-based API
- Pydantic models for validation
- Dependency injection
- Async-first design

## Next Steps

- [Getting Started](getting-started.md) - Install and create your first project
- [Adapters Overview](adapters/index.md) - Choose the right adapter for your framework
- [Comparison](comparison.md) - Detailed feature comparison with alternatives
