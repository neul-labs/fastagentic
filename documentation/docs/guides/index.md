# Guides

Decision-focused guides to help you choose the right tools and patterns for your use case. These guides explain **why** you should use specific features and **when** they apply.

---

## Quick Decision Trees

### What adapter should I use?

```
Start Here
    │
    ├─ "I use PydanticAI"
    │       → PydanticAIAdapter (native streaming, Logfire support)
    │
    ├─ "I have complex state machines"
    │       → LangGraphAdapter (node checkpoints, conditional routing)
    │
    ├─ "I need multiple specialized agents"
    │       → CrewAIAdapter (role-based, task delegation)
    │
    ├─ "I use LangChain chains/agents"
    │       → LangChainAdapter (LCEL support, tool binding)
    │
    └─ "I have custom agent logic"
            → BaseAdapter (full control, minimal overhead)
```

**[Full Adapter Guide →](choosing-an-adapter.md)**

---

### What integrations do I need?

```
Production Deployment
    │
    ├─ "I need to see what my agents are doing"
    │       → Observability: Langfuse, Logfire, or Datadog
    │
    ├─ "I'm worried about prompt injection"
    │       → Guardrails: Lakera (recommended) or Guardrails AI
    │
    ├─ "My agents need to remember users"
    │       → Memory: Mem0 (long-term) or Zep (session)
    │
    ├─ "I need fallbacks when OpenAI is down"
    │       → LLM Gateway: Portkey or LiteLLM
    │
    └─ "I need to measure agent quality"
            → Evaluation: Braintrust or LangSmith
```

**[Full Integration Guide →](choosing-integrations.md)**

---

### Am I ready for production?

```
Production Readiness
    │
    ├─ [ ] Can I see what's happening? (Observability)
    ├─ [ ] Is my agent secure? (Guardrails)
    ├─ [ ] What happens when the LLM fails? (Reliability)
    ├─ [ ] How do I control costs? (Rate limiting, budgets)
    ├─ [ ] Can I recover from crashes? (Durability)
    └─ [ ] How do I deploy updates safely? (DevOps)
```

**[Full Production Checklist →](production-checklist.md)**

---

## Guide Index

### Getting Started

| Guide | Description | When to Read |
|-------|-------------|--------------|
| [Getting Started](../getting-started.md) | First steps with FastAgentic | New to FastAgentic |
| [Why FastAgentic](../why-fastagentic.md) | Problems we solve | Evaluating tools |
| [Comparison](../comparison.md) | How we compare to alternatives | Making decisions |

### Architecture & Concepts

| Guide | Description | When to Read |
|-------|-------------|--------------|
| [Architecture](../architecture.md) | System design and layers | Understanding the framework |
| [Decorators](../decorators.md) | @tool, @resource, @prompt, @agent_endpoint | Writing agent code |
| [Hooks](../hooks.md) | Lifecycle hooks for customization | Adding integrations |

### Choosing the Right Tools

| Guide | Description | When to Read |
|-------|-------------|--------------|
| [Choosing an Adapter](choosing-an-adapter.md) | Which agent framework adapter | Starting a new project |
| [Choosing Integrations](choosing-integrations.md) | Which tools to integrate | Planning production |
| [Production Checklist](production-checklist.md) | What you need before go-live | Before deployment |

### Deep Dives

| Guide | Description | When to Read |
|-------|-------------|--------------|
| [Reliability](../reliability.md) | Retries, circuit breakers, timeouts | Building resilient agents |
| [Memory](../memory.md) | Session and long-term memory | Adding personalization |
| [Protocols](../protocols/index.md) | MCP and A2A protocols | Agent interoperability |

---

## The FastAgentic Philosophy

### Build vs Integrate

FastAgentic follows a clear philosophy: **we own the deployment layer, not the entire stack**.

| We Build | We Integrate |
|----------|--------------|
| Protocol hosting (REST + MCP + A2A) | Observability (Langfuse, Datadog) |
| Schema fusion (Pydantic → OpenAPI/MCP/A2A) | Guardrails (Lakera, Guardrails AI) |
| Framework adapters | Memory (Mem0, Zep) |
| Durability (checkpoints, resume) | Evaluation (Braintrust, LangSmith) |
| Auth (OIDC bridge) | LLM Gateway (Portkey, LiteLLM) |

**Why this matters:**

1. **Best-of-breed tools**: Langfuse does observability better than we could. Lakera does prompt injection detection better. We integrate with them.

2. **Your choice**: Don't like Langfuse? Use Datadog. Don't need memory? Skip it. You're not locked in.

3. **Focused excellence**: We're experts at deployment, not at building 50 different features poorly.

---

## Common Questions

### "Do I need all the integrations?"

No. Start minimal:

```python
# Minimum viable production
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook

app = App(
    title="My Agent",
    hooks=[LangfuseHook()],  # Just observability
)
```

Add integrations as you need them. Most teams start with:
1. **Observability** (always — you need to see what's happening)
2. **Guardrails** (if user-facing)
3. **Reliability** (if production traffic)

### "Which adapter is best?"

There's no "best" — each fits different use cases:

- **PydanticAI**: Best for Pydantic-native apps, type safety
- **LangGraph**: Best for complex workflows, state machines
- **CrewAI**: Best for multi-agent collaboration
- **LangChain**: Best if you already use LangChain

See [Choosing an Adapter](choosing-an-adapter.md) for details.

### "Should I use built-in reliability or Portkey?"

| Use Case | Recommendation |
|----------|----------------|
| Simple retry/timeout | Built-in `RetryPolicy`, `Timeout` |
| Multi-provider fallback | Portkey or LiteLLM |
| Semantic caching | Portkey |
| Cross-provider load balancing | Portkey |

See [Reliability](../reliability.md) for the full breakdown.

### "Mem0 vs Zep vs Redis?"

| Use Case | Recommendation |
|----------|----------------|
| Long-term user personalization | Mem0 |
| Session memory with auto-summarization | Zep |
| Simple key-value, no semantic search | Redis |

See [Memory](../memory.md) for details.

---

## Learning Paths

### Path 1: New to AI Agents

1. [Getting Started](../getting-started.md)
2. [Decorators](../decorators.md)
3. [Choosing an Adapter](choosing-an-adapter.md)
4. [Your first template](../templates/index.md)

### Path 2: Going to Production

1. [Production Checklist](production-checklist.md)
2. [Choosing Integrations](choosing-integrations.md)
3. [Reliability](../reliability.md)
4. [Deployment](../operations/deployment/docker.md)

### Path 3: Advanced Patterns

1. [Architecture](../architecture.md)
2. [Hooks](../hooks.md)
3. [Protocols](../protocols/index.md)
4. [Custom Adapters](../adapters/custom.md)

---

## Next Steps

- [Choosing an Adapter](choosing-an-adapter.md) — Which framework adapter fits your use case
- [Choosing Integrations](choosing-integrations.md) — What tools to add for production
- [Production Checklist](production-checklist.md) — Are you ready to deploy?
