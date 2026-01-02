---
title: Home
description: FastAgentic - The deployment layer for agentic applications
hide:
  - navigation
  - toc
---

# FastAgentic

<p class="hero-tagline" style="font-size: 1.25rem; color: var(--md-default-fg-color--light); text-align: center; margin-bottom: 2rem;">
<strong>Build agents with anything. Ship them with FastAgentic.</strong>
</p>

<p style="text-align: center; margin-bottom: 2rem;">
FastAgentic is the deployment layer for agentic applications. It transforms agents built with PydanticAI, LangChain, LangGraph, or CrewAI into production-ready services.
</p>

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get Started**

    ---

    Install FastAgentic and deploy your first agent in minutes

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-puzzle:{ .lg .middle } **Adapters**

    ---

    Connect your favorite agent framework - PydanticAI, LangGraph, CrewAI, LangChain

    [:octicons-arrow-right-24: Adapters](adapters/index.md)

-   :material-protocol:{ .lg .middle } **Protocols**

    ---

    Native support for MCP and A2A protocols

    [:octicons-arrow-right-24: Protocols](protocols/index.md)

-   :material-shield-check:{ .lg .middle } **Production Ready**

    ---

    Built-in reliability, observability, and security

    [:octicons-arrow-right-24: Operations](operations/index.md)

</div>

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Getting Started](getting-started.md) |
| Understand what FastAgentic does | [Why FastAgentic?](why-fastagentic.md) |
| Choose an adapter for my framework | [Choosing an Adapter](guides/choosing-an-adapter.md) |
| Test my agent interactively | [Agent CLI](cli-agent.md) |
| Deploy to production | [Operations Guide](operations/index.md) |

## Installation

=== "pip"

    ```bash
    pip install fastagentic
    ```

=== "uv"

    ```bash
    uv add fastagentic
    ```

=== "With adapters"

    ```bash
    pip install fastagentic[pydanticai]  # or langgraph, crewai, langchain
    ```

## Quick Example

```python
from fastagentic import App, tool, resource

app = App(name="my-agent")

@app.tool()
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

@app.resource("config://app")
def get_config() -> dict:
    """Get application configuration."""
    return {"version": "1.0", "env": "production"}
```

Run your agent:

```bash
fastagentic run
```

## Core Features

<div class="grid" markdown>

<div class="grid-item" markdown>

### :material-connection: Framework Adapters

Connect agents from any framework with zero code changes.

- [PydanticAI](adapters/pydanticai.md) - Type-safe agents
- [LangGraph](adapters/langgraph.md) - Stateful workflows
- [CrewAI](adapters/crewai.md) - Multi-agent teams
- [LangChain](adapters/langchain.md) - Chains & runnables

</div>

<div class="grid-item" markdown>

### :material-shield-lock: Governance & Policy

Enterprise-grade security and compliance.

- [RBAC & Scopes](policy.md) - Fine-grained access control
- [PII Detection](compliance.md) - Automatic data protection
- [Audit Logging](policy.md#audit-logging) - Complete audit trail

</div>

<div class="grid-item" markdown>

### :material-chart-line: Observability

Full visibility into your agent operations.

- [Metrics](operations/observability/metrics.md) - Prometheus export
- [Tracing](operations/observability/tracing.md) - OpenTelemetry support
- [Dashboard](dashboard.md) - Built-in monitoring

</div>

<div class="grid-item" markdown>

### :material-repeat: Reliability

Production-grade resilience patterns.

- [Retry & Circuit Breaker](reliability.md) - Automatic recovery
- [Checkpointing](checkpoint.md) - Durable state
- [Rate Limiting](reliability.md#rate-limiting) - Traffic control

</div>

</div>

## Integrations

FastAgentic integrates with leading AI infrastructure tools:

<div class="grid cards" markdown>

-   **Langfuse** - Observability & tracing
-   **Portkey** - AI Gateway & routing
-   **Lakera** - Security guardrails
-   **Mem0** - Intelligent memory
-   **Braintrust** - Evaluation & testing

</div>

[:octicons-arrow-right-24: View all integrations](integrations/index.md)

## Version History

| Version | Highlights |
|---------|------------|
| **v1.2** | Interactive Agent CLI for testing |
| **v1.1** | New adapters (Semantic Kernel, AutoGen, LlamaIndex, DSPy), template ecosystem |
| **v1.0** | Python SDK, PII detection, dashboard, production readiness |
| **v0.5** | Cluster orchestration, distributed checkpointing |

[:octicons-arrow-right-24: Full Roadmap](roadmap.md)
