# FastAgentic Documentation

> **Build agents with anything. Ship them with FastAgentic.**

FastAgentic is the deployment layer for agentic applications. It transforms agents built with PydanticAI, LangChain, LangGraph, or CrewAI into production-ready services.

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Getting Started](getting-started.md) |
| Understand what FastAgentic does | [Why FastAgentic?](why-fastagentic.md) |
| Choose an adapter for my framework | [Choosing an Adapter](guides/choosing-an-adapter.md) |
| Test my agent interactively | [Agent CLI](cli-agent.md) |
| Deploy to production | [Operations Guide](operations/index.md) |

---

## Getting Started

- [Getting Started Guide](getting-started.md) - Install, scaffold, and run your first agent
- [Why FastAgentic?](why-fastagentic.md) - Understand the deployment layer concept
- [Comparison with Alternatives](comparison.md) - How FastAgentic compares to other tools

---

## Core Concepts

### Architecture
- [Architecture Overview](architecture.md) - How FastAgentic works
- [Decorators Reference](decorators.md) - `@tool`, `@resource`, `@prompt`, `@agent_endpoint`
- [Platform Services](platform-services.md) - Auth, policy, observability

### Adapters
- [Adapters Overview](adapters/index.md) - Framework adapter system
- [PydanticAI Adapter](adapters/pydanticai.md) - Type-safe agents
- [LangGraph Adapter](adapters/langgraph.md) - Stateful graph workflows
- [CrewAI Adapter](adapters/crewai.md) - Multi-agent collaboration
- [LangChain Adapter](adapters/langchain.md) - Chains and runnables
- [Custom Adapters](adapters/custom.md) - Build your own
- [New Adapters (v1.1)](adapters-new.md) - Semantic Kernel, AutoGen, LlamaIndex, DSPy
- [Adapter Comparison](adapters/comparison.md) - Feature matrix

### Protocols
- [Protocols Overview](protocols/index.md) - MCP, A2A, REST
- [MCP Implementation](protocols/mcp.md) - Model Context Protocol
- [A2A Integration](protocols/a2a.md) - Agent-to-Agent protocol

---

## Features

### Reliability & Resilience
- [Reliability Patterns](reliability.md) - Retry, circuit breaker, rate limiting
- [Checkpointing](checkpoint.md) - Durable state and resume

### Governance & Policy
- [Policy Engine](policy.md) - RBAC, scopes, budgets
- [Compliance & PII](compliance.md) - PII detection and masking

### Observability
- [Dashboard & Metrics](dashboard.md) - Stats, metrics, Prometheus
- [Production Readiness](ops.md) - Readiness checks

### Advanced Features
- [Hooks System](hooks.md) - Lifecycle hooks for integrations
- [Memory Providers](memory.md) - Redis, Mem0, Zep
- [Prompt Management](prompts-management.md) - Templates, versioning, A/B testing
- [Human-in-the-Loop](hitl.md) - Approval workflows, escalation
- [Cluster Orchestration](cluster.md) - Workers, task queues, coordination

---

## Developer Tools

### CLI Reference
- [CLI Reference](cli-reference.md) - All commands
- [Agent CLI](cli-agent.md) - Interactive testing and development

### SDK
- [Python SDK](sdk.md) - `FastAgenticClient`, `AsyncFastAgenticClient`

### Templates
- [Templates Overview](templates/index.md) - Project scaffolding
- [Template Ecosystem](templates-ecosystem.md) - Registry, marketplace, composition
- [PydanticAI Template](templates/pydanticai.md)
- [LangGraph Template](templates/langgraph.md)
- [CrewAI Template](templates/crewai.md)
- [LangChain Template](templates/langchain.md)
- [Contributing Templates](templates/contributing.md)

---

## Integrations

- [Integrations Overview](integrations/index.md) - First-class integrations
- [Langfuse](integrations/langfuse.md) - Observability and tracing
- [Portkey](integrations/portkey.md) - AI Gateway
- [Lakera](integrations/lakera.md) - Security guardrails
- [Mem0](integrations/mem0.md) - Intelligent memory
- [Braintrust](integrations/braintrust.md) - Evaluation

---

## Operations

- [Operations Overview](operations/index.md) - Production deployment guide

### Deployment
- [Docker](operations/deployment/docker.md) - Container deployment
- [Kubernetes](operations/deployment/kubernetes.md) - K8s with Helm
- [Serverless](operations/deployment/serverless.md) - AWS Lambda, Cloud Functions

### Configuration
- [Environment Variables](operations/configuration/environment-vars.md)
- [Secrets Management](operations/configuration/secrets-management.md)

### Observability
- [Metrics](operations/observability/metrics.md) - Prometheus metrics
- [Tracing](operations/observability/tracing.md) - OpenTelemetry traces
- [Alerting](operations/observability/alerting.md) - Alert rules

### Security
- [Hardening](operations/security/hardening.md) - Security best practices
- [Compliance](operations/security/compliance.md) - Regulatory compliance

### Runbook
- [Troubleshooting](operations/runbook/troubleshooting.md) - Common issues

---

## Guides

- [Guides Overview](guides/index.md) - Decision guides
- [Choosing an Adapter](guides/choosing-an-adapter.md) - Framework selection
- [Choosing Integrations](guides/choosing-integrations.md) - What tools to add
- [Production Checklist](guides/production-checklist.md) - Deployment readiness

---

## Reference

- [MCP Authorization](reference/mcp-authorization.md) - OAuth for MCP
- [A2A Specification](reference/a2a-specification.md) - Protocol spec
- [Roadmap](roadmap.md) - Feature roadmap and releases

---

## Version History

| Version | Highlights |
|---------|------------|
| **v1.2** | Interactive Agent CLI for testing |
| **v1.1** | New adapters (SK, AutoGen, LlamaIndex, DSPy), template ecosystem |
| **v1.0** | Python SDK, PII detection, dashboard, production readiness |
| **v0.5** | Cluster orchestration, distributed checkpointing |
| **v0.4** | Prompt management, human-in-the-loop |
| **v0.3** | Policy engine, cost tracking, audit logging |
| **v0.2** | Reliability patterns, MCP stdio, integrations |
| **v0.1** | Core decorators, MCP schema fusion, adapters |

See the full [Roadmap](roadmap.md) for details.
