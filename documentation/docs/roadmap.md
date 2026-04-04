# Roadmap

FastAgentic targets staged releases that progressively deliver the deployment layer for agentic applications. This roadmap outlines planned milestones across adapters, operations, governance, integrations, **and documentation**.

**Core Mission:** Build agents with anything. Ship them with FastAgentic.

**Philosophy:** FastAgentic owns the deployment layer. For specialized concerns (observability, guardrails, memory, evaluation), we provide fine-grained hooks to integrate best-of-breed solutions rather than reinventing wheels.

**Documentation Philosophy:** Every feature ships with decision-focused documentation that explains **why** you should use it and **when** it applies. Reference docs tell you how; guides tell you why.

---

## Build vs Integrate

FastAgentic focuses on deployment-layer concerns. Specialized tools handle their domains better.

| Category | FastAgentic Builds | Integrate Via Hooks |
|----------|-------------------|---------------------|
| **Protocol hosting** | REST + MCP + A2A | - |
| **Schema fusion** | Pydantic → OpenAPI/MCP/A2A | - |
| **Adapters** | PydanticAI, LangGraph, CrewAI, LangChain | - |
| **Durability** | Checkpoints, resume, replay | - |
| **Auth** | OIDC bridge to MCP + A2A | - |
| **Agent Registry** | Internal A2A discovery | - |
| **Observability** | OTEL spans (basic) | Langfuse, Logfire, Datadog |
| **Cost tracking** | Token counters (basic) | Helicone, Portkey |
| **Rate limiting** | Simple limiter | Portkey, Kong |
| **Guardrails** | Hook interface only | Lakera, Guardrails AI, NeMo |
| **Prompt injection** | Hook interface only | Lakera, Prompt Armor |
| **Evaluation** | Hook interface only | Braintrust, LangSmith, Maxim |
| **Memory** | Redis basic | Mem0, Zep |
| **Prompt versioning** | Hook interface only | PromptLayer, Latitude |

---

## First-Class Integrations

These integrations are officially documented, tested, and maintained:

| Category | Integrations | Status |
|----------|--------------|--------|
| **Observability** | Langfuse, Logfire (PydanticAI), Datadog | v0.3 |
| **Guardrails** | Lakera, Guardrails AI | v0.3 |
| **Memory** | Mem0, Zep, Redis | v0.3 |
| **LLM Gateway** | Portkey, LiteLLM | v0.3 |
| **Evaluation** | Braintrust, LangSmith | v0.4 |
| **HITL** | HumanLayer, Slack/Email webhooks | v0.4 |

Community integrations welcome via plugin system.

---

## Documentation Strategy

Documentation is a first-class deliverable in every release. Each feature ships with:

| Doc Type | Purpose | Example |
|----------|---------|---------|
| **Reference** | How to use it | API docs, configuration options |
| **Guide** | Why and when to use it | Decision trees, comparisons |
| **Tutorial** | Step-by-step learning | Getting started, templates |
| **Runbook** | Operations and troubleshooting | Production checklist, incident response |

### Documentation Index

| Category | Documents | Status |
|----------|-----------|--------|
| **Getting Started** | [Getting Started](getting-started.md), [Why FastAgentic](why-fastagentic.md), [Comparison](comparison.md) | v0.1 |
| **Architecture** | [Architecture](architecture.md), [Decorators](decorators.md), [Protocols](protocols/index.md) | v0.1 |
| **Decision Guides** | [Choosing an Adapter](guides/choosing-an-adapter.md), [Choosing Integrations](guides/choosing-integrations.md), [Production Checklist](guides/production-checklist.md) | v0.2 |
| **Adapters** | [PydanticAI](adapters/pydanticai.md), [LangGraph](adapters/langgraph.md), [CrewAI](adapters/crewai.md), [LangChain](adapters/langchain.md), [Custom](adapters/custom.md) | v0.2 |
| **Integrations** | [Langfuse](integrations/langfuse.md), [Lakera](integrations/lakera.md), [Mem0](integrations/mem0.md), [Portkey](integrations/portkey.md), [Braintrust](integrations/braintrust.md) | v0.3 |
| **Operations** | [Docker](operations/deployment/docker.md), [Kubernetes](operations/deployment/kubernetes.md), [Troubleshooting](operations/runbook/troubleshooting.md) | v0.2 |
| **Reference** | [CLI Reference](cli-reference.md), [Hooks](hooks.md), [Reliability](reliability.md), [Memory](memory.md) | v0.3 |

---

## v0.1 — Foundations

**Status: Shipped**

- `App` lifecycle with ASGI + MCP dual boot
- Decorators: `@tool`, `@resource`, `@prompt`, `@agent_endpoint`
- Schema fusion between OpenAPI 3.1 and MCP manifests
- LangChain adapter with streaming support
- Server-Sent Events for long-running operations
- Contract testing via `fastagentic test contract`

**Documentation:**
- Getting Started guide
- Why FastAgentic positioning
- Architecture overview
- Decorator reference
- Basic CLI reference

---

## v0.2 — Workflow Durability & Adapter Ecosystem

**Status: Shipped**

**Adapters:**
- PydanticAI adapter with full streaming support
- LangGraph adapter with node-level checkpointing
- CrewAI adapter with per-agent observability

**Protocols:**
- MCP 2025-11-25 specification compliance
- MCP Tasks for long-running operations
- A2A v0.3 protocol implementation
- Agent Registry for internal agent discovery
- Agent Card auto-generation from endpoints

**Durability:**
- Durable store backends (Redis and PostgreSQL)
- Resume, replay, and checkpoint inspection endpoints
- Idempotency keys and run deduplication
- MCP Task state mapping to durable runs

**Templates:**
- `fastagentic-templates` repository with index.json registry
- `fastagentic new --template pydanticai`
- `fastagentic new --template langgraph`
- `fastagentic new --template crewai`
- `fastagentic new --template langchain`
- `fastagentic templates list` command with remote discovery
- Community template contribution workflow

**DevOps:**
- Docker reference image with production defaults
- Helm chart beta with basic configuration
- Environment variable configuration reference

**CLI:**
- Enhanced scaffolding for agent workflows
- `fastagentic mcp validate` command
- `fastagentic a2a validate` command

**Documentation:**
- **Decision Guides:**
  - [Choosing an Adapter](guides/choosing-an-adapter.md) — Which framework adapter fits your use case
  - [Choosing Integrations](guides/choosing-integrations.md) — What tools to add for production
  - [Production Checklist](guides/production-checklist.md) — Are you ready to deploy?
- **Adapter Guides:**
  - Complete guides for PydanticAI, LangGraph, CrewAI, LangChain adapters
  - Adapter comparison with feature matrix
  - Migration guides between adapters
- **Protocol Documentation:**
  - MCP 2025-11-25 implementation guide
  - A2A v0.3 with Agent Registry
  - Protocol interoperability guide
- **Operations:**
  - Docker deployment guide
  - Kubernetes with Helm
  - Environment configuration reference

---

## v0.3 — Hooks, Integrations & Governance

**Status: Shipped**

**Hooks Architecture:**
- Fine-grained hook system with lifecycle points:
  - `on_request` / `on_response` — Request/response transformation
  - `on_tool_call` / `on_tool_result` — Per-tool interception
  - `on_node_enter` / `on_node_exit` — LangGraph node-level hooks
  - `on_llm_start` / `on_llm_end` — LLM call instrumentation
  - `on_checkpoint` / `on_resume` — Durability lifecycle
  - `on_error` / `on_retry` — Error handling hooks
- Hook registration via decorators and configuration
- Async hook execution (non-blocking where possible)
- Hook context with run metadata, user info, cost tracking

**Observability Integrations:**
- `LangfuseHook` — LLM tracing, prompt analytics
- `LogfireHook` — PydanticAI native observability
- `DatadogHook` — APM integration
- Built-in OTEL exporter improvements
- Per-agent cost attribution in traces

**Guardrail Integrations:**
- `LakeraHook` — Prompt injection detection, content moderation
- `GuardrailsAIHook` — Output validation with RAIL specs
- `NeMoGuardrailsHook` — Conversational guardrails
- Pre/post execution hook points
- Fail-open vs fail-closed configuration

**Memory Integrations:**
- `MemoryProvider` abstract interface
- `Mem0Provider` — Persistent user memory
- `ZepProvider` — Session memory with summaries
- `RedisProvider` — Simple built-in option
- Memory injection into agent context

**Rate Limiting & Cost Control:**
- Simple built-in rate limiter (RPM, TPM, by user/tenant)
- `PortkeyGateway` — LLM gateway with fallbacks, caching
- `LiteLLMGateway` — Multi-provider routing
- Cost guardrails with automatic model downgrade
- Budget alerts and hard limits

**Policy Engine:**
- Quotas and scope-role mapping
- Tenant isolation with separate stores
- Audit logging for run decisions and model usage

**Protocols:**
- MCP Extensions framework support
- MCP OAuth authorization (SEP-991, SEP-1046)
- A2A push notifications (webhooks)
- A2A signed Agent Cards
- External agent registration in registry

**Observability (Built-in):**
- Expanded metrics (token utilization, queue depth, hook latency)
- Prometheus-compatible telemetry endpoints
- Grafana dashboard templates

**DevOps:**
- Kubernetes Operator for CRD-based agent management
- Prometheus ServiceMonitor integration
- Alert rules library

**Documentation:**
- **Integration Guides (Why & When):**
  - [Langfuse](integrations/langfuse.md) — When to use, configuration, best practices
  - [Lakera](integrations/lakera.md) — Prompt injection defense strategies
  - [Mem0](integrations/mem0.md) — Long-term memory patterns
  - [Portkey](integrations/portkey.md) — When to use gateway vs built-in reliability
  - [Guardrails AI](integrations/guardrails-ai.md) — Output validation patterns
- **Reference Documentation:**
  - [Hooks Architecture](hooks.md) — Complete hook lifecycle reference
  - [Reliability Patterns](reliability.md) — When to use built-in vs external tools
  - [Memory Providers](memory.md) — Provider comparison and selection guide
  - [CLI Reference](cli-reference.md) — Complete command reference
- **Decision Guides:**
  - "When to use Portkey vs built-in reliability"
  - "Mem0 vs Zep vs Redis: Memory provider selection"
  - "Observability tool comparison"
- **Operations:**
  - Grafana dashboard setup
  - Alert rules documentation
  - Troubleshooting runbook

---

## v0.4 — Human-in-the-Loop & Evaluation

**Status: Shipped**

**HITL Workflows:**
- `interrupt()` checkpoint for human approval
- Async approval via webhooks (Slack, Email, custom)
- `HumanLayerHook` — Multi-channel approval routing
- Three-way decision model (approve / edit / reject)
- Approval timeout and escalation policies
- WebSocket-based interactive sessions

**Evaluation Integrations:**
- `EvalHook` abstract interface
- `BraintrustHook` — Experiment tracking, scoring
- `LangSmithHook` — Trace-based evaluation
- `MaximHook` — Production eval pipelines
- Custom `LLMJudge` for inline evaluation
- Soft failure thresholds (don't hard-fail on borderline)
- Eval results in traces and dashboards

**Prompt Management Integrations:**
- `PromptProvider` abstract interface
- `PromptLayerProvider` — Versioning, A/B testing
- `LatitudeProvider` — Prompt CMS
- `AgentaProvider` — Prompt + eval workflow
- Prompt injection into agent context
- Version pinning in configuration

**Reliability Patterns:**
- `RetryPolicy` configuration (max attempts, backoff)
- `CircuitBreaker` for failing dependencies
- Timeout enforcement per endpoint
- Fallback chains for graceful degradation

**Protocols:**
- A2A gRPC transport support
- A2A task streaming (SSE + gRPC)
- MCP Sampling with Tools (SEP-1577)
- Cross-agent task delegation via A2A

**Artifacts:**
- Artifact storage for generated assets
- S3/GCS/Azure Blob providers
- Artifact references in run metadata

**DevOps:**
- Blue/green deployment support with traffic shifting
- Canary analysis integration (Flagger compatibility)
- Production runbook documentation

**Documentation:**
- **HITL Guides:**
  - Human-in-the-loop patterns guide
  - Approval workflow design patterns
  - Integration with existing approval systems
- **Evaluation Guides:**
  - [Braintrust](integrations/braintrust.md) — Experiment tracking setup
  - LangSmith integration for LangChain users
  - Custom evaluation strategies
  - "When to evaluate: Development vs Production"
- **Prompt Management:**
  - Prompt versioning best practices
  - A/B testing prompts guide
  - Prompt provider comparison
- **Advanced Operations:**
  - Blue/green deployment guide
  - Canary release patterns
  - Production runbook template

---

## v0.5 — Scale and Orchestration

**Status: Shipped**

**Scale:**
- Distributed checkpointing across clusters
- Job scheduler integration for batch workflows
- Auto-scaling based on queue depth metrics
- Multi-region deployment patterns

**Enterprise:**
- Multi-tenant isolation with per-tenant durable stores
- Fleet-aware cost and policy management
- SSO/SAML enterprise authentication

**Advanced Hooks:**
- Hook composition and chaining
- Conditional hook execution
- Hook performance monitoring
- Hook marketplace (community plugins)

**Documentation:**
- **Scale Guides:**
  - Multi-region deployment patterns
  - Distributed checkpointing guide
  - Auto-scaling configuration
- **Enterprise:**
  - Multi-tenant architecture guide
  - SSO/SAML integration
  - Fleet management patterns
- **Advanced Hooks:**
  - Hook composition patterns
  - Building custom hooks guide
  - Hook marketplace submission guide

---

## v1.0 — Production Suite

**Status: Shipped**

**Python SDK:**
- `FastAgenticClient` and `AsyncFastAgenticClient` for API interaction
- Streaming support with `StreamEvent` and `StreamEventType`
- Automatic retry with exponential backoff
- Comprehensive error handling (`AuthenticationError`, `RateLimitError`, etc.)
- `RunRequest`, `RunResponse`, `ToolCall`, `ToolResult` models

**Compliance & PII:**
- `PIIDetector` with built-in patterns for email, phone, SSN, credit cards, IP addresses
- `PIIMasker` for masking and redaction
- `PIIConfig` with allowlists, blocklists, and custom patterns
- `PIIDetectionHook` and `PIIMaskingHook` for request/response filtering
- Configurable confidence thresholds and type filtering

**Dashboard & Metrics:**
- `StatsCollector` for run tracking and statistics
- `RunStats`, `EndpointStats`, `SystemStats` data models
- `MetricsRegistry` with `Counter`, `Gauge`, `Histogram` metrics
- `PrometheusExporter` for Prometheus-compatible output
- `DashboardAPI` with health, metrics, and stats endpoints
- Configurable dashboard with authentication support

**Production Readiness:**
- `ReadinessChecker` with comprehensive production checks
- Security checks: auth, HTTPS, secrets management
- Reliability checks: timeouts, retries, rate limiting
- Observability checks: logging, metrics, health endpoints
- Compliance checks: PII detection, audit logging
- Custom check support with `ReadinessCheck`
- `ReadinessReport` with scoring and recommendations

**Documentation:**
- [SDK Guide](sdk.md) — Python client usage
- [Compliance Guide](compliance.md) — PII detection and masking
- [Dashboard Guide](dashboard.md) — Metrics and monitoring
- [Ops Guide](ops.md) — Production readiness checks

---

## v1.1 — Adapter & Template Ecosystem

**Status: Shipped**

**New Adapters:**
- `SemanticKernelAdapter` - Microsoft Semantic Kernel functions and agents
- `AutoGenAdapter` - Microsoft AutoGen multi-agent conversations
- `LlamaIndexAdapter` - LlamaIndex query engines, agents, and chat engines
- `DSPyAdapter` and `DSPyProgramAdapter` - DSPy modules and compiled programs

**Community Adapter SDK:**
- `CommunityAdapter` base class for custom adapters
- `SimpleAdapter` for wrapping simple functions
- `AdapterMetadata` for adapter registration
- `AdapterRegistry` for discovery and management
- `@register_adapter` decorator

**Template Ecosystem:**
- `Template`, `TemplateMetadata`, `TemplateVariable`, `TemplateFile`
- `TemplateVersion` with compatibility checking
- `LocalRegistry` for file-based templates
- `RemoteRegistry` for remote template sources
- `EnterpriseRegistry` with access control and auditing

**Template Marketplace:**
- `Marketplace` for community template discovery
- `TemplateRating` and `TemplateReview` for feedback
- Browse by category, framework, tags
- Search, ratings, and reviews

**Template Composition:**
- `TemplateComposer` for combining templates
- `CompositionConfig` for merge strategies
- File conflict detection and resolution
- Python, JSON, YAML merge support

**Documentation:**
- [New Adapters Guide](adapters-new.md) — SK, AutoGen, LlamaIndex, DSPy
- [Template Ecosystem Guide](templates-ecosystem.md) — Templates, marketplace, composition

---

## v1.2 — Interactive Agent CLI

**Status: Shipped**

**Agent CLI:**
- Interactive REPL for agent testing (Claude Code / Gemini CLI-like experience)
- `fastagentic agent chat` - Full interactive chat session
- `fastagentic agent query` - Single queries for scripting and piping
- `fastagentic agent config` - Configuration management
- `fastagentic agent history` - Conversation history management

**REPL Features:**
- Streaming responses with real-time updates
- Tool call visualization with formatted output
- Conversation history with save/load to JSON
- Configurable output formats (markdown, plain, json)
- Slash commands for configuration and navigation
- Server health checking
- File input support

**Scripting Support:**
- Stdin/stdout piping for automation
- Output file support
- Environment variable configuration
- Config file at `~/.fastagentic/config.json`

**Documentation:**
- [Agent CLI Guide](cli-agent.md) — Commands, configuration, examples

---

## Beyond v1.2

**Integration Ecosystem:**
- Community hook marketplace
- Certified integration program
- Integration testing framework
- Partner integration guides

**Advanced Features:**
- Multi-modal streaming (audio, image artifacts)
- Marketplace for shared prompts, tools, and adapters
- Compliance packs for industry-specific regulations (HIPAA, PCI-DSS)
- Agent simulation and load testing

---

## Hook Lifecycle Reference

```
Request Flow:
─────────────────────────────────────────────────────────────────
│ on_request │ → │ on_llm_start │ → │ on_tool_call │
─────────────────────────────────────────────────────────────────
                         │                    │
                         ▼                    ▼
              ┌─────────────────┐   ┌─────────────────┐
              │ Guardrail Hooks │   │   Eval Hooks    │
              │ (Lakera, etc.)  │   │ (Braintrust)    │
              └─────────────────┘   └─────────────────┘
                         │                    │
                         ▼                    ▼
─────────────────────────────────────────────────────────────────
│ on_tool_result │ → │ on_llm_end │ → │ on_response │
─────────────────────────────────────────────────────────────────
                              │
                              ▼
                    ┌─────────────────┐
                    │ on_checkpoint   │ (if durable)
                    └─────────────────┘

Error Flow:
─────────────────────────────────────────────────────────────────
│ on_error │ → │ on_retry │ (if retry policy) → │ resume flow │
─────────────────────────────────────────────────────────────────

LangGraph-Specific:
─────────────────────────────────────────────────────────────────
│ on_node_enter │ → │ node execution │ → │ on_node_exit │
─────────────────────────────────────────────────────────────────
```

---

## Contributing

Timelines are subject to change as community feedback and adoption inform priorities. Contributions, RFCs, and plugin proposals are welcome to accelerate the roadmap.

- [GitHub Discussions](https://github.com/fastagentic/fastagentic/discussions) - Feature requests
- [GitHub Issues](https://github.com/fastagentic/fastagentic/issues) - Bug reports
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Integration Guide](integrations/index.md) - Build a hook or provider
