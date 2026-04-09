# Integrations

FastAgentic integrates with best-of-breed tools for observability, guardrails, memory, evaluation, and more. Rather than building everything from scratch, we provide hooks and providers to connect your agents with specialized solutions.

## Philosophy

> FastAgentic owns the deployment layer. Specialized tools handle their domains better.

We build hooks. You choose the tools.

## Current Integrations

### Observability

Track LLM calls, token usage, costs, and trace agent execution.

| Integration | What It Does | Status |
|-------------|--------------|--------|
| **[Langfuse](langfuse.md)** | LLM tracing, prompt analytics, cost tracking | ✅ Stable |
| OTEL (built-in) | OpenTelemetry span export | ✅ Stable |

### Guardrails & Security

Protect against prompt injection, validate outputs, enforce content policies.

| Integration | What It Does | Status |
|-------------|--------------|--------|
| **[Lakera](lakera.md)** | Prompt injection detection, content moderation | ✅ Stable |

### Memory

Persistent user memory, session context, and conversation history.

| Integration | What It Does | Status |
|-------------|--------------|--------|
| **[Mem0](mem0.md)** | Persistent user memory across sessions | ✅ Stable |

### LLM Gateway

Rate limiting, fallbacks, caching, and multi-provider routing.

| Integration | What It Does | Status |
|-------------|--------------|--------|
| **[Portkey](portkey.md)** | Gateway with fallbacks, caching, load balancing | ✅ Stable |

### Evaluation

Score agent outputs, track experiments, measure quality.

| Integration | What It Does | Status |
|-------------|--------------|--------|
| **[Braintrust](braintrust.md)** | Experiment tracking, scoring, datasets | ✅ Stable |

---

## Future Integrations (Roadmap)

The following integrations are planned for future releases:

| Integration | Category | Description |
|-------------|----------|-------------|
| Logfire | Observability | PydanticAI native observability, structured logging |
| Datadog | Observability | APM integration, dashboards, alerting |
| Guardrails AI | Guardrails | Output validation with RAIL specs |
| NeMo Guardrails | Guardrails | Conversational guardrails, topic control |
| Zep | Memory | Session memory with auto-summarization |
| LiteLLM | Gateway | Multi-provider routing, unified API |
| LangSmith | Evaluation | Trace-based evaluation, feedback |
| HumanLayer | HITL | Multi-channel approval (Slack, Email) |
| PromptLayer | Prompt Mgmt | Versioning, A/B testing, analytics |
| Agenta | Prompt Mgmt | Prompt + eval workflow |

Contributions are welcome! See [Building Custom Integrations](#building-custom-integrations) to get started.

---

## Quick Start

### Install Integration

```bash
# Observability
pip install fastagentic[langfuse]
pip install fastagentic[logfire]

# Guardrails
pip install fastagentic[lakera]
pip install fastagentic[guardrails]

# Memory
pip install fastagentic[mem0]
pip install fastagentic[zep]

# All first-class integrations
pip install fastagentic[integrations]
```

### Configure Hooks

```python
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook
from fastagentic.integrations.lakera import LakeraHook
from fastagentic.integrations.mem0 import Mem0Provider

app = App(
    title="My Agent",
    version="1.0.0",

    # Global hooks (apply to all endpoints)
    hooks=[
        LangfuseHook(
            public_key="pk-...",
            secret_key="sk-...",
        ),
        LakeraHook(
            api_key="...",
            on_failure="warn",  # fail-open
        ),
    ],

    # Memory provider
    memory=Mem0Provider(api_key="..."),
)
```

### Per-Endpoint Hooks

```python
from fastagentic import agent_endpoint
from fastagentic.integrations.guardrails import GuardrailsAIHook
from fastagentic.integrations.braintrust import BraintrustHook

@agent_endpoint(
    path="/triage",
    runnable=...,
    post_hooks=[
        GuardrailsAIHook(rail="triage_output.rail"),
    ],
    eval_hooks=[
        BraintrustHook(project="support-triage"),
    ],
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

---

## Configuration Patterns

### Environment Variables

All integrations support environment variable configuration:

```bash
# Langfuse
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."

# Lakera
export LAKERA_API_KEY="..."

# Mem0
export MEM0_API_KEY="..."

# Portkey
export PORTKEY_API_KEY="..."
```

```python
from fastagentic.integrations.langfuse import LangfuseHook

# Automatically reads from environment
app = App(hooks=[LangfuseHook()])
```

### Configuration File

```yaml
# config/settings.yaml
integrations:
  langfuse:
    public_key: ${LANGFUSE_PUBLIC_KEY}
    secret_key: ${LANGFUSE_SECRET_KEY}
    host: https://cloud.langfuse.com  # or self-hosted

  lakera:
    api_key: ${LAKERA_API_KEY}
    on_failure: warn  # warn | reject

  mem0:
    api_key: ${MEM0_API_KEY}
```

```python
from fastagentic import App
from fastagentic.config import load_config

config = load_config("config/settings.yaml")
app = App.from_config(config)
```

---

## Hook Execution Model

### Blocking vs Non-Blocking

| Hook Type | Execution | Use Case |
|-----------|-----------|----------|
| `pre_hooks` | Blocking | Input validation, guardrails |
| `post_hooks` | Blocking | Output filtering, transformation |
| `eval_hooks` | Non-blocking | Async scoring, doesn't delay response |

### Error Handling

Configure fail-open or fail-closed per integration:

```python
# Fail-closed: Block if Lakera fails
LakeraHook(api_key="...", on_failure="reject")

# Fail-open: Log warning, continue if Lakera fails
LakeraHook(api_key="...", on_failure="warn")
```

---

## Building Custom Integrations

### Hook Interface

```python
from fastagentic.hooks import BaseHook, HookContext

class MyCustomHook(BaseHook):
    hooks = ["on_llm_start", "on_llm_end"]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = MyClient(api_key)

    async def on_llm_start(self, ctx: HookContext):
        # Pre-LLM logic
        pass

    async def on_llm_end(self, ctx: HookContext):
        # Post-LLM logic
        await self.client.track(
            model=ctx.model,
            tokens=ctx.usage.total_tokens,
        )
```

### Provider Interface

```python
from fastagentic.memory import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    async def get(self, user_id: str, key: str) -> Any:
        ...

    async def set(self, user_id: str, key: str, value: Any) -> None:
        ...

    async def search(self, user_id: str, query: str) -> list[dict]:
        ...
```

### Publishing

Community integrations can be published as separate packages:

```
fastagentic-myintegration/
├── fastagentic_myintegration/
│   ├── __init__.py
│   └── hook.py
├── pyproject.toml
└── README.md
```

Register in `pyproject.toml`:

```toml
[project.entry-points."fastagentic.integrations"]
myintegration = "fastagentic_myintegration:MyHook"
```

---

## Integration Matrix

| Integration | on_request | on_llm_* | on_tool_* | on_response | on_error | Memory |
|-------------|:----------:|:--------:|:---------:|:-----------:|:--------:|:------:|
| Langfuse | | ✓ | ✓ | | ✓ | |
| Lakera | | ✓ | | | | |
| Mem0 | | | | | | ✓ |
| Portkey | | ✓ | | | | |
| Braintrust | | | | ✓ | | |

---

## Next Steps

Choose an integration to get started:

- [Langfuse](langfuse.md) — Most popular for LLM observability
- [Lakera](lakera.md) — Essential for production security
- [Mem0](mem0.md) — Best for persistent user memory

Or read the [Hooks Architecture](../hooks.md) to understand how integrations work under the hood.
