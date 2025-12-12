# Choosing Integrations

This guide helps you decide which integrations to add for your FastAgentic deployment. We'll explain **why** you need each category and **which tool** to choose.

---

## The Big Picture

FastAgentic provides hooks to integrate specialized tools. Here's what each category solves:

| Category | Problem It Solves | When You Need It |
|----------|-------------------|------------------|
| **Observability** | "I can't see what my agent is doing" | Always (production) |
| **Guardrails** | "Users can trick my agent" | User-facing agents |
| **Memory** | "My agent forgets users" | Personalization |
| **LLM Gateway** | "OpenAI went down and broke everything" | High availability |
| **Evaluation** | "I don't know if my agent is good" | Quality assurance |

---

## Start Simple

You don't need everything on day one. Here's a staged approach:

### Stage 1: MVP (Minimum Viable Production)

```python
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook

app = App(
    title="My Agent",
    hooks=[LangfuseHook()],  # Just observability
)
```

**Why Langfuse first?** You need to see what's happening before you can fix problems. Start here.

### Stage 2: Production

```python
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook
from fastagentic.integrations.lakera import LakeraHook

app = App(
    hooks=[
        LangfuseHook(),       # See what's happening
        LakeraHook(),         # Block attacks
    ],
    retry=RetryPolicy(...),   # Handle failures
)
```

### Stage 3: Scale

```python
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook
from fastagentic.integrations.lakera import LakeraHook
from fastagentic.integrations.portkey import PortkeyGateway
from fastagentic.integrations.mem0 import Mem0Provider

app = App(
    hooks=[
        LangfuseHook(),
        LakeraHook(),
    ],
    llm_gateway=PortkeyGateway(...),  # Fallbacks, caching
    memory=Mem0Provider(...),          # User memory
)
```

---

## Category 1: Observability

### Why You Need It

Without observability, your agent is a black box:
- You can't debug issues
- You don't know what's slow
- You can't track costs
- You can't improve quality

### Decision Tree

```
What's your stack?
│
├─► "We use PydanticAI"
│   └─► Logfire — Native integration, built by Pydantic team
│
├─► "We have an existing APM"
│   │
│   ├─► Datadog? → DatadogHook
│   └─► Other? → Built-in OTEL export
│
└─► "We need LLM-specific analytics"
    └─► Langfuse — Purpose-built for LLM observability
```

### Comparison

| Tool | Best For | Key Features | Self-Hosted |
|------|----------|--------------|-------------|
| **Langfuse** | LLM-specific analytics | Prompt analytics, cost tracking, evaluations | Yes |
| **Logfire** | PydanticAI native | Automatic tracing, Pydantic integration | No |
| **Datadog** | Existing Datadog users | APM integration, dashboards, alerting | No |
| **OTEL** | Custom/existing systems | Standard protocol, flexible | N/A |

### Recommendation

| Situation | Use |
|-----------|-----|
| Starting fresh | **Langfuse** — Best LLM-specific features |
| PydanticAI + want native | **Logfire** — Zero config with PydanticAI |
| Enterprise with Datadog | **DatadogHook** — Unified observability |
| Custom requirements | **Built-in OTEL** — Export to anything |

```python
# Langfuse (recommended for most)
from fastagentic.integrations.langfuse import LangfuseHook

LangfuseHook(
    public_key="...",
    secret_key="...",
    # Track costs, see prompts, debug issues
)

# Logfire (for PydanticAI)
from fastagentic.integrations.logfire import LogfireHook

LogfireHook()  # Auto-configures with PydanticAI
```

---

## Category 2: Guardrails

### Why You Need It

Users will try to:
- Inject prompts ("Ignore your instructions and...")
- Extract system prompts
- Make your agent do harmful things
- Bypass safety measures

**If your agent is user-facing, you need guardrails.**

### Decision Tree

```
What do you need to protect against?
│
├─► "Prompt injection attacks"
│   └─► Lakera — Best-in-class prompt injection detection
│
├─► "Output format validation"
│   └─► Guardrails AI — RAIL specs, structured validation
│
├─► "Conversational safety"
│   └─► NeMo Guardrails — Dialog-level policies
│
└─► "All of the above"
    └─► Lakera + Guardrails AI — Layer them
```

### Comparison

| Tool | Best For | Key Features | Latency |
|------|----------|--------------|---------|
| **Lakera** | Prompt injection | Real-time detection, PII, jailbreak | ~50-100ms |
| **Guardrails AI** | Output validation | RAIL specs, structural checks | ~10-50ms |
| **NeMo Guardrails** | Dialog policies | Conversation flow, topic control | ~20-100ms |

### Recommendation

| Situation | Use |
|-----------|-----|
| User-facing chatbot | **Lakera** — Prompt injection is #1 risk |
| API with structured output | **Guardrails AI** — Validate responses |
| Sensitive domain (healthcare, finance) | **Lakera + Guardrails AI** — Defense in depth |
| Internal tools only | Optional — Lower risk |

```python
# Lakera (recommended for user-facing)
from fastagentic.integrations.lakera import LakeraHook

LakeraHook(
    categories=["prompt_injection", "jailbreak", "pii"],
    on_detection="reject",  # Block attacks
    on_failure="reject",    # Fail-closed in production
)

# Guardrails AI (for output validation)
from fastagentic.integrations.guardrails import GuardrailsAIHook

GuardrailsAIHook(
    rail_spec="...",
    on_failure="retry",  # Try again with valid output
)
```

---

## Category 3: Memory

### Why You Need It

Without memory, your agent:
- Can't remember user preferences
- Treats every conversation as new
- Can't provide personalized responses
- Has no long-term context

### Decision Tree

```
What type of memory do you need?
│
├─► "Remember users across sessions"
│   └─► Mem0 — Long-term memory with semantic search
│
├─► "Summarize long conversations"
│   └─► Zep — Session memory with auto-summarization
│
├─► "Simple key-value storage"
│   └─► Redis — Fast, no semantic search
│
└─► "All user data in my control"
    └─► Mem0 self-hosted or Postgres
```

### Comparison

| Tool | Best For | Key Features | Semantic Search |
|------|----------|--------------|-----------------|
| **Mem0** | Long-term personalization | Auto-extraction, entity tracking | Yes |
| **Zep** | Session memory | Summarization, conversation history | Yes |
| **Redis** | Simple storage | Fast, lightweight | No |
| **Postgres** | Enterprise, custom | pgvector support, full control | Optional |

### Recommendation

| Situation | Use |
|-----------|-----|
| "Remember that I like dark mode" | **Mem0** — Extracts and stores user preferences |
| Long chat sessions | **Zep** — Summarizes to fit context window |
| Simple state (no search) | **Redis** — Lightweight and fast |
| Enterprise compliance | **Postgres** — Self-hosted, full control |

```python
# Mem0 (recommended for personalization)
from fastagentic.integrations.mem0 import Mem0Provider

Mem0Provider(
    api_key="...",
    # Automatically extracts and stores user context
)

# Zep (recommended for sessions)
from fastagentic.integrations.zep import ZepProvider

ZepProvider(
    api_key="...",
    auto_summarize=True,  # Handle long conversations
)

# Redis (simple option)
from fastagentic.memory import RedisProvider

RedisProvider(
    url="redis://localhost:6379",
    ttl_seconds=3600,  # Expire after 1 hour
)
```

---

## Category 4: LLM Gateway

### Why You Need It

Direct LLM calls have risks:
- OpenAI goes down → your app breaks
- Rate limits hit → requests fail
- No caching → redundant costs
- Single provider → no negotiating leverage

### Decision Tree

```
What do you need?
│
├─► "Fallbacks when OpenAI fails"
│   └─► Portkey or LiteLLM — Multi-provider routing
│
├─► "Cache similar requests"
│   └─► Portkey — Semantic caching
│
├─► "Load balance across providers"
│   └─► Portkey — Weighted distribution
│
├─► "Open source self-hosted"
│   └─► LiteLLM — No external dependency
│
└─► "Simple retry/timeout"
    └─► Built-in RetryPolicy — No gateway needed
```

### Comparison

| Tool | Best For | Key Features | Self-Hosted |
|------|----------|--------------|-------------|
| **Portkey** | Full gateway features | Fallbacks, caching, load balancing, analytics | No |
| **LiteLLM** | Open source | Multi-provider, simple | Yes |
| **Built-in** | Simple cases | Retry, timeout, basic fallback | N/A |

### Recommendation

| Situation | Use |
|-----------|-----|
| Production with high availability needs | **Portkey** — Full-featured gateway |
| Cost optimization (caching) | **Portkey** — Semantic cache |
| Open source requirement | **LiteLLM** — Self-hosted |
| Low traffic, simple needs | **Built-in** — RetryPolicy + Timeout |

```python
# Portkey (recommended for production)
from fastagentic.integrations.portkey import PortkeyGateway

PortkeyGateway(
    config={
        "strategy": {"mode": "fallback"},
        "targets": [
            {"virtual_key": "openai-key"},
            {"virtual_key": "anthropic-key"},
        ],
    },
    cache={"mode": "semantic", "max_age": 3600},
)

# LiteLLM (open source)
from fastagentic.integrations.litellm import LiteLLMGateway

LiteLLMGateway(
    fallback_models=["gpt-4o", "claude-3-sonnet", "gemini-pro"],
)

# Built-in (simple cases)
@agent_endpoint(
    retry=RetryPolicy(max_attempts=3, backoff="exponential"),
    timeout=Timeout(total_ms=60000),
)
```

---

## Category 5: Evaluation

### Why You Need It

Without evaluation:
- You don't know if your agent is improving
- No data for prompt optimization
- Can't catch regressions
- No way to compare approaches

### Decision Tree

```
What do you need to evaluate?
│
├─► "Track experiments, A/B test prompts"
│   └─► Braintrust — Experiment tracking, scoring
│
├─► "Evaluate based on traces"
│   └─► LangSmith — Trace-based evaluation
│
├─► "Inline quality checks"
│   └─► Custom LLMJudge — Real-time scoring
│
└─► "Production monitoring"
    └─► Any of the above with sampling
```

### Comparison

| Tool | Best For | Key Features | Real-time |
|------|----------|--------------|-----------|
| **Braintrust** | Experiments | Datasets, scoring, tracking | Optional |
| **LangSmith** | LangChain users | Trace evaluation, native integration | Optional |
| **Custom** | Specific needs | Your own scorers | Yes |

### Recommendation

| Situation | Use |
|-----------|-----|
| Iterating on prompts | **Braintrust** — Experiment tracking |
| LangChain ecosystem | **LangSmith** — Native integration |
| Real-time quality checks | **Custom LLMJudge** — Inline evaluation |
| Budget conscious | Sample 10% with any tool |

```python
# Braintrust (recommended for experiments)
from fastagentic.integrations.braintrust import BraintrustHook, LLMJudge

@agent_endpoint(
    eval_hooks=[
        BraintrustHook(
            project="my-agent",
            scores=[LLMJudge(criteria="Is the response helpful?")],
            sample_rate=0.1,  # Evaluate 10% in production
        ),
    ],
)

# LangSmith (for LangChain users)
from fastagentic.integrations.langsmith import LangSmithHook

@agent_endpoint(
    eval_hooks=[
        LangSmithHook(
            project="my-agent",
            evaluators=["relevance", "coherence"],
        ),
    ],
)
```

---

## Quick Reference: What to Use When

| Need | Tool | Why |
|------|------|-----|
| See what's happening | Langfuse | LLM-specific analytics |
| Block prompt injection | Lakera | Best detection accuracy |
| Remember users | Mem0 | Semantic memory extraction |
| Handle LLM failures | Portkey | Multi-provider fallbacks |
| Measure quality | Braintrust | Experiment tracking |

---

## Integration Priority by Stage

### Early Development
1. **Langfuse** — Debug and understand agent behavior

### Pre-Production
2. **Lakera** — Security is not optional
3. **RetryPolicy** — Handle transient failures

### Production
4. **Portkey** — High availability
5. **Mem0/Zep** — User experience

### Growth
6. **Braintrust** — Continuous improvement

---

## Cost Considerations

| Integration | Typical Cost | Notes |
|-------------|--------------|-------|
| Langfuse | Free tier available | Pay per trace |
| Lakera | Per-request | Often < LLM cost |
| Mem0 | Per-operation | Consider self-hosted |
| Portkey | Per-request or free tier | Caching saves money |
| Braintrust | Per-evaluation | Sample in production |

**Tip**: Most tools have free tiers sufficient for development and small-scale production.

---

## Next Steps

- [Production Checklist](production-checklist.md) — Verify you're ready
- [Hooks Architecture](../hooks.md) — How integrations work
- [Integration Index](../integrations/index.md) — Detailed setup guides
