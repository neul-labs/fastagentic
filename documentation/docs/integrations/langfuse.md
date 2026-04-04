# Langfuse Integration

[Langfuse](https://langfuse.com) provides LLM observability with tracing, prompt analytics, and cost tracking. FastAgentic integrates via the `LangfuseHook`.

## Installation

```bash
pip install fastagentic[langfuse]
```

## Quick Start

```python
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook

app = App(
    title="My Agent",
    hooks=[
        LangfuseHook(
            public_key="pk-...",
            secret_key="sk-...",
            host="https://cloud.langfuse.com",  # or self-hosted
        ),
    ],
)
```

## Configuration

### Environment Variables

```bash
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

```python
# Auto-reads from environment
app = App(hooks=[LangfuseHook()])
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `public_key` | Langfuse public key | `$LANGFUSE_PUBLIC_KEY` |
| `secret_key` | Langfuse secret key | `$LANGFUSE_SECRET_KEY` |
| `host` | Langfuse host URL | `https://cloud.langfuse.com` |
| `release` | Release/version tag | App version |
| `sample_rate` | Trace sampling rate (0.0-1.0) | `1.0` |
| `flush_interval_ms` | Batch flush interval | `5000` |
| `on_failure` | Failure mode: `warn`, `reject` | `warn` |

## What Gets Traced

The `LangfuseHook` captures:

| Event | Langfuse Concept | Data |
|-------|------------------|------|
| `on_request` | Trace start | User ID, session, metadata |
| `on_llm_start` | Generation start | Model, messages, temperature |
| `on_llm_end` | Generation end | Response, tokens, cost, latency |
| `on_tool_call` | Span start | Tool name, arguments |
| `on_tool_result` | Span end | Result, duration |
| `on_response` | Trace end | Final output, total cost |
| `on_error` | Error event | Exception details |

## Example Trace

```python
@agent_endpoint(
    path="/triage",
    runnable=PydanticAIAdapter(agent),
    a2a_skill="support-triage",
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

Creates this trace in Langfuse:

```
Trace: triage (run_abc123)
├── Generation: gpt-4o
│   ├── Input: [system, user messages]
│   ├── Output: [assistant response]
│   ├── Tokens: 150 in, 80 out
│   └── Cost: $0.0023
├── Span: tool:search_knowledge
│   ├── Input: {"query": "..."}
│   ├── Output: [results]
│   └── Duration: 234ms
├── Generation: gpt-4o
│   └── ...
└── Total: 450ms, $0.0051
```

## User and Session Tracking

```python
from fastagentic.integrations.langfuse import LangfuseHook

LangfuseHook(
    # Map user from auth context
    user_id=lambda ctx: ctx.user.id if ctx.user else "anonymous",
    # Map session
    session_id=lambda ctx: ctx.request.headers.get("X-Session-ID"),
    # Custom metadata
    metadata=lambda ctx: {
        "tenant": ctx.tenant,
        "environment": os.getenv("ENV"),
    },
)
```

## Prompt Management Integration

Use Langfuse prompts with FastAgentic:

```python
from fastagentic.integrations.langfuse import LangfusePromptProvider

app = App(
    prompt_provider=LangfusePromptProvider(
        public_key="...",
        secret_key="...",
    ),
)

# Prompts fetched from Langfuse
@prompt(name="triage_system", provider="langfuse")
def triage_prompt() -> str:
    ...  # Returns prompt from Langfuse
```

## Scores and Feedback

Send evaluation scores back to Langfuse:

```python
from fastagentic.hooks import hook, HookContext

@hook("on_response")
async def score_response(ctx: HookContext):
    # Automatic quality scoring
    score = await evaluate_quality(ctx.response)

    ctx.langfuse.score(
        name="quality",
        value=score,
        comment="Automated quality check",
    )
```

## Cost Tracking

Langfuse automatically tracks costs. FastAgentic enriches with:

- Per-user cost attribution
- Per-tenant cost rollups
- Model-specific breakdowns

```python
# Access cost in hooks
@hook("on_llm_end")
async def log_cost(ctx: HookContext):
    print(f"This call cost: ${ctx.usage.cost:.4f}")
    print(f"Session total: ${ctx.metadata['session_cost']:.4f}")
```

## Self-Hosted Langfuse

```python
LangfuseHook(
    host="https://langfuse.your-company.com",
    public_key="...",
    secret_key="...",
)
```

## Performance Considerations

- Traces are batched and sent asynchronously
- Does not block request/response flow
- Sampling available for high-volume endpoints

```python
LangfuseHook(
    sample_rate=0.1,  # Only trace 10% of requests
    flush_interval_ms=10000,  # Batch for 10 seconds
)
```

## Troubleshooting

### Traces not appearing

1. Check API keys are correct
2. Verify host URL (especially for self-hosted)
3. Check network connectivity
4. Enable debug logging: `LANGFUSE_DEBUG=true`

### High memory usage

Reduce batch size and flush more frequently:

```python
LangfuseHook(
    flush_interval_ms=1000,
    max_batch_size=50,
)
```

## Next Steps

- [Langfuse Docs](https://langfuse.com/docs)
- [Hooks Architecture](../hooks.md)
- [Observability Guide](../operations/observability/index.md)
