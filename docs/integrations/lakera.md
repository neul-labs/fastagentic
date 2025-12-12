# Lakera Integration

[Lakera Guard](https://lakera.ai) provides prompt injection detection and content moderation. FastAgentic integrates via the `LakeraHook` for real-time protection.

## Installation

```bash
pip install fastagentic[lakera]
```

## Quick Start

```python
from fastagentic import App
from fastagentic.integrations.lakera import LakeraHook

app = App(
    title="My Agent",
    hooks=[
        LakeraHook(api_key="..."),
    ],
)
```

## Configuration

### Environment Variables

```bash
export LAKERA_API_KEY="..."
```

```python
app = App(hooks=[LakeraHook()])  # Auto-reads from env
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `api_key` | Lakera API key | `$LAKERA_API_KEY` |
| `endpoint` | API endpoint | `https://api.lakera.ai` |
| `categories` | Detection categories to enable | All |
| `on_failure` | Hook failure mode: `warn`, `reject` | `reject` |
| `on_detection` | Detection mode: `reject`, `warn`, `flag` | `reject` |
| `timeout_ms` | API timeout | `5000` |

## What Gets Checked

The `LakeraHook` runs on `on_llm_start` and checks:

| Category | Description |
|----------|-------------|
| `prompt_injection` | Attempts to override system prompts |
| `jailbreak` | Attempts to bypass safety measures |
| `pii` | Personal identifiable information |
| `toxic_content` | Harmful or offensive content |
| `relevant_language` | Off-topic or irrelevant input |

## Detection Behavior

### Reject (Default)

Block the request immediately:

```python
LakeraHook(
    on_detection="reject",  # Return 400 error
)
```

Response:

```json
{
  "error": "Content policy violation",
  "code": "GUARDRAIL_REJECTED",
  "category": "prompt_injection"
}
```

### Warn

Log and continue:

```python
LakeraHook(
    on_detection="warn",  # Log warning, continue execution
)
```

### Flag

Continue but mark in metadata:

```python
LakeraHook(
    on_detection="flag",  # Add flag to response metadata
)

# In response:
# {"result": "...", "metadata": {"guardrail_flags": ["prompt_injection"]}}
```

## Category Configuration

Enable specific categories:

```python
LakeraHook(
    categories=[
        "prompt_injection",  # Always recommended
        "jailbreak",
        # "pii",  # Disable PII detection
        # "toxic_content",
    ],
)
```

## Per-Endpoint Configuration

```python
@agent_endpoint(
    path="/public",
    runnable=...,
    pre_hooks=[
        LakeraHook(
            on_detection="reject",
            categories=["prompt_injection", "jailbreak", "toxic_content"],
        ),
    ],
)
async def public_endpoint(input: Input) -> Output:
    ...

@agent_endpoint(
    path="/internal",
    runnable=...,
    pre_hooks=[
        LakeraHook(
            on_detection="warn",  # Less strict for internal
            categories=["prompt_injection"],
        ),
    ],
)
async def internal_endpoint(input: Input) -> Output:
    ...
```

## Custom Handling

```python
from fastagentic.hooks import hook, HookContext, HookResult
from fastagentic.integrations.lakera import LakeraClient

lakera = LakeraClient(api_key="...")

@hook("on_llm_start")
async def custom_guardrail(ctx: HookContext) -> HookResult:
    result = await lakera.check(ctx.messages)

    if result.flagged:
        # Custom logging
        await audit_log.record(
            user=ctx.user.id,
            category=result.category,
            input=ctx.messages,
        )

        if result.category == "prompt_injection":
            return HookResult.reject("Prompt injection detected")
        else:
            # Allow with warning
            ctx.metadata["guardrail_warning"] = result.category
            return HookResult.proceed()

    return HookResult.proceed()
```

## Fail-Open vs Fail-Closed

### Fail-Closed (Recommended for Production)

If Lakera API is unavailable, block the request:

```python
LakeraHook(
    on_failure="reject",  # API errors = blocked
)
```

### Fail-Open

If Lakera API is unavailable, log and continue:

```python
LakeraHook(
    on_failure="warn",  # API errors = continue with warning
)
```

## Metrics

FastAgentic exposes Lakera metrics:

```
fastagentic_guardrail_checks_total{provider="lakera"} 10523
fastagentic_guardrail_rejections_total{provider="lakera", category="prompt_injection"} 47
fastagentic_guardrail_latency_ms{provider="lakera", quantile="p99"} 89
```

## Caching

Reduce API calls with caching:

```python
LakeraHook(
    cache_enabled=True,
    cache_ttl_seconds=300,  # 5 minute cache
    cache_max_size=1000,
)
```

## Rate Limiting

Handle Lakera rate limits gracefully:

```python
LakeraHook(
    rate_limit_behavior="queue",  # or "fail", "skip"
    max_concurrent_requests=10,
)
```

## Troubleshooting

### High latency

- Enable caching for repeated inputs
- Increase timeout if network is slow
- Check if categories can be reduced

### False positives

- Review detection categories
- Use `flag` mode instead of `reject`
- Implement custom post-processing

## Next Steps

- [Lakera Docs](https://docs.lakera.ai)
- [Guardrails AI Integration](guardrails-ai.md)
- [Security Guide](../operations/security/index.md)
