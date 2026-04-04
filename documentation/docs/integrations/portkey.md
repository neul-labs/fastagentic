# Portkey Integration

[Portkey](https://portkey.ai) is an AI gateway providing fallbacks, caching, load balancing, and observability for LLM calls. FastAgentic integrates via the `PortkeyGateway`.

## Installation

```bash
pip install fastagentic[portkey]
```

## Quick Start

```python
from fastagentic import App
from fastagentic.integrations.portkey import PortkeyGateway

app = App(
    title="My Agent",
    llm_gateway=PortkeyGateway(api_key="..."),
)
```

## Configuration

### Environment Variables

```bash
export PORTKEY_API_KEY="..."
```

```python
app = App(llm_gateway=PortkeyGateway())  # Auto-reads from env
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `api_key` | Portkey API key | `$PORTKEY_API_KEY` |
| `virtual_key` | Virtual key for provider | None |
| `config` | Portkey config object | None |
| `cache` | Enable semantic caching | `false` |
| `retry` | Retry configuration | Default |
| `fallback` | Fallback models | None |

## Why Use a Gateway?

FastAgentic provides simple built-in reliability patterns. Portkey handles advanced scenarios:

| Feature | FastAgentic Built-in | Portkey |
|---------|---------------------|---------|
| Retry | Exponential backoff | Configurable strategies |
| Fallback | Model chain | Cross-provider, conditional |
| Caching | None | Semantic, exact match |
| Load balancing | None | Round-robin, least-latency |
| Observability | OTEL spans | Full dashboard, analytics |
| Rate limiting | Simple RPM/TPM | Advanced quotas |

## Fallback Configuration

```python
PortkeyGateway(
    api_key="...",
    config={
        "strategy": {
            "mode": "fallback",
        },
        "targets": [
            {"virtual_key": "openai-key", "weight": 1},
            {"virtual_key": "anthropic-key", "weight": 1},
            {"virtual_key": "azure-key", "weight": 1},
        ],
    },
)
```

When OpenAI fails, automatically tries Anthropic, then Azure.

## Semantic Caching

Cache semantically similar requests:

```python
PortkeyGateway(
    api_key="...",
    cache={
        "mode": "semantic",
        "max_age": 3600,  # 1 hour
    },
)
```

Benefits:
- Reduced latency for similar queries
- Lower costs
- More consistent responses

## Load Balancing

Distribute load across providers:

```python
PortkeyGateway(
    api_key="...",
    config={
        "strategy": {
            "mode": "loadbalance",
            "on_status_codes": [429, 500, 502, 503],
        },
        "targets": [
            {"virtual_key": "openai-1", "weight": 50},
            {"virtual_key": "openai-2", "weight": 30},
            {"virtual_key": "azure", "weight": 20},
        ],
    },
)
```

## Retry Configuration

```python
PortkeyGateway(
    api_key="...",
    retry={
        "attempts": 3,
        "on_status_codes": [429, 500, 502, 503, 504],
    },
)
```

## Conditional Routing

Route based on request properties:

```python
PortkeyGateway(
    api_key="...",
    config={
        "strategy": {
            "mode": "conditional",
            "conditions": [
                {
                    "query": {"model": "gpt-4"},
                    "then": "openai-key",
                },
                {
                    "query": {"model": "claude-3"},
                    "then": "anthropic-key",
                },
            ],
            "default": "openai-key",
        },
    },
)
```

## Per-Endpoint Configuration

```python
from fastagentic import agent_endpoint
from fastagentic.integrations.portkey import PortkeyGateway

@agent_endpoint(
    path="/premium",
    runnable=...,
    llm_gateway=PortkeyGateway(
        config={
            "strategy": {"mode": "fallback"},
            "targets": [
                {"virtual_key": "gpt4-key"},
                {"virtual_key": "claude-opus-key"},
            ],
        },
    ),
)
async def premium_endpoint(input: Input) -> Output:
    ...

@agent_endpoint(
    path="/standard",
    runnable=...,
    llm_gateway=PortkeyGateway(
        config={
            "targets": [{"virtual_key": "gpt35-key"}],
        },
    ),
)
async def standard_endpoint(input: Input) -> Output:
    ...
```

## Observability

Portkey provides built-in observability:

```python
PortkeyGateway(
    api_key="...",
    metadata={
        "environment": "production",
        "app": "my-agent",
        "_user": lambda ctx: ctx.user.id,  # Dynamic metadata
    },
    trace_id=lambda ctx: ctx.run_id,  # Link to FastAgentic traces
)
```

View in Portkey dashboard:
- Request/response logs
- Latency metrics
- Cost tracking
- Error analysis

## Integration with FastAgentic Hooks

Portkey works alongside FastAgentic hooks:

```python
from fastagentic import App
from fastagentic.integrations.portkey import PortkeyGateway
from fastagentic.integrations.langfuse import LangfuseHook

app = App(
    # Portkey handles LLM routing
    llm_gateway=PortkeyGateway(api_key="..."),

    # Langfuse handles application-level tracing
    hooks=[LangfuseHook(...)],
)
```

## Cost Tracking

Portkey tracks costs automatically. Access via hooks:

```python
from fastagentic.hooks import hook, HookContext

@hook("on_llm_end")
async def log_portkey_cost(ctx: HookContext):
    # Portkey adds cost to response metadata
    portkey_meta = ctx.metadata.get("portkey", {})
    print(f"Cost: ${portkey_meta.get('cost', 0):.4f}")
    print(f"Cache hit: {portkey_meta.get('cache_hit', False)}")
```

## Alternative: LiteLLM

For open-source multi-provider routing:

```python
from fastagentic.integrations.litellm import LiteLLMGateway

app = App(
    llm_gateway=LiteLLMGateway(
        fallback_models=["gpt-4o", "claude-3-sonnet", "gemini-pro"],
    ),
)
```

## Troubleshooting

### Fallbacks not triggering

- Check `on_status_codes` includes the error code
- Verify all virtual keys are configured in Portkey

### Cache not working

- Ensure `cache.mode` is set
- Check if requests are similar enough for semantic cache
- Verify `max_age` hasn't expired

### High latency

- Consider using nearest Portkey region
- Check if fallbacks are adding latency
- Review load balancing weights

## Next Steps

- [Portkey Docs](https://docs.portkey.ai)
- [Reliability Patterns](../reliability.md)
- [LiteLLM Integration](litellm.md)
