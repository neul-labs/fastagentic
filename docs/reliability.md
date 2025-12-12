# Reliability Patterns

FastAgentic provides lightweight reliability patterns for agent execution. These are intentionally simple—for advanced resilience, integrate with battle-tested libraries like [Tenacity](https://github.com/jd/tenacity), [CircuitBreaker](https://github.com/fabfuel/circuitbreaker), or use an LLM gateway like [Portkey](https://portkey.ai).

## Philosophy

| Pattern | FastAgentic Builds | Consider Instead |
|---------|-------------------|------------------|
| **Retry** | Simple exponential backoff | Tenacity (complex policies) |
| **Circuit Breaker** | Basic state machine | pybreaker, CircuitBreaker lib |
| **Timeout** | Per-endpoint limits | Handled by async frameworks |
| **Fallback** | Model chain configuration | Portkey, LiteLLM (advanced routing) |
| **Rate Limiting** | Simple RPM/TPM counter | Redis-based, API gateways |

FastAgentic's built-in patterns cover 80% of use cases. For the other 20%, plug in specialized tools.

---

## Retry Policy

Automatically retry failed LLM calls and tool executions.

### Configuration

```python
from fastagentic import agent_endpoint
from fastagentic.reliability import RetryPolicy

@agent_endpoint(
    path="/analyze",
    runnable=...,
    retry=RetryPolicy(
        max_attempts=3,
        backoff="exponential",      # or "fixed", "linear"
        initial_delay_ms=1000,      # 1 second
        max_delay_ms=30000,         # 30 seconds cap
        jitter=True,                # Add randomness to prevent thundering herd
        retry_on=[
            "rate_limit",           # Retry on rate limit errors
            "timeout",              # Retry on timeouts
            "server_error",         # Retry on 5xx errors
        ],
        # Don't retry these
        no_retry_on=[
            "invalid_input",        # Client errors
            "auth_error",           # Auth failures
        ],
    ),
)
async def analyze(data: DataInput) -> AnalysisResult:
    ...
```

### Backoff Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `fixed` | `delay = initial_delay` | Predictable retry timing |
| `linear` | `delay = initial_delay * attempt` | Gradual increase |
| `exponential` | `delay = initial_delay * 2^attempt` | API rate limits (recommended) |

### Jitter

Jitter adds randomness to prevent multiple clients retrying simultaneously:

```python
# Without jitter: All clients retry at exactly 1s, 2s, 4s...
# With jitter: Client A at 1.2s, Client B at 0.9s, Client C at 1.1s...

RetryPolicy(
    backoff="exponential",
    jitter=True,           # ±25% randomness
    jitter_factor=0.25,    # Customize jitter range
)
```

### Retry Context in Hooks

```python
from fastagentic.hooks import hook, HookContext

@hook("on_retry")
async def log_retry(ctx: HookContext):
    print(f"Retry attempt {ctx.retry_count}/{ctx.max_retries}")
    print(f"Error: {ctx.error}")
    print(f"Next delay: {ctx.next_delay_ms}ms")

    # Optionally modify retry behavior
    if ctx.retry_count > 2:
        # Switch to cheaper model on repeated failures
        ctx.metadata["model_override"] = "gpt-3.5-turbo"
```

### Retry Metrics

FastAgentic exposes retry metrics:

```
fastagentic_retries_total{endpoint="/analyze", attempt="1"} 150
fastagentic_retries_total{endpoint="/analyze", attempt="2"} 23
fastagentic_retries_total{endpoint="/analyze", attempt="3"} 5
fastagentic_retry_exhausted_total{endpoint="/analyze"} 2
```

---

## Circuit Breaker

Prevent cascading failures by stopping calls to failing services.

### Configuration

```python
from fastagentic import agent_endpoint
from fastagentic.reliability import CircuitBreaker

@agent_endpoint(
    path="/external-api",
    runnable=...,
    circuit_breaker=CircuitBreaker(
        # Failure threshold to open circuit
        failure_threshold=5,        # Open after 5 failures
        failure_window_ms=60000,    # Within 60 seconds

        # Recovery
        reset_timeout_ms=30000,     # Try again after 30 seconds
        half_open_requests=2,       # Allow 2 test requests in half-open

        # Success threshold to close
        success_threshold=3,        # Close after 3 successes in half-open

        # What counts as failure
        failure_on=[
            "timeout",
            "server_error",
            "connection_error",
        ],
    ),
)
async def call_external(request: Request) -> Response:
    ...
```

### Circuit States

```
     ┌──────────────────────────────────────────────────┐
     │                                                   │
     ▼                                                   │
┌─────────┐  failure_threshold  ┌─────────┐             │
│ CLOSED  │ ──────────────────► │  OPEN   │             │
└────┬────┘                     └────┬────┘             │
     │                               │                   │
     │ success                       │ reset_timeout     │
     │                               ▼                   │
     │                         ┌───────────┐            │
     │                         │ HALF-OPEN │            │
     │                         └─────┬─────┘            │
     │                               │                   │
     │      success_threshold        │ failure          │
     └───────────────────────────────┴───────────────────┘
```

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation, requests pass through |
| **OPEN** | Requests fail immediately, no calls made |
| **HALF-OPEN** | Limited test requests allowed |

### Circuit Breaker Response

When circuit is open:

```python
from fastagentic.reliability import CircuitOpenError

try:
    result = await call_external(request)
except CircuitOpenError as e:
    # Circuit is open, fail fast
    print(f"Circuit open since: {e.opened_at}")
    print(f"Will retry at: {e.retry_after}")

    # Return cached/fallback response
    return cached_response
```

### Per-Dependency Circuits

```python
from fastagentic.reliability import CircuitBreakerRegistry

# Shared circuit breaker registry
circuits = CircuitBreakerRegistry()

@agent_endpoint(
    path="/multi-api",
    runnable=...,
    circuit_breakers={
        "openai": CircuitBreaker(failure_threshold=5),
        "anthropic": CircuitBreaker(failure_threshold=3),
        "database": CircuitBreaker(failure_threshold=10),
    },
)
async def multi_call(request: Request) -> Response:
    ...
```

### Circuit Metrics

```
fastagentic_circuit_state{endpoint="/external-api", state="closed"} 1
fastagentic_circuit_state{endpoint="/external-api", state="open"} 0
fastagentic_circuit_failures_total{endpoint="/external-api"} 127
fastagentic_circuit_opens_total{endpoint="/external-api"} 3
```

---

## Timeout

Enforce time limits on agent execution.

### Configuration

```python
from fastagentic import agent_endpoint
from fastagentic.reliability import Timeout

@agent_endpoint(
    path="/analyze",
    runnable=...,
    timeout=Timeout(
        total_ms=300000,            # 5 minutes total
        llm_call_ms=60000,          # 1 minute per LLM call
        tool_call_ms=30000,         # 30 seconds per tool
        checkpoint_ms=5000,         # 5 seconds for checkpointing
    ),
)
async def analyze(data: DataInput) -> AnalysisResult:
    ...
```

### Timeout Levels

| Level | Scope | Default |
|-------|-------|---------|
| `total_ms` | Entire request | 120000 (2 min) |
| `llm_call_ms` | Single LLM API call | 60000 (1 min) |
| `tool_call_ms` | Single tool execution | 30000 (30 sec) |
| `checkpoint_ms` | Checkpoint save/load | 5000 (5 sec) |

### Timeout Behavior

```python
from fastagentic.reliability import TimeoutError

try:
    result = await analyze(data)
except TimeoutError as e:
    print(f"Timeout after {e.elapsed_ms}ms")
    print(f"Stage: {e.stage}")  # "llm_call", "tool_call", "total"

    # If durable, can resume later
    if e.run_id:
        print(f"Resume with: POST /runs/{e.run_id}/resume")
```

### Graceful Timeout with Checkpoints

```python
@agent_endpoint(
    path="/long-task",
    runnable=...,
    durable=True,
    timeout=Timeout(
        total_ms=60000,
        # On timeout, save checkpoint instead of failing
        on_timeout="checkpoint",  # or "fail", "warn"
    ),
)
async def long_task(input: Input) -> Output:
    ...
```

---

## Fallback Chains

Automatically fall back to alternative models or strategies.

### Model Fallback

```python
from fastagentic import App
from fastagentic.reliability import FallbackChain

app = App(
    title="My Agent",
    # Global fallback chain
    model_fallback=FallbackChain(
        primary="gpt-4o",
        fallbacks=[
            {"model": "gpt-4o-mini", "on": ["rate_limit", "timeout"]},
            {"model": "gpt-3.5-turbo", "on": ["rate_limit", "timeout", "server_error"]},
        ],
        # Cost tracking across fallbacks
        track_fallback_costs=True,
    ),
)
```

### Per-Endpoint Fallback

```python
@agent_endpoint(
    path="/critical",
    runnable=...,
    fallback=FallbackChain(
        primary="claude-3-opus",
        fallbacks=[
            {"model": "claude-3-sonnet", "on": ["rate_limit"]},
            {"model": "gpt-4o", "on": ["rate_limit", "timeout"]},  # Cross-provider
        ],
    ),
)
async def critical_task(input: Input) -> Output:
    ...
```

### Strategy Fallback

```python
from fastagentic.reliability import StrategyFallback

@agent_endpoint(
    path="/flexible",
    runnable=...,
    fallback=StrategyFallback(
        strategies=[
            # Try full agent first
            {"runnable": full_agent, "timeout_ms": 30000},
            # Fall back to simpler chain
            {"runnable": simple_chain, "timeout_ms": 10000},
            # Last resort: cached response
            {"runnable": cached_lookup, "timeout_ms": 1000},
        ],
    ),
)
async def flexible_task(input: Input) -> Output:
    ...
```

### Fallback Metrics

```
fastagentic_fallback_triggered_total{endpoint="/critical", from="claude-3-opus", to="gpt-4o"} 15
fastagentic_fallback_success_total{endpoint="/critical", model="gpt-4o"} 14
fastagentic_fallback_exhausted_total{endpoint="/critical"} 1
```

---

## Rate Limiting

Simple rate limiting for agent endpoints.

### Configuration

```python
from fastagentic import App
from fastagentic.reliability import RateLimiter

app = App(
    title="My Agent",
    rate_limiter=RateLimiter(
        # Global limits
        requests_per_minute=100,
        tokens_per_minute=100000,

        # Per-user limits (if authenticated)
        per_user_rpm=10,
        per_user_tpm=10000,

        # Per-tenant limits
        per_tenant_rpm=50,
        per_tenant_tpm=50000,

        # Behavior when limited
        on_limit="reject",  # or "queue", "delay"
    ),
)
```

### Per-Endpoint Limits

```python
@agent_endpoint(
    path="/expensive",
    runnable=...,
    rate_limit={
        "requests_per_minute": 10,
        "tokens_per_minute": 50000,
    },
)
async def expensive_task(input: Input) -> Output:
    ...
```

### Rate Limit Response

```python
from fastagentic.reliability import RateLimitError

try:
    result = await expensive_task(input)
except RateLimitError as e:
    print(f"Rate limited: {e.limit_type}")  # "rpm", "tpm"
    print(f"Retry after: {e.retry_after_ms}ms")
    print(f"Current usage: {e.current}/{e.limit}")
```

### Rate Limit Headers

FastAgentic returns standard rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 23
X-RateLimit-Reset: 1699876543
Retry-After: 45
```

---

## Combining Patterns

Patterns can be combined:

```python
@agent_endpoint(
    path="/resilient",
    runnable=...,

    # Retry on transient failures
    retry=RetryPolicy(
        max_attempts=3,
        backoff="exponential",
        retry_on=["rate_limit", "timeout"],
    ),

    # Circuit breaker for downstream protection
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        reset_timeout_ms=30000,
    ),

    # Timeout to prevent runaway execution
    timeout=Timeout(
        total_ms=120000,
        llm_call_ms=30000,
    ),

    # Fallback when primary fails
    fallback=FallbackChain(
        primary="gpt-4o",
        fallbacks=[{"model": "gpt-4o-mini", "on": ["rate_limit"]}],
    ),

    # Rate limiting
    rate_limit={"requests_per_minute": 50},
)
async def resilient_task(input: Input) -> Output:
    ...
```

### Execution Order

```
Request
   │
   ▼
┌─────────────┐
│ Rate Limit  │ ─── Reject if over limit
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Timeout   │ ─── Start timer
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Circuit Break│ ─── Fail fast if open
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Execute   │ ─── Run agent
└──────┬──────┘
       │
   ┌───┴───┐
   │Success│
   └───┬───┘
       │
       ▼
   Response

   │Failure│
   └───┬───┘
       │
       ▼
┌─────────────┐
│   Retry?    │ ─── If retries remain
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Fallback?  │ ─── If fallback configured
└──────┬──────┘
       │
       ▼
   Error Response
```

---

## When to Use External Tools

FastAgentic's built-in patterns are intentionally simple. Use external tools when you need:

| Need | Use |
|------|-----|
| Complex retry policies (fibonacci, custom) | [Tenacity](https://github.com/jd/tenacity) |
| Distributed circuit breakers | [pybreaker](https://github.com/danielfm/pybreaker) + Redis |
| Advanced rate limiting | Redis + [limits](https://github.com/alisaifee/limits) |
| LLM-specific routing | [Portkey](https://portkey.ai), [LiteLLM](https://github.com/BerriAI/litellm) |
| Full observability | [Langfuse](https://langfuse.com), [Datadog](https://datadoghq.com) |

### Portkey Integration Example

```python
from fastagentic import App
from fastagentic.integrations.portkey import PortkeyGateway

app = App(
    title="My Agent",
    # Use Portkey for advanced LLM routing
    llm_gateway=PortkeyGateway(
        api_key="...",
        # Portkey handles: retries, fallbacks, caching, load balancing
        config={
            "retry": {"attempts": 3, "on_status_codes": [429, 500, 502, 503]},
            "cache": {"mode": "semantic", "max_age": 3600},
            "loadbalance": {"strategy": "round-robin"},
        },
    ),
)
```

---

## Next Steps

- [Hooks Architecture](hooks.md) — Intercept retry/circuit events
- [Integrations](integrations/index.md) — Portkey, LiteLLM setup
- [Observability](operations/observability/index.md) — Monitor reliability metrics
