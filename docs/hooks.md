# Hooks Architecture

FastAgentic provides a fine-grained hook system that allows you to intercept, observe, and modify agent execution at multiple lifecycle points. Hooks enable integration with external tools for observability, guardrails, evaluation, and memory without coupling your agent logic to specific vendors.

## Philosophy

FastAgentic owns the deployment layer. Specialized tools handle their domains better:

- **Observability** → Langfuse, Logfire, Datadog
- **Guardrails** → Lakera, Guardrails AI, NeMo
- **Evaluation** → Braintrust, LangSmith
- **Memory** → Mem0, Zep

Hooks are the integration points. You choose the tools.

## Hook Lifecycle

```
Request Flow
════════════════════════════════════════════════════════════════════

  ┌─────────────┐
  │  on_request │ ← Transform/validate incoming request
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     ┌─────────────────┐
  │on_llm_start │ ──► │ Guardrail Hooks │ ← Pre-LLM checks (Lakera)
  └──────┬──────┘     └─────────────────┘
         │
         ▼
  ┌─────────────┐     ┌─────────────────┐
  │on_tool_call │ ──► │ Tool Validation │ ← Per-tool interception
  └──────┬──────┘     └─────────────────┘
         │
         ▼
  ┌──────────────┐
  │on_tool_result│ ← Tool output processing
  └──────┬───────┘
         │
         ▼
  ┌─────────────┐     ┌─────────────────┐
  │ on_llm_end  │ ──► │   Eval Hooks    │ ← Post-LLM scoring (Braintrust)
  └──────┬──────┘     └─────────────────┘
         │
         ▼
  ┌─────────────┐
  │on_checkpoint│ ← Durability snapshot (if durable=True)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     ┌─────────────────┐
  │ on_response │ ──► │ Output Guardrails│ ← Response validation
  └─────────────┘     └─────────────────┘


Error Flow
════════════════════════════════════════════════════════════════════

  ┌─────────────┐
  │  on_error   │ ← Capture and log errors
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  on_retry   │ ← Before retry attempt (if RetryPolicy configured)
  └──────┬──────┘
         │
         ▼
      Resume flow or fail


LangGraph-Specific
════════════════════════════════════════════════════════════════════

  ┌───────────────┐
  │ on_node_enter │ ← Before node execution
  └───────┬───────┘
          │
          ▼
    [Node Execution]
          │
          ▼
  ┌──────────────┐
  │ on_node_exit │ ← After node execution, before next node
  └──────────────┘
```

## Hook Types

### Lifecycle Hooks

Core hooks that fire during agent execution:

| Hook | When | Use Cases |
|------|------|-----------|
| `on_request` | Request received | Auth enrichment, input validation, rate limiting |
| `on_response` | Response ready | Output filtering, response transformation |
| `on_llm_start` | Before LLM call | Prompt logging, guardrail checks |
| `on_llm_end` | After LLM call | Token tracking, cost attribution, eval scoring |
| `on_tool_call` | Before tool execution | Tool authorization, parameter validation |
| `on_tool_result` | After tool execution | Result caching, output filtering |
| `on_checkpoint` | Checkpoint created | Durability logging, state inspection |
| `on_resume` | Resuming from checkpoint | State validation, context refresh |
| `on_error` | Error occurred | Error logging, alerting, recovery |
| `on_retry` | Before retry | Backoff logging, circuit breaker checks |

### Framework-Specific Hooks

Additional hooks for specific adapters:

| Hook | Adapter | When |
|------|---------|------|
| `on_node_enter` | LangGraph | Before graph node execution |
| `on_node_exit` | LangGraph | After graph node execution |
| `on_agent_start` | CrewAI | Before individual agent in crew |
| `on_agent_end` | CrewAI | After individual agent in crew |
| `on_task_start` | CrewAI | Before crew task |
| `on_task_end` | CrewAI | After crew task |

## Registering Hooks

### Via App Configuration

```python
from fastagentic import App
from fastagentic.hooks import LangfuseHook, LakeraHook

app = App(
    title="My Agent",
    hooks=[
        LangfuseHook(public_key="...", secret_key="..."),
        LakeraHook(api_key="..."),
    ],
)
```

### Via Endpoint Decorator

```python
from fastagentic import agent_endpoint
from fastagentic.hooks import GuardrailsAIHook, BraintrustHook

@agent_endpoint(
    path="/triage",
    runnable=...,
    pre_hooks=[
        LakeraHook(api_key="..."),  # Before execution
    ],
    post_hooks=[
        GuardrailsAIHook(rail_spec="validation.rail"),  # After execution
    ],
    eval_hooks=[
        BraintrustHook(project="triage"),  # Async evaluation
    ],
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

### Via Decorator

```python
from fastagentic.hooks import hook

@hook("on_llm_end")
async def log_tokens(ctx: HookContext):
    print(f"Tokens used: {ctx.usage.total_tokens}")
    print(f"Cost: ${ctx.usage.cost:.4f}")
```

## Hook Context

Every hook receives a `HookContext` with relevant metadata:

```python
from fastagentic.hooks import HookContext

@hook("on_llm_end")
async def my_hook(ctx: HookContext):
    # Run information
    ctx.run_id          # Unique run identifier
    ctx.endpoint        # Endpoint path
    ctx.adapter         # Adapter name (pydanticai, langgraph, etc.)

    # User information
    ctx.user            # Authenticated user (if OIDC configured)
    ctx.tenant          # Tenant identifier (if multi-tenant)
    ctx.scopes          # OAuth scopes

    # Request/Response
    ctx.request         # Original request data
    ctx.response        # Response data (in post hooks)

    # LLM-specific (in on_llm_* hooks)
    ctx.model           # Model name
    ctx.messages        # Messages sent to LLM
    ctx.usage           # Token usage and cost
    ctx.usage.input_tokens
    ctx.usage.output_tokens
    ctx.usage.total_tokens
    ctx.usage.cost

    # Tool-specific (in on_tool_* hooks)
    ctx.tool_name       # Tool being called
    ctx.tool_args       # Tool arguments
    ctx.tool_result     # Tool result (in on_tool_result)

    # Graph-specific (in on_node_* hooks)
    ctx.node_name       # Current node
    ctx.graph_state     # Current graph state

    # Checkpoint (in on_checkpoint/on_resume)
    ctx.checkpoint_id   # Checkpoint identifier
    ctx.checkpoint_data # Serialized state

    # Error (in on_error/on_retry)
    ctx.error           # Exception object
    ctx.retry_count     # Current retry attempt
    ctx.max_retries     # Configured max retries

    # Timing
    ctx.started_at      # Request start time
    ctx.duration_ms     # Duration so far (in post hooks)

    # Metadata
    ctx.metadata        # Custom metadata dict (mutable)
```

## Hook Return Values

Hooks can return values to modify execution:

```python
from fastagentic.hooks import HookContext, HookResult

@hook("on_request")
async def validate_input(ctx: HookContext) -> HookResult:
    if contains_pii(ctx.request):
        # Block execution
        return HookResult.reject("Request contains PII")

    # Modify request
    return HookResult.modify(request=sanitize(ctx.request))

@hook("on_tool_call")
async def authorize_tool(ctx: HookContext) -> HookResult:
    if ctx.tool_name == "delete_record" and not ctx.user.is_admin:
        # Skip this tool call
        return HookResult.skip("Unauthorized tool access")

    # Continue normally
    return HookResult.proceed()

@hook("on_response")
async def filter_output(ctx: HookContext) -> HookResult:
    # Modify response before returning
    filtered = redact_sensitive(ctx.response)
    return HookResult.modify(response=filtered)
```

### HookResult Options

| Result | Effect |
|--------|--------|
| `HookResult.proceed()` | Continue execution normally |
| `HookResult.modify(...)` | Continue with modified data |
| `HookResult.skip(reason)` | Skip current operation (tool/node) |
| `HookResult.reject(reason)` | Abort execution with error |
| `HookResult.retry(after_ms)` | Trigger retry after delay |

## Async vs Sync Execution

### Blocking Hooks

Most hooks block execution until complete:

```python
@hook("on_llm_start")
async def guardrail_check(ctx: HookContext) -> HookResult:
    # This runs before the LLM call proceeds
    result = await lakera.check(ctx.messages)
    if result.is_unsafe:
        return HookResult.reject("Content policy violation")
    return HookResult.proceed()
```

### Non-Blocking Hooks

Eval hooks run asynchronously and don't block the response:

```python
@agent_endpoint(
    path="/chat",
    runnable=...,
    eval_hooks=[
        BraintrustHook(project="chat"),  # Runs async, doesn't delay response
    ],
)
```

To make any hook non-blocking:

```python
from fastagentic.hooks import hook, HookMode

@hook("on_response", mode=HookMode.ASYNC)
async def log_analytics(ctx: HookContext):
    # Fire and forget - doesn't block response
    await analytics.track(ctx.run_id, ctx.usage)
```

## Writing Custom Hooks

### Simple Function Hook

```python
from fastagentic.hooks import hook, HookContext

@hook("on_llm_end")
async def track_costs(ctx: HookContext):
    await cost_tracker.record(
        user=ctx.user.id,
        model=ctx.model,
        tokens=ctx.usage.total_tokens,
        cost=ctx.usage.cost,
    )
```

### Hook Class

For hooks with configuration or state:

```python
from fastagentic.hooks import BaseHook, HookContext

class CostAlertHook(BaseHook):
    hooks = ["on_llm_end"]  # Which lifecycle points to listen

    def __init__(self, threshold: float, slack_webhook: str):
        self.threshold = threshold
        self.slack_webhook = slack_webhook
        self.session_cost = 0.0

    async def on_llm_end(self, ctx: HookContext):
        self.session_cost += ctx.usage.cost

        if self.session_cost > self.threshold:
            await self.send_alert(ctx)

    async def send_alert(self, ctx: HookContext):
        await httpx.post(self.slack_webhook, json={
            "text": f"Cost alert: Run {ctx.run_id} exceeded ${self.threshold}"
        })

# Usage
app = App(
    hooks=[CostAlertHook(threshold=1.0, slack_webhook="...")]
)
```

### Multi-Hook Class

```python
class ObservabilityHook(BaseHook):
    hooks = ["on_request", "on_response", "on_error"]

    async def on_request(self, ctx: HookContext):
        ctx.metadata["trace_id"] = generate_trace_id()
        span = tracer.start_span("agent_request")
        ctx.metadata["span"] = span

    async def on_response(self, ctx: HookContext):
        span = ctx.metadata.get("span")
        if span:
            span.set_attribute("tokens", ctx.usage.total_tokens)
            span.end()

    async def on_error(self, ctx: HookContext):
        span = ctx.metadata.get("span")
        if span:
            span.record_exception(ctx.error)
            span.end()
```

## Fail-Open vs Fail-Closed

Configure hook failure behavior:

```python
from fastagentic.hooks import LakeraHook, FailureMode

# Fail-closed: Block execution if hook fails
LakeraHook(api_key="...", on_failure=FailureMode.REJECT)

# Fail-open: Log and continue if hook fails
LakeraHook(api_key="...", on_failure=FailureMode.WARN)

# Custom handler
LakeraHook(
    api_key="...",
    on_failure=lambda ctx, error: log_and_alert(error),
)
```

## Hook Ordering

Hooks execute in registration order:

```python
app = App(
    hooks=[
        AuthEnrichmentHook(),      # 1st
        LakeraHook(api_key="..."), # 2nd
        LangfuseHook(...),         # 3rd
    ],
)
```

For endpoint-specific hooks:

```python
@agent_endpoint(
    path="/chat",
    pre_hooks=[InputValidationHook(), RateLimitHook()],  # In order
    post_hooks=[OutputFilterHook()],
    eval_hooks=[BraintrustHook(...)],  # Async, order doesn't matter
)
```

## Built-in Hooks

FastAgentic includes simple built-in hooks:

| Hook | Purpose |
|------|---------|
| `OTELHook` | OpenTelemetry span export |
| `CostTrackingHook` | Basic token/cost counters |
| `AuditLogHook` | Structured audit logging |
| `RateLimitHook` | Simple rate limiting |

For advanced use cases, use first-class integrations.

## First-Class Integrations

See [Integrations Guide](integrations/index.md) for detailed setup:

| Integration | Hooks Provided | Purpose |
|-------------|----------------|---------|
| [Langfuse](integrations/langfuse.md) | `on_llm_*`, `on_tool_*` | LLM observability |
| [Logfire](integrations/logfire.md) | All hooks | PydanticAI tracing |
| [Lakera](integrations/lakera.md) | `on_llm_start` | Prompt injection detection |
| [Guardrails AI](integrations/guardrails.md) | `on_response` | Output validation |
| [Mem0](integrations/mem0.md) | Memory provider | Persistent memory |
| [Braintrust](integrations/braintrust.md) | Eval hooks | Experiment tracking |

## Performance Considerations

1. **Keep hooks fast** — Blocking hooks add latency to every request
2. **Use async mode** for non-critical logging/analytics
3. **Batch operations** in hooks when possible
4. **Set timeouts** on external API calls in hooks
5. **Monitor hook latency** via the `hook_duration_ms` metric

```python
# Bad: Slow blocking hook
@hook("on_response")
async def slow_hook(ctx: HookContext):
    await external_api.call()  # Blocks response

# Good: Non-blocking for non-critical work
@hook("on_response", mode=HookMode.ASYNC)
async def fast_hook(ctx: HookContext):
    await external_api.call()  # Doesn't block response
```

## Next Steps

- [Integrations Index](integrations/index.md) — Setup guides for each integration
- [Observability Guide](operations/observability/index.md) — Monitoring and tracing
- [Security Guide](operations/security/index.md) — Guardrails and compliance
