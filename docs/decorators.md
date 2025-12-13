# Decorators and Schema Fusion

FastAgentic exposes decorators that make tools, resources, prompts, and agent workflows available through REST, streaming channels, and MCP without duplicating definitions. Each decorator reuses Pydantic models for validation and schema generation.

## Common Concepts

- **Pydantic models** act as the single source of truth for request and response shapes.
- **Scopes** define authorization requirements for both HTTP and MCP invocations.
- **Examples** provide documentation snippets for OpenAPI and MCP discovery.
- **Traits** such as streaming or durability propagate to all transports.

## `App`

```python
from fastagentic import App

app = App(
    title="Support Triage",
    version="1.0.0",
    oidc_issuer="https://auth.mycompany.com",
    telemetry=True,
    durable_store="redis://localhost:6379",
)
```

`App` extends FastAPI with:

- Dual boot modes (`asgi()`, `stdio()`)
- Automatic `/openapi.json`, `/mcp/schema`, `/mcp/discovery`, `/mcp/health`
- Lifespan orchestration for database connections, telemetry exporters, and durable run stores
- Shared dependencies such as `current_user()` for authentication and policy enforcement

## `@tool`

Registers a callable as both a REST endpoint and an MCP tool.

```python
from fastagentic import tool


@tool(
    name="summarize_tickets",
    description="Summarize support tickets into key points",
    scopes=["summaries:run"],
    examples=[{"text": "Customer cannot connect to VPN in the morning."}],
)
async def summarize(text: str) -> str:
    ...
```

| Parameter     | Purpose                                                              |
| ------------- | -------------------------------------------------------------------- |
| `name`        | Tool identifier used in MCP and REST path (`/tools/{name}`)         |
| `description` | Human-readable description for MCP and OpenAPI documentation        |
| `scopes`      | OAuth2 scopes required for invocation                                |
| `examples`    | Sample payloads for OpenAPI and MCP documentation                    |
| `stream`      | Enable SSE/WebSocket/MCP streaming for long-running tool responses   |

### Input Binding

- Function parameters are mapped from the request body for JSON verbs (`POST`, `PUT`, `PATCH`) or query params for `GET`.
- Use Pydantic models for structured inputs: `async def summarize(payload: SummaryRequest) -> SummaryResponse`.
- MCP tool calls reuse the same model to validate JSON arguments.

## `@resource`

Exposes persistent or queryable data sources and maps directly to MCP resources.

```python
from fastagentic import resource


@resource(
    name="run-status",
    uri="runs/{run_id}",
    description="Fetch the status of a durable run",
    scopes=["runs:read"],
    cache_ttl=60,
)
async def fetch_run(run_id: str) -> dict:
    ...
```

| Parameter     | Purpose                                                       |
| ------------- | ------------------------------------------------------------- |
| `name`        | Resource namespace used in MCP (`resources.run-status`)       |
| `uri`         | URI template with path parameters (no leading slash)          |
| `description` | Human-readable description for MCP discovery                  |
| `scopes`      | Access control definitions                                    |
| `cache_ttl`   | Optional cache hint (seconds) for clients and intermediate caches |

Resources typically return JSON-serializable dictionaries or Pydantic models. All path parameters are exposed as MCP resource identifiers, e.g., `resources.run-status:runs/{run_id}`.

## `@prompt`

Defines reusable prompt templates that appear in MCP discovery and documentation.

```python
from fastagentic import prompt


@prompt(
    name="triage_prompt",
    description="Guides support ticket triage conversations",
    arguments=[
        {"name": "ticket", "description": "Structured ticket metadata", "required": True}
    ],
)
def triage_prompt(ticket: dict) -> str:
    return """
    You are a support triage assistant.
    Ask clarifying questions when urgency or impact is unclear.
    Ticket: {{ ticket }}
    """
```

| Parameter     | Purpose                                                       |
| ------------- | ------------------------------------------------------------- |
| `name`        | Prompt identifier used in MCP                                 |
| `description` | Human-readable description for MCP discovery                  |
| `arguments`   | List of argument definitions with name, description, required |

Prompt functions must **return a string** or rich prompt object (for frameworks that support multi-part prompts). Function parameters document available template variables. FastAgentic inspects the signature to expose variable descriptions and to validate MCP prompt invocations.

## `@agent_endpoint`

Wraps agent frameworks or custom runnables and exposes them through REST, streaming channels, and MCP tools.

```python
from fastagentic import agent_endpoint
from fastagentic.adapters.langgraph import LangGraphAdapter
from models import TicketIn, TicketOut
from workflows import triage_graph


@agent_endpoint(
    path="/triage",
    runnable=LangGraphAdapter(triage_graph),
    input_model=TicketIn,
    output_model=TicketOut,
    stream=True,
    durable=True,
    scopes=["triage:run"],
    mcp_tool="triage_ticket",      # Expose as MCP tool
    a2a_skill="support-triage",    # Expose as A2A skill
)
async def triage(ticket: TicketIn, user=Depends(current_user)) -> TicketOut:
    ...
```

| Parameter      | Purpose                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| `path`         | REST path and base MCP tool identifier (`agents.triage.run`)            |
| `runnable`     | Adapter wrapping LangChain, LangGraph, CrewAI, or custom workflows      |
| `input_model`  | Pydantic model for request payload                                      |
| `output_model` | Pydantic model for final result                                         |
| `stream`       | Enables SSE/WebSocket/MCP event streaming during execution              |
| `durable`      | Persists checkpoints for resuming long-lived runs                       |
| `scopes`       | Authorization requirements                                              |
| `mcp_tool`     | Optional MCP tool name (defaults to path-derived name)                  |
| `a2a_skill`    | Optional A2A skill name for agent-to-agent discovery                    |

### Invocation Semantics

- HTTP requests whose payloads conform to `input_model` are provided as the first argument (named after the model type or parameter).
- MCP invocations use the same schema and produce identical event streams.
- Additional dependencies (e.g., `current_user`) can be injected via FastAPI dependency injection.
- When `stream=True`, clients receive event frames such as `token`, `node_start`, and `checkpoint`.

### Durable Runs

Enabling `durable=True` registers helper endpoints:

- `GET /runs/{run_id}` for run metadata
- `GET /runs/{run_id}/events` for streaming history replay
- `POST /runs/{run_id}/resume` to continue execution from the latest checkpoint

## Schema Fusion Pipeline

1. Decorators collect metadata during import time.
2. At application startup, the fusion engine builds OpenAPI paths, components, and security definitions.
3. The same metadata is transformed into MCP `tool`, `resource`, and `prompt` manifests.
4. `fastagentic test contract` validates parity by comparing generated documents.

## Error Handling

- Exceptions raised by handlers are mapped to structured error responses with trace IDs.
- Validation errors from Pydantic propagate to both HTTP and MCP callers with detailed field information.
- Custom error handlers can be registered on the `App` instance; they automatically apply to all transports.

