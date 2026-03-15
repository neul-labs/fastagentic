# MCP Implementation

FastAgentic implements the Model Context Protocol (MCP) specification [2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25), providing tool, resource, and prompt exposure to LLM hosts like Claude Desktop, IDE extensions, and custom applications.

## What MCP Provides

MCP enables LLM applications to access external capabilities through a standardized interface:

| Capability | Description | FastAgentic Mapping |
|------------|-------------|---------------------|
| **Tools** | Functions executable by AI models | `@tool` decorator, agent endpoints |
| **Resources** | Data accessible to models | `@resource` decorator |
| **Prompts** | Templated messages and workflows | `@prompt` decorator |
| **Sampling** | Server-initiated LLM interactions | Agent adapter loops |
| **Tasks** | Long-running operation tracking | Durable runs with checkpoints |

## Configuration

```python
from fastagentic import App
from fastagentic.protocols.mcp import configure_mcp

app = App(title="My Agent", version="1.0.0")

configure_mcp(
    app,
    # Core settings
    enabled=True,
    path_prefix="/mcp",

    # Transport options
    stdio_enabled=True,           # Enable stdio transport
    http_enabled=True,            # Enable HTTP transport

    # November 2025 features
    tasks_enabled=True,           # Enable MCP Tasks
    extensions_enabled=True,      # Enable Extensions framework
    parallel_tools=True,          # Enable parallel tool execution

    # Authorization (OAuth 2.0)
    authorization_enabled=True,
    authorization_server="https://auth.example.com",
    default_scopes=["read", "write"],

    # Registry integration
    registry_enabled=True,
    registry_url="https://registry.modelcontextprotocol.io",
)
```

## MCP Endpoints

FastAgentic exposes these MCP endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/schema` | GET | Full MCP manifest (tools, resources, prompts) |
| `/mcp/tools` | GET | List available tools |
| `/mcp/tools/{name}` | POST | Execute a tool |
| `/mcp/resources` | GET | List available resources |
| `/mcp/resources/{uri}` | GET | Fetch a resource |
| `/mcp/prompts` | GET | List available prompts |
| `/mcp/prompts/{name}` | GET | Get prompt template |
| `/mcp/tasks` | GET | List active tasks (2025-11) |
| `/mcp/tasks/{id}` | GET | Get task status (2025-11) |
| `/mcp/health` | GET | Health check |

## Tools

Tools are functions that LLMs can invoke:

```python
from fastagentic import tool
from pydantic import BaseModel

class SummarizeInput(BaseModel):
    text: str
    max_length: int = 100

class SummarizeOutput(BaseModel):
    summary: str
    word_count: int

@tool(
    name="summarize_text",
    description="Summarize long text into key points",
    # MCP-specific options
    mcp_annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def summarize(input: SummarizeInput) -> SummarizeOutput:
    # Implementation
    return SummarizeOutput(summary="...", word_count=50)
```

### Tool Naming Convention (SEP-986)

FastAgentic follows the MCP standardized naming convention:

```
{namespace}_{action}_{target}
```

Examples:
- `github_search_issues`
- `slack_send_message`
- `database_query_users`

## Resources

Resources provide data that LLMs can read:

```python
from fastagentic import resource

@resource(
    name="user-profile",
    uri="users/{user_id}/profile",
    description="User profile information",
    mime_type="application/json",
    # Caching
    cache_ttl=60,
)
async def get_user_profile(user_id: str) -> dict:
    return {"id": user_id, "name": "..."}
```

### Resource URI Templates

Resources support URI templates (RFC 6570):

```python
@resource(uri="repos/{owner}/{repo}/issues")
async def get_issues(owner: str, repo: str) -> list:
    ...

@resource(uri="files/{path*}")  # Wildcard path
async def get_file(path: str) -> bytes:
    ...
```

## Prompts

Prompts provide templated messages:

```python
from fastagentic import prompt

@prompt(
    name="code_review",
    description="Review code for bugs and improvements",
    arguments=[
        {"name": "language", "description": "Programming language", "required": True},
        {"name": "code", "description": "Code to review", "required": True},
    ],
)
def code_review_prompt(language: str, code: str) -> str:
    return f"""Review this {language} code for:
1. Bugs and potential errors
2. Performance improvements
3. Security issues
4. Code style

```{language}
{code}
```
"""
```

## MCP Tasks (November 2025)

Tasks track long-running operations with states:

| State | Description |
|-------|-------------|
| `working` | Task is in progress |
| `input_required` | Task needs additional input |
| `completed` | Task finished successfully |
| `failed` | Task failed with error |
| `cancelled` | Task was cancelled |

```python
from fastagentic import agent_endpoint
from fastagentic.protocols.mcp import MCPTask

@agent_endpoint(
    path="/analyze",
    mcp_task=True,  # Enable MCP Task tracking
    task_ttl=3600,  # Results available for 1 hour
)
async def analyze_data(data: DataInput) -> AnalysisResult:
    # Long-running analysis
    ...
```

### Task Lifecycle

```python
# Client initiates task
POST /mcp/tools/analyze_data
{
    "input": {"data": "..."},
    "task": true  # Request task-based execution
}

# Response includes task ID
{
    "task_id": "task_abc123",
    "status": "working"
}

# Client polls for status
GET /mcp/tasks/task_abc123
{
    "task_id": "task_abc123",
    "status": "completed",
    "result": {"analysis": "..."}
}
```

### Integration with Durable Runs

MCP Tasks map directly to FastAgentic's durable run system:

| MCP Task State | FastAgentic Run State |
|----------------|----------------------|
| `working` | `running` |
| `input_required` | `awaiting_input` |
| `completed` | `completed` |
| `failed` | `failed` |
| `cancelled` | `cancelled` |

## Sampling with Tools (November 2025)

Enable server-side agentic loops:

```python
from fastagentic.protocols.mcp import configure_sampling

configure_sampling(
    app,
    enabled=True,
    max_iterations=10,
    parallel_tools=True,      # Execute tools in parallel
    tool_call_timeout=30,     # Per-tool timeout
    total_timeout=300,        # Total loop timeout
)
```

This allows MCP servers to:
- Execute multi-step reasoning
- Call tools during sampling
- Run parallel tool executions
- Maintain conversation context

## Extensions (November 2025)

Extensions provide modular additions to core MCP:

```python
from fastagentic.protocols.mcp import register_extension

# Register a custom extension
register_extension(
    app,
    name="custom-analytics",
    version="1.0.0",
    capabilities=["token_tracking", "cost_attribution"],
    handler=analytics_handler,
)
```

### Built-in Extensions

FastAgentic provides these MCP extensions:

| Extension | Purpose |
|-----------|---------|
| `fastagentic.costs` | Token usage and cost tracking |
| `fastagentic.checkpoints` | Checkpoint metadata exposure |
| `fastagentic.audit` | Audit log access |
| `fastagentic.a2a` | A2A protocol bridge |

## Authorization (OAuth 2.0)

MCP authorization using OAuth 2.0:

```python
from fastagentic.protocols.mcp import configure_mcp_auth

configure_mcp_auth(
    app,
    # OAuth 2.0 Resource Server
    authorization_server="https://auth.example.com",
    audience="my-mcp-server",

    # Scopes
    default_scopes=["mcp:read"],
    tool_scopes={
        "summarize_text": ["mcp:read"],
        "create_ticket": ["mcp:write"],
    },

    # Client registration (SEP-991)
    client_id_metadata_url="https://auth.example.com/.well-known/oauth-client",

    # Machine-to-machine (SEP-1046)
    client_credentials_enabled=True,
)
```

## Transports

### HTTP Transport

Default for web clients:

```python
# Served at /mcp/* endpoints
configure_mcp(app, http_enabled=True)
```

### Stdio Transport

For local agent hosting (Claude Desktop, etc.):

```bash
# Run with stdio transport
fastagentic run --stdio

# Or programmatically
python -m fastagentic.mcp.stdio
```

### Streamable HTTP (SSE)

For long-running operations:

```python
@tool(stream=True)
async def generate_report(params: ReportParams):
    async for chunk in generate_chunks(params):
        yield {"type": "progress", "data": chunk}
    yield {"type": "complete", "data": final_report}
```

## Schema Generation

FastAgentic generates MCP schemas from Pydantic models:

```python
class TicketInput(BaseModel):
    title: str = Field(description="Ticket title")
    priority: Literal["low", "medium", "high"]
    labels: list[str] = []

# Automatically generates MCP tool schema:
# {
#   "name": "create_ticket",
#   "description": "Create a support ticket",
#   "inputSchema": {
#     "type": "object",
#     "properties": {
#       "title": {"type": "string", "description": "Ticket title"},
#       "priority": {"type": "string", "enum": ["low", "medium", "high"]},
#       "labels": {"type": "array", "items": {"type": "string"}}
#     },
#     "required": ["title", "priority"]
#   }
# }
```

## Testing MCP

```bash
# Validate MCP schema
fastagentic mcp validate

# Test tool execution
fastagentic mcp call summarize_text --input '{"text": "..."}'

# Interactive stdio session
fastagentic mcp stdio --interactive

# Generate MCP manifest
fastagentic mcp manifest > mcp.json
```

## Claude Desktop Integration

Configure Claude Desktop to use your FastAgentic server:

```json
{
  "mcpServers": {
    "my-agent": {
      "command": "fastagentic",
      "args": ["run", "--stdio"],
      "cwd": "/path/to/project"
    }
  }
}
```

Or via HTTP:

```json
{
  "mcpServers": {
    "my-agent": {
      "url": "http://localhost:8000/mcp",
      "transport": "http"
    }
  }
}
```

## Next Steps

- [A2A Integration](a2a.md) - Agent-to-agent protocol
- [Adapters Guide](../adapters/index.md) - Framework adapters
- [Platform Services](../platform-services.md) - Auth, observability, durability
