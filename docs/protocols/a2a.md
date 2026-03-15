# A2A Integration

FastAgentic implements the Agent-to-Agent (A2A) Protocol [v0.3](https://a2a-protocol.org/latest/specification/), enabling agents to discover, communicate, and delegate tasks to each other—whether internal to your deployment or external services.

## What A2A Provides

A2A enables multi-agent collaboration:

| Capability | Description | FastAgentic Implementation |
|------------|-------------|---------------------------|
| **Agent Cards** | Capability discovery | Auto-generated from adapter metadata |
| **Task Management** | Track delegated work | Unified with durable runs |
| **Streaming** | Real-time updates | SSE and gRPC support |
| **Push Notifications** | Async webhooks | Webhook registration |
| **Security** | Auth schemes | OAuth2/OIDC aligned |

## Agent Registry

FastAgentic provides an internal Agent Registry for agent discovery and delegation:

```python
from fastagentic import App
from fastagentic.protocols.a2a import configure_a2a, AgentRegistry

app = App(title="Multi-Agent Platform", version="1.0.0")

# Configure A2A with internal registry
registry = AgentRegistry()
configure_a2a(app, registry=registry)

# Agents are auto-registered when deployed
@app.agent_endpoint(
    path="/triage",
    runnable=PydanticAIAdapter(triage_agent),
    a2a_skill="support-triage",
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...

@app.agent_endpoint(
    path="/research",
    runnable=LangGraphAdapter(research_graph),
    a2a_skill="deep-research",
)
async def research(query: ResearchQuery) -> ResearchResult:
    ...
```

### Registry Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Registry                        │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │support-triage│  │deep-research│  │code-review │     │
│  │  (internal)  │  │  (internal) │  │ (external) │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌─────────────────────┐                    │
│              │  A2A Task Router    │                    │
│              └─────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Discovering Agents

```python
from fastagentic.protocols.a2a import AgentRegistry

registry = app.state.a2a_registry

# List all registered agents
agents = await registry.list_agents()
# [
#   {"skill": "support-triage", "internal": True, ...},
#   {"skill": "deep-research", "internal": True, ...},
#   {"skill": "code-review", "internal": False, "url": "https://..."},
# ]

# Find agent by skill
agent = await registry.find_agent("support-triage")

# Find agents by capability
researchers = await registry.find_agents_by_capability("research")
```

### Registering External Agents

```python
from fastagentic.protocols.a2a import ExternalAgent

# Register external A2A agent
registry.register_external(
    ExternalAgent(
        name="code-review",
        url="https://code-review.example.com",
        agent_card_url="https://code-review.example.com/.well-known/agent.json",
        # Authentication for this agent
        auth_type="oauth2",
        client_id="...",
        client_secret="...",
    )
)
```

## Agent Cards

Agent Cards advertise capabilities in JSON format. FastAgentic auto-generates these from your endpoint definitions:

### Generated Agent Card

```json
{
  "name": "Support Triage",
  "description": "AI-powered support ticket triage and routing",
  "url": "https://triage.example.com",
  "protocolVersion": "0.3",
  "provider": {
    "organization": "Example Corp",
    "url": "https://example.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "extendedCard": true
  },
  "skills": [
    {
      "id": "support-triage",
      "name": "Triage Support Ticket",
      "description": "Analyze and prioritize support tickets",
      "inputSchema": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "description": {"type": "string"},
          "customer_tier": {"type": "string", "enum": ["free", "pro", "enterprise"]}
        },
        "required": ["title", "description"]
      },
      "outputSchema": {
        "type": "object",
        "properties": {
          "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
          "category": {"type": "string"},
          "suggested_assignee": {"type": "string"}
        }
      }
    }
  ],
  "security": [
    {
      "type": "oauth2",
      "flows": {
        "clientCredentials": {
          "tokenUrl": "https://auth.example.com/oauth/token",
          "scopes": {
            "triage:read": "Read ticket data",
            "triage:write": "Create triage decisions"
          }
        }
      }
    }
  ],
  "interfaces": [
    {
      "type": "jsonrpc",
      "url": "https://triage.example.com/a2a/rpc"
    },
    {
      "type": "grpc",
      "url": "grpcs://triage.example.com:443"
    }
  ]
}
```

### Customizing Agent Card

```python
from fastagentic.protocols.a2a import AgentCardConfig

configure_a2a(
    app,
    agent_card=AgentCardConfig(
        name="Support Triage",
        description="AI-powered support ticket triage",
        provider={
            "organization": "Example Corp",
            "url": "https://example.com",
        },
        # Extended card requires authentication
        extended_skills=["advanced-analytics"],
    ),
)
```

## A2A Endpoints

FastAgentic exposes these A2A endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Public Agent Card |
| `/a2a/agent-card` | GET | Extended Agent Card (authenticated) |
| `/a2a/message/send` | POST | Send message to agent |
| `/a2a/message/stream` | POST | Send message with streaming |
| `/a2a/tasks` | GET | List tasks |
| `/a2a/tasks/{id}` | GET | Get task status |
| `/a2a/tasks/{id}/cancel` | POST | Cancel task |
| `/a2a/tasks/{id}/subscribe` | GET | Subscribe to task updates (SSE) |
| `/a2a/push/config` | POST | Configure push notifications |

## Task Lifecycle

A2A tasks progress through defined states:

```
SUBMITTED → WORKING → COMPLETED
                   ↘ FAILED
                   ↘ CANCELLED
                   ↘ REJECTED
                   → INPUT_REQUIRED → (resume) → WORKING
```

### Delegating Tasks

```python
from fastagentic.protocols.a2a import A2AClient

# Within an agent, delegate to another agent
async def triage_handler(ticket: TicketIn, ctx: AgentContext) -> TicketOut:
    # Check if research is needed
    if ticket.requires_research:
        # Delegate to research agent
        research_result = await ctx.delegate(
            skill="deep-research",
            message={
                "role": "user",
                "parts": [{"text": f"Research: {ticket.description}"}]
            },
            # Wait for completion
            await_result=True,
            timeout=60,
        )
        ticket.research_context = research_result

    # Continue with triage
    return await triage_logic(ticket)
```

### Task Streaming

```python
# Stream task updates
async for event in ctx.delegate_stream(
    skill="deep-research",
    message={"role": "user", "parts": [{"text": query}]},
):
    match event["type"]:
        case "status":
            print(f"Status: {event['status']}")
        case "artifact":
            print(f"Artifact: {event['artifact']}")
        case "message":
            print(f"Message: {event['message']}")
```

## Message Parts

A2A messages support multiple part types:

```python
from fastagentic.protocols.a2a import TextPart, FilePart, DataPart

# Send multi-part message
await ctx.delegate(
    skill="document-analysis",
    message={
        "role": "user",
        "parts": [
            TextPart(text="Analyze this document"),
            FilePart(
                uri="s3://bucket/doc.pdf",
                mime_type="application/pdf",
            ),
            DataPart(
                data={"metadata": {"source": "upload"}},
                schema_uri="https://schema.example.com/metadata",
            ),
        ]
    }
)
```

## Security

### Authentication Schemes

FastAgentic supports all A2A security schemes:

```python
from fastagentic.protocols.a2a import configure_a2a_security

configure_a2a_security(
    app,
    schemes=[
        # OAuth 2.0 (recommended)
        {
            "type": "oauth2",
            "flows": {
                "clientCredentials": {
                    "tokenUrl": "https://auth.example.com/token",
                    "scopes": {"agent:invoke": "Invoke agent skills"},
                }
            }
        },
        # API Key (simple)
        {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        },
        # Mutual TLS (high security)
        {
            "type": "mutualTLS",
        },
    ],
)
```

### Signed Agent Cards (v0.3)

Sign your Agent Card for verification:

```python
from fastagentic.protocols.a2a import sign_agent_card

configure_a2a(
    app,
    signing_key="/path/to/private-key.pem",
    signing_algorithm="RS256",
)
```

## gRPC Support (v0.3)

Enable gRPC transport for high-performance communication:

```python
from fastagentic.protocols.a2a import configure_a2a_grpc

configure_a2a_grpc(
    app,
    enabled=True,
    port=50051,
    reflection=True,  # Enable gRPC reflection
    max_message_size=4 * 1024 * 1024,  # 4MB
)
```

## Push Notifications

Configure webhook-based notifications:

```python
from fastagentic.protocols.a2a import PushConfig

# As a client, register for push notifications
await a2a_client.configure_push(
    PushConfig(
        url="https://my-service.example.com/webhooks/a2a",
        events=["task.completed", "task.failed", "task.input_required"],
        authentication={
            "type": "bearer",
            "token": "webhook-secret",
        },
    )
)
```

## Integration with MCP

A2A and MCP work together:

| Use Case | Protocol |
|----------|----------|
| LLM needs to call a tool | MCP |
| Agent needs to delegate to another agent | A2A |
| External system needs agent capability | Both (MCP for tools, A2A for tasks) |

### Bridge Example

```python
# Expose A2A skill as MCP tool
@app.tool(
    name="delegate_research",
    mcp_enabled=True,
)
async def delegate_research(query: str, ctx: ToolContext) -> str:
    """Delegate research to specialized agent via A2A."""
    result = await ctx.a2a_delegate(
        skill="deep-research",
        message={"role": "user", "parts": [{"text": query}]},
    )
    return result.artifacts[0].parts[0].text
```

## Testing A2A

```bash
# Validate Agent Card
fastagentic a2a validate

# Test skill invocation
fastagentic a2a invoke support-triage --input '{"title": "...", "description": "..."}'

# List registered agents
fastagentic a2a list

# Check external agent connectivity
fastagentic a2a ping https://external-agent.example.com
```

## External Integrations

FastAgentic's A2A implementation interoperates with:

| Platform | Integration |
|----------|-------------|
| **Google Vertex AI** | Agent Builder A2A support |
| **Azure AI Foundry** | Copilot Studio A2A agents |
| **LangChain** | LangGraph Cloud A2A |
| **Salesforce** | Agentforce A2A |
| **Custom** | Any A2A v0.3 compliant agent |

## Next Steps

- [MCP Implementation](mcp.md) - Model Context Protocol
- [Templates](../templates/index.md) - Starter projects
- [Operations Guide](../operations/index.md) - Production deployment

## Sources

- [A2A Specification v0.3](https://a2a-protocol.org/latest/specification/)
- [A2A GitHub](https://github.com/google-a2a/A2A)
- [A2A Python SDK](https://github.com/a2aproject/a2a-python)
