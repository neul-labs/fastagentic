# Comparison with Alternatives

FastAgentic is a deployment layer, not an agent framework. This page compares deployment approaches for agent applications.

## Quick Comparison

| Capability | Raw FastAPI | PydanticAI + uvicorn | LangServe | FastAgentic |
|------------|-------------|----------------------|-----------|-------------|
| REST endpoints | Manual | Manual | Auto | Auto |
| MCP protocol | DIY | N/A | N/A | Auto |
| A2A protocol | DIY | N/A | N/A | Auto |
| OpenAPI + MCP + A2A schema fusion | N/A | N/A | N/A | Auto |
| SSE streaming | DIY | N/A | Partial | Auto |
| WebSocket | DIY | N/A | N/A | Auto |
| OAuth2/OIDC auth | DIY | DIY | DIY | Built-in |
| Rate limiting | DIY | DIY | DIY | Built-in |
| Cost tracking | DIY | Via Logfire | N/A | Built-in |
| Durable checkpoints | DIY | Via Temporal | N/A | Built-in |
| Resume from failure | DIY | Via Temporal | N/A | Built-in |
| OpenTelemetry | DIY | Via Logfire | DIY | Built-in |
| Agent-to-agent delegation | DIY | N/A | N/A | Built-in |
| Multi-framework support | N/A | PydanticAI only | LangChain only | All adapters |
| CLI tooling | N/A | N/A | Limited | Full |
| Contract testing | N/A | N/A | N/A | Built-in |

## Detailed Comparisons

### FastAgentic vs. Raw FastAPI

**When to use Raw FastAPI:**
- Simple REST API with no agent functionality
- Full control over every implementation detail
- Team already has production patterns established

**When to use FastAgentic:**
- Agent needs REST + MCP + A2A + streaming protocols
- Production governance required (auth, policy, audit)
- Multiple agent frameworks in the same application
- Agent-to-agent collaboration required

**Code comparison:**

Raw FastAPI (simplified, ~80 lines for basic functionality):
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
from sse_starlette import EventSourceResponse
import jwt
import redis
import json

app = FastAPI()
security = HTTPBearer()
redis_client = redis.Redis()

async def verify_token(credentials = Depends(security)):
    # 20+ lines of JWT validation
    pass

@app.post("/agent")
async def run_agent(input: dict, user = Depends(verify_token)):
    # Manual streaming setup
    # Manual checkpoint logic
    # Manual cost tracking
    # Manual error handling
    pass

@app.get("/agent/{run_id}")
async def get_run(run_id: str):
    # Manual run retrieval
    pass

# Plus: MCP server (100+ lines)
# Plus: WebSocket handler (50+ lines)
# Plus: Telemetry setup (30+ lines)
# Plus: Policy enforcement (50+ lines)
```

FastAgentic (10 lines):
```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter

app = App(title="My Agent", oidc_issuer="...", durable_store="redis://...")

@agent_endpoint(
    path="/agent",
    runnable=PydanticAIAdapter(my_agent),
    stream=True,
    durable=True,
    scopes=["agent:run"],
)
async def run_agent(input: AgentInput) -> AgentOutput:
    pass
```

### FastAgentic vs. PydanticAI Standalone

**PydanticAI excels at:**
- Type-safe agent definition
- Structured outputs with validation
- Model-agnostic LLM calls
- Dependency injection for tools

**FastAgentic adds:**
- Multi-protocol hosting (REST, MCP, streaming)
- Authentication and authorization
- Durable checkpoints (beyond retries)
- Policy enforcement
- Cost tracking with guardrails
- CLI and contract testing

**Relationship:** Complementary. Use PydanticAI to build agents, FastAgentic to deploy them.

```python
# Build with PydanticAI
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4',
    deps_type=MyDeps,
    result_type=MyResult,
)

@agent.tool
async def search(ctx, query: str) -> str:
    return await ctx.deps.search_service.search(query)

# Deploy with FastAgentic
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter

app = App(title="Search Agent", oidc_issuer="...", durable_store="redis://...")

@agent_endpoint(path="/search", runnable=PydanticAIAdapter(agent), stream=True)
async def search(query: str) -> MyResult:
    pass
```

### FastAgentic vs. LangServe

**LangServe provides:**
- REST endpoints for LangChain runnables
- Basic streaming support
- Playground UI

**FastAgentic provides:**
- Support for any framework (not just LangChain)
- MCP protocol (LangServe doesn't support MCP)
- A2A protocol for agent collaboration
- Built-in auth and policy
- Durable checkpoints
- Cost tracking
- CLI tooling

**When to choose:**
- **LangServe:** LangChain-only project, simple deployment needs
- **FastAgentic:** Multi-framework, MCP needed, production governance required

## Feature Deep Dive

### Multi-Protocol Schema Fusion

FastAgentic's unique capability: define once, expose via REST, MCP, AND A2A with identical schemas.

```python
@tool(name="summarize", description="Summarize text into key points", scopes=["summaries:run"])
async def summarize(text: str) -> str:
    """Summarize the provided text."""
    return await llm.summarize(text)
```

**Automatically generates:**

1. **OpenAPI 3.1 operation:**
```yaml
/tools/summarize:
  post:
    summary: Summarize text into key points
    security:
      - oauth2: [summaries:run]
    requestBody:
      content:
        application/json:
          schema:
            properties:
              text: {type: string}
```

2. **MCP tool definition:**
```json
{
  "name": "summarize",
  "description": "Summarize text into key points",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {"type": "string"}
    }
  }
}
```

3. **A2A skill capability** (when using `@agent_endpoint` with `a2a_skill`):
```json
{
  "name": "summarize-agent",
  "skills": [{
    "id": "summarize",
    "name": "Text Summarization",
    "description": "Summarize text into key points"
  }]
}
```

No other framework provides this automatic schema parity across all three protocols.

### Durable Execution Comparison

| Approach | Failure Handling | Checkpoint Storage | Resume API |
|----------|------------------|-------------------|------------|
| PydanticAI | Retry on validation errors | None | None |
| PydanticAI + Temporal | Full durability | Temporal cluster | Temporal workflows |
| LangGraph | Checkpoint support | Memory by default | Manual |
| FastAgentic | Full durability | Redis/Postgres/S3 | `POST /path/{run_id}/resume` |

FastAgentic provides built-in durability without requiring external orchestrators.

### Cost Tracking Comparison

| Approach | Token Counting | Cost Calculation | Budget Enforcement | Reporting |
|----------|----------------|------------------|-------------------|-----------|
| PydanticAI | Via Logfire | Via Logfire | None | Logfire dashboard |
| LangChain | Callbacks | Manual | Manual | Custom |
| FastAgentic | Built-in | Built-in | Built-in guardrails | Per-run, user, tenant |

## Decision Guide

### Use FastAgentic when:

1. **Multiple protocols needed**
   - REST API for web clients
   - MCP for AI assistants and IDEs
   - A2A for agent-to-agent collaboration
   - Streaming for real-time UX

2. **Production governance required**
   - OAuth2/OIDC authentication
   - Rate limiting and quotas
   - Audit logging for compliance
   - Cost tracking and guardrails

3. **Framework flexibility important**
   - Different agents use different frameworks
   - Want to migrate between frameworks
   - Need unified deployment pattern

4. **Operational maturity needed**
   - Durable checkpoints for long-running workflows
   - OpenTelemetry observability
   - Contract testing for schema stability

### Don't use FastAgentic when:

1. **Pure experimentation**
   - Prototyping agent logic locally
   - No deployment requirements yet

2. **Single internal consumer**
   - One service calling agent directly
   - No authentication needed
   - No compliance requirements

3. **Already have deployment infrastructure**
   - Existing production patterns work well
   - Team prefers full control

## Migration Paths

### From Raw FastAPI
1. Install FastAgentic
2. Wrap existing agent logic in adapter
3. Replace manual endpoints with decorators
4. Configure auth and policy
5. Remove redundant boilerplate

### From PydanticAI + Custom Deployment
1. Keep PydanticAI agent code unchanged
2. Wrap with `PydanticAIAdapter`
3. Use `@agent_endpoint` decorator
4. Remove custom hosting code

### From LangServe
1. Keep LangChain runnables unchanged
2. Wrap with `LangChainAdapter`
3. Add MCP, auth, durability via FastAgentic
4. Migrate endpoints incrementally

## Next Steps

- [Adapters Overview](adapters/index.md) - Choose the right adapter
- [Getting Started](getting-started.md) - Quick start guide
- [Operations Guide](operations/index.md) - Production deployment
