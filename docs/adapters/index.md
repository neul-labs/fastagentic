# Adapters: From Framework to Production in Minutes

Adapters are FastAgentic's bridge between agent frameworks and production deployment. Wrap your existing agent logic with an adapter, and FastAgentic handles everything else.

## The Transformation

**Before FastAgentic** - 50+ lines of deployment code per agent:
```python
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer
from sse_starlette import EventSourceResponse
from pydantic_ai import Agent
import redis
import json

app = FastAPI()
security = HTTPBearer()
redis_client = redis.Redis()

agent = Agent('openai:gpt-4', system_prompt="...")

async def verify_token(credentials = Depends(security)):
    # JWT validation logic...
    pass

@app.post("/chat")
async def chat(message: str, user = Depends(verify_token)):
    # Run agent
    result = await agent.run(message)
    # Save to Redis for durability
    redis_client.set(f"run:{run_id}", json.dumps(result))
    return result

@app.post("/chat/stream")
async def chat_stream(message: str, user = Depends(verify_token)):
    async def generate():
        async for chunk in agent.run_stream(message):
            yield {"event": "token", "data": json.dumps(chunk)}
    return EventSourceResponse(generate())

# Plus MCP server, WebSocket, telemetry, cost tracking...
```

**After FastAgentic** - 10 lines:
```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic_ai import Agent

agent = Agent('openai:gpt-4', system_prompt="...")

app = App(title="Chat", oidc_issuer="https://auth.company.com", durable_store="redis://localhost")

@agent_endpoint(path="/chat", runnable=PydanticAIAdapter(agent), stream=True, durable=True)
async def chat(message: str) -> str:
    pass
```

## What You Get For Free

| Feature | Without FastAgentic | With FastAgentic |
|---------|---------------------|------------------|
| **REST API** | Build from scratch | `@agent_endpoint` decorator |
| **MCP Protocol** | 100+ lines or N/A | Automatic schema fusion |
| **SSE Streaming** | Manual EventSource | `stream=True` |
| **WebSocket** | Manual handler | Built-in |
| **OAuth2/OIDC** | DIY middleware | `oidc_issuer` config |
| **Rate Limiting** | Build or buy | `rate_limit` parameter |
| **Durability** | Custom checkpointing | `durable=True` |
| **Resume** | Build from scratch | `POST /path/{run_id}/resume` |
| **Cost Tracking** | Manual instrumentation | Automatic per-run |
| **Observability** | OTEL boilerplate | `telemetry=True` |
| **Contract Tests** | N/A | `fastagentic test contract` |

## Supported Adapters

### PydanticAI Adapter

Best for: Type-safe agents, structured outputs, model-agnostic code

```python
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic_ai import Agent

agent = Agent('openai:gpt-4', deps_type=MyDeps, result_type=MyResult)
adapter = PydanticAIAdapter(agent)
```

[Full PydanticAI Adapter Guide](pydanticai.md)

### LangGraph Adapter

Best for: Stateful workflows, conditional logic, cycles, human-in-the-loop

```python
from fastagentic.adapters.langgraph import LangGraphAdapter
from langgraph.graph import StateGraph

graph = StateGraph(MyState)
# ... define nodes and edges ...
compiled = graph.compile()
adapter = LangGraphAdapter(compiled)
```

[Full LangGraph Adapter Guide](langgraph.md)

### CrewAI Adapter

Best for: Multi-agent collaboration, role-based tasks, delegation

```python
from fastagentic.adapters.crewai import CrewAIAdapter
from crewai import Crew, Agent, Task

crew = Crew(agents=[...], tasks=[...])
adapter = CrewAIAdapter(crew)
```

[Full CrewAI Adapter Guide](crewai.md)

### LangChain Adapter

Best for: Existing LangChain investments, LCEL chains, retrieval pipelines

```python
from fastagentic.adapters.langchain import LangChainAdapter
from langchain_core.runnables import RunnableSequence

chain = prompt | llm | parser
adapter = LangChainAdapter(chain)
```

[Full LangChain Adapter Guide](langchain.md)

### Custom Adapter

Best for: Proprietary frameworks, unique requirements

```python
from fastagentic.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    async def invoke(self, input, config):
        return await self.runnable.run(input)

    async def stream(self, input, config):
        async for chunk in self.runnable.stream(input):
            yield self.create_event("token", chunk)
```

[Custom Adapter Guide](custom.md)

## 30-Second Decision Guide

| If you have... | Use this adapter |
|----------------|------------------|
| PydanticAI `Agent` | `PydanticAIAdapter` |
| LangGraph `StateGraph` | `LangGraphAdapter` |
| CrewAI `Crew` | `CrewAIAdapter` |
| LangChain `Runnable` or chain | `LangChainAdapter` |
| Something else | `BaseAdapter` subclass |

## Using Multiple Adapters

Different endpoints can use different adapters in the same application:

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from fastagentic.adapters.langgraph import LangGraphAdapter
from fastagentic.adapters.crewai import CrewAIAdapter

app = App(title="Multi-Agent Service", ...)

# Simple chat with PydanticAI
@agent_endpoint(path="/chat", runnable=PydanticAIAdapter(chat_agent))
async def chat(message: str) -> str:
    pass

# Complex workflow with LangGraph
@agent_endpoint(path="/workflow", runnable=LangGraphAdapter(workflow_graph), durable=True)
async def workflow(input: WorkflowInput) -> WorkflowOutput:
    pass

# Research tasks with CrewAI
@agent_endpoint(path="/research", runnable=CrewAIAdapter(research_crew), stream=True)
async def research(topic: str) -> ResearchReport:
    pass
```

## Automatic Endpoints

Every `@agent_endpoint` creates:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/path` | POST | Run agent (sync response) |
| `/path/stream` | POST | Run agent (SSE streaming) |
| `/path/{run_id}` | GET | Get run status and result |
| `/path/{run_id}/events` | GET | Replay event stream |
| `/path/{run_id}/resume` | POST | Resume from checkpoint |

Plus MCP tool registration at `/mcp/schema`.

## Event Types

All adapters emit standardized events:

| Event | Description | Payload |
|-------|-------------|---------|
| `token` | LLM output token | `{content, delta}` |
| `node_start` | Workflow node begins | `{node, state}` |
| `node_end` | Workflow node completes | `{node, result}` |
| `tool_call` | Tool invocation starts | `{tool, args}` |
| `tool_result` | Tool returns result | `{tool, output}` |
| `checkpoint` | State persisted | `{checkpoint_id}` |
| `cost` | Usage metrics | `{tokens, amount}` |
| `run_complete` | Agent finished | `{result}` |

## Next Steps

- [PydanticAI Adapter](pydanticai.md) - Full guide with examples
- [LangGraph Adapter](langgraph.md) - Stateful workflow deployment
- [Adapter Comparison](comparison.md) - Side-by-side feature matrix
- [Custom Adapters](custom.md) - Build your own adapter
