# LangGraph Adapter

The LangGraph adapter wraps [LangGraph](https://langchain-ai.github.io/langgraph/) state graphs for deployment through FastAgentic. Deploy complex stateful workflows with full durability and streaming.

## TL;DR

Wrap any compiled LangGraph `StateGraph` and get REST + MCP + node-level streaming + checkpoint durability.

## Before FastAgentic

```python
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer
from sse_starlette import EventSourceResponse
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

app = FastAPI()
security = HTTPBearer()

# Define graph
class AgentState(TypedDict):
    messages: list
    next_step: str

workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_edge("research", "write")
workflow.add_edge("write", END)
workflow.set_entry_point("research")

checkpointer = MemorySaver()  # Not durable!
graph = workflow.compile(checkpointer=checkpointer)

@app.post("/workflow")
async def run_workflow(input: dict, user = Depends(verify_token)):
    config = {"configurable": {"thread_id": str(uuid4())}}
    result = await graph.ainvoke(input, config)
    return result

@app.post("/workflow/stream")
async def stream_workflow(input: dict, user = Depends(verify_token)):
    async def generate():
        config = {"configurable": {"thread_id": str(uuid4())}}
        async for event in graph.astream_events(input, config, version="v2"):
            yield {"event": event["event"], "data": json.dumps(event["data"])}
    return EventSourceResponse(generate())

# Missing: durable checkpoints, resume, MCP, auth, cost tracking, etc.
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.langgraph import LangGraphAdapter
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    next_step: str

workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_edge("research", "write")
workflow.add_edge("write", END)
workflow.set_entry_point("research")

graph = workflow.compile()

app = App(
    title="Research Workflow",
    oidc_issuer="https://auth.company.com",
    durable_store="postgres://...",
)

@agent_endpoint(
    path="/workflow",
    runnable=LangGraphAdapter(graph),
    input_model=WorkflowInput,
    output_model=WorkflowOutput,
    stream=True,
    durable=True,
)
async def run_workflow(input: WorkflowInput) -> WorkflowOutput:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /workflow` | Run graph synchronously |
| `POST /workflow/stream` | Run with node-level streaming |
| `GET /workflow/{run_id}` | Get run status and result |
| `GET /workflow/{run_id}/events` | Replay event stream |
| `POST /workflow/{run_id}/resume` | Resume from checkpoint |
| `GET /workflow/{run_id}/state` | Get current graph state |

### LangGraph-Specific Features

**Node-Level Streaming**

Events emitted for each node:

```
node_start: {node: "research", state: {...}}
token: {content: "Researching..."}
tool_call: {tool: "search", args: {...}}
tool_result: {tool: "search", output: [...]}
node_end: {node: "research", result: {...}}
checkpoint: {checkpoint_id: "...", node: "research"}
node_start: {node: "write", state: {...}}
...
```

**State Checkpointing**

Every node transition creates a durable checkpoint:

```python
# Resume from last checkpoint after failure
curl -X POST /workflow/run-123/resume

# Or resume from specific checkpoint
curl -X POST /workflow/run-123/resume?checkpoint=chk-456
```

**Human-in-the-Loop**

Use LangGraph interrupts with FastAgentic:

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("human", human_approval_node)

# Interrupt before human node
workflow.set_interrupt_before(["human"])
```

When interrupt is hit:
1. Checkpoint is saved
2. `interrupt` event emitted
3. Run pauses, waiting for resume
4. Call `/workflow/{run_id}/resume` with approval

**Conditional Branching**

```python
def route(state):
    if state["needs_review"]:
        return "review"
    return "publish"

workflow.add_conditional_edges("analyze", route, {
    "review": "human_review",
    "publish": "auto_publish"
})
```

FastAgentic streams the decision:
```
node_end: {node: "analyze", result: {...}}
edge: {from: "analyze", to: "human_review", condition: "needs_review=True"}
node_start: {node: "human_review", ...}
```

## Configuration Options

### LangGraphAdapter Constructor

```python
LangGraphAdapter(
    graph: CompiledStateGraph,
    state_schema: type[BaseModel] | None = None,
    checkpoint_every_node: bool = True,
    include_state_in_events: bool = False,
    interrupt_handler: Callable | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `CompiledStateGraph` | required | Compiled LangGraph |
| `state_schema` | `type[BaseModel]` | `None` | Pydantic model for state validation |
| `checkpoint_every_node` | `bool` | `True` | Checkpoint after each node |
| `include_state_in_events` | `bool` | `False` | Include full state in events |
| `interrupt_handler` | `Callable` | `None` | Custom interrupt handling |

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `node_start` | Node begins | `{node, state, timestamp}` |
| `node_end` | Node completes | `{node, result, duration}` |
| `edge` | Transition occurs | `{from, to, condition}` |
| `token` | LLM output | `{content, node}` |
| `tool_call` | Tool invoked | `{tool, args, node}` |
| `tool_result` | Tool returns | `{tool, output, node}` |
| `checkpoint` | State saved | `{checkpoint_id, node}` |
| `interrupt` | Human input needed | `{node, state, reason}` |

## Checkpoint State

The adapter persists:
- Full graph state at each node
- Node execution history
- Tool call results
- Token and cost counters
- Interrupt status

## Migration Guide

### From LangGraph with MemorySaver

Replace:
```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

With:
```python
# No checkpointer needed - FastAgentic handles it
graph = workflow.compile()
adapter = LangGraphAdapter(graph)
```

### From LangGraph with PostgresSaver

Replace:
```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(conn_string)
graph = workflow.compile(checkpointer=checkpointer)
```

With:
```python
# FastAgentic manages the connection
graph = workflow.compile()
app = App(..., durable_store="postgres://...")

@agent_endpoint(path="/workflow", runnable=LangGraphAdapter(graph), durable=True)
```

## Common Patterns

### Multi-Step Research

```python
class ResearchState(TypedDict):
    query: str
    sources: list[str]
    findings: list[dict]
    report: str

workflow = StateGraph(ResearchState)
workflow.add_node("search", search_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "synthesize")
workflow.add_edge("synthesize", END)

@agent_endpoint(
    path="/research",
    runnable=LangGraphAdapter(workflow.compile()),
    stream=True,
    durable=True,
)
async def research(query: str) -> ResearchReport:
    pass
```

### Approval Workflows

```python
workflow.add_node("draft", draft_node)
workflow.add_node("review", human_review_node)
workflow.add_node("publish", publish_node)

workflow.set_interrupt_before(["review"])

@agent_endpoint(
    path="/content",
    runnable=LangGraphAdapter(workflow.compile()),
    durable=True,
)
async def create_content(topic: str) -> Content:
    pass
```

Usage:
```bash
# Start workflow
curl -X POST /content -d '{"topic": "AI trends"}'
# Returns: {run_id: "run-123", status: "interrupted", node: "review"}

# Approve and continue
curl -X POST /content/run-123/resume -d '{"approved": true}'
```

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [PydanticAI Adapter](pydanticai.md) - For type-safe agents
- [Custom Adapters](custom.md) - Build your own
