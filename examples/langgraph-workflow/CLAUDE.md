# LangGraph Workflow Agent - Claude Code Guide

This is a FastAgentic example using LangGraph for a stateful workflow agent.

## Project Structure

```
langgraph-workflow/
├── CLAUDE.md          # This file
├── app.py             # FastAgentic application
├── workflow.py        # LangGraph workflow definition
├── nodes.py           # Workflow node functions
├── state.py           # State schema
├── models.py          # API models
├── pyproject.toml     # Dependencies
├── .env.example       # Environment template
└── README.md          # Documentation
```

## Key Commands

```bash
# Install
uv sync

# Run server
uv run fastagentic run

# Test workflow
uv run fastagentic agent chat --endpoint /research

# View workflow graph
uv run python -c "from workflow import graph; graph.get_graph().print_ascii()"
```

## Architecture

This example implements a **research workflow** with multiple steps:

```
[Start] → [Plan] → [Research] → [Analyze] → [Summarize] → [End]
                       ↓
                  [More Research?] → [Research]
```

- **State** (`state.py`): Shared state across all nodes
- **Nodes** (`nodes.py`): Individual workflow steps
- **Workflow** (`workflow.py`): Graph definition with edges
- **App** (`app.py`): FastAgentic wrapper

## When Modifying

1. **Add a new node**: Create function in `nodes.py`, add to graph in `workflow.py`
2. **Change state**: Update `ResearchState` in `state.py`
3. **Add conditional routing**: Use `add_conditional_edges` in `workflow.py`
4. **Enable checkpointing**: Set `durable=True` in `@agent_endpoint`

## Common Patterns

### Add a conditional edge

```python
# In workflow.py
def should_continue(state: ResearchState) -> str:
    if len(state.findings) < 3:
        return "research"  # Go back to research
    return "analyze"  # Move to analysis

graph.add_conditional_edges(
    "research",
    should_continue,
    {"research": "research", "analyze": "analyze"}
)
```

### Add human-in-the-loop

```python
# In nodes.py
from langgraph.prebuilt import interrupt

async def review_node(state: ResearchState) -> ResearchState:
    # Pause for human review
    interrupt("Please review the findings before continuing")
    return state
```

### Enable persistence

```python
# In app.py
from langgraph.checkpoint.redis import RedisSaver

checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")
compiled = graph.compile(checkpointer=checkpointer)
```
