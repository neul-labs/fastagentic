# Adapter Comparison

Side-by-side comparison of all FastAgentic adapters to help you choose the right one.

## Quick Decision Matrix

| If you need... | Use |
|----------------|-----|
| Type-safe agents with structured outputs | PydanticAI |
| Stateful workflows with cycles | LangGraph |
| Multi-agent collaboration | CrewAI |
| Existing LangChain chains | LangChain |
| Your own framework | Custom |

## Feature Comparison

| Feature | PydanticAI | LangGraph | CrewAI | LangChain |
|---------|------------|-----------|--------|-----------|
| **Streaming** |
| Token-level | Yes | Yes | Yes | Yes |
| Node/step-level | N/A | Yes | Per-agent | Per-chain |
| Tool call events | Yes | Yes | Yes | Yes |
| **Checkpointing** |
| Granularity | Per-run | Per-node | Per-task | Per-step |
| State size | Small | Medium | Medium | Small |
| Resume capability | Full | Full | Full | Partial |
| **Structure** |
| Workflow type | Linear | Graph | Hierarchical | Pipeline |
| Cycles supported | No | Yes | Via delegation | No |
| Human-in-the-loop | Via tools | Native | Callbacks | Via tools |
| **Observability** |
| Cost per component | Per-run | Per-node | Per-agent | Per-chain |
| Execution trace | Linear | Graph | Tree | Pipeline |
| **Type Safety** |
| Input validation | Pydantic | TypedDict | Pydantic | Varies |
| Output validation | Pydantic | TypedDict | String | Varies |
| IDE support | Excellent | Good | Good | Good |

## Streaming Events

### PydanticAI
```
token → token → tool_call → tool_result → token → run_complete
```

### LangGraph
```
node_start → token → checkpoint → node_end → edge → node_start → ...
```

### CrewAI
```
agent_start → token → tool_call → tool_result → agent_end → task_complete → agent_start → ...
```

### LangChain
```
chain_start → token → tool_call → tool_result → chain_end
```

## Checkpoint State Size

| Adapter | Typical Size | Contents |
|---------|--------------|----------|
| PydanticAI | 1-10 KB | Conversation, tool results |
| LangGraph | 10-100 KB | Full graph state per node |
| CrewAI | 10-50 KB | Task outputs, agent contexts |
| LangChain | 1-10 KB | Chain intermediate results |

## Use Case Recommendations

### Chatbots and Assistants
**Recommended: PydanticAI**
- Type-safe responses
- Clean tool integration
- Lightweight checkpoints

### Complex Workflows
**Recommended: LangGraph**
- Conditional branching
- Cycles and loops
- Per-node visibility

### Research and Analysis
**Recommended: CrewAI**
- Role-based agents
- Task delegation
- Parallel execution

### RAG and Retrieval
**Recommended: LangChain**
- Retriever integration
- LCEL pipelines
- Existing ecosystem

### Migration Projects
**Recommended: Matching adapter**
- Keep existing framework
- Add governance layer
- Migrate incrementally

## Performance Characteristics

| Adapter | Cold Start | Streaming Latency | Memory |
|---------|------------|-------------------|--------|
| PydanticAI | Fast | Low | Low |
| LangGraph | Medium | Medium | Medium |
| CrewAI | Slow | High | High |
| LangChain | Fast | Low | Low |

## Combining Adapters

Different endpoints can use different adapters:

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from fastagentic.adapters.langgraph import LangGraphAdapter
from fastagentic.adapters.crewai import CrewAIAdapter

app = App(title="Multi-Framework Service", ...)

# Quick responses with PydanticAI
@agent_endpoint(path="/chat", runnable=PydanticAIAdapter(chat_agent))
async def chat(message: str) -> str:
    pass

# Complex workflows with LangGraph
@agent_endpoint(path="/workflow", runnable=LangGraphAdapter(workflow))
async def workflow(input: WorkflowInput) -> WorkflowOutput:
    pass

# Research tasks with CrewAI
@agent_endpoint(path="/research", runnable=CrewAIAdapter(research_crew))
async def research(topic: str) -> Report:
    pass
```

## Migration Paths

### From Raw Framework to FastAgentic

1. Keep framework code unchanged
2. Wrap with appropriate adapter
3. Add `@agent_endpoint` decorator
4. Configure App with auth, durability
5. Remove manual deployment code

### Between Adapters

If switching frameworks:
1. Both endpoints can coexist
2. Migrate traffic gradually
3. Checkpoints are adapter-specific (restart runs)

## Next Steps

- [PydanticAI Adapter](pydanticai.md) - Type-safe agents
- [LangGraph Adapter](langgraph.md) - Stateful workflows
- [CrewAI Adapter](crewai.md) - Multi-agent collaboration
- [LangChain Adapter](langchain.md) - Chain deployment
- [Custom Adapters](custom.md) - Build your own
