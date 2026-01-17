# Adapter Comparison

Side-by-side comparison of all FastAgentic adapters to help you choose the right one.

## Quick Decision Matrix

| If you need... | Use |
|----------------|-----|
| Type-safe agents with structured outputs | PydanticAI |
| Stateful workflows with cycles | LangGraph |
| Multi-agent collaboration | CrewAI or AutoGen |
| Existing LangChain chains | LangChain |
| RAG and retrieval applications | LlamaIndex |
| Prompt optimization and compilation | DSPy |
| Microsoft ecosystem integration | Semantic Kernel |
| Your own framework | Custom |

## Feature Comparison

### Core Features

| Feature | PydanticAI | LangGraph | CrewAI | LangChain | DSPy | LlamaIndex | SemanticKernel | AutoGen |
|---------|------------|-----------|--------|-----------|------|------------|----------------|---------|
| **Streaming** | Yes | Yes | Yes | Yes | Yes | Partial | Yes | Yes |
| **Checkpointing** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Tool Events** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Partial |
| **Resume Support** | Full | Full | Full | Full | Full | Full | Full | Full |

### Streaming Capabilities

| Adapter | Token Streaming | Node/Step Events | Tool Events | Native Streaming |
|---------|-----------------|------------------|-------------|------------------|
| PydanticAI | Yes | N/A | Yes | Yes |
| LangGraph | Yes | Yes | Yes | Yes |
| CrewAI | Yes | Per-agent | Yes | Event Bus |
| LangChain | Yes | Per-chain | Yes | Yes |
| DSPy | Yes | N/A | Yes | Via streamify |
| LlamaIndex | Yes | Query/Retrieve | Yes | Partial |
| SemanticKernel | Yes | Per-function | Yes | Yes |
| AutoGen | Yes | Per-turn | Partial | v0.4+ API |

### Checkpointing Details

| Adapter | Granularity | State Contents | Resume Capability |
|---------|-------------|----------------|-------------------|
| PydanticAI | Per-run, per-tool | Messages, tool calls | Full |
| LangGraph | Per-node | Full graph state | Full |
| CrewAI | Per-task | Task outputs, agent state | Full |
| LangChain | Per-chain-step | Accumulated state, tool calls | Full |
| DSPy | Per-module | Result, trace | Full |
| LlamaIndex | Per-query | Chat history, sources | Full |
| SemanticKernel | Per-function | Chat history, function calls | Full |
| AutoGen | Per-turn | Message buffer | Full |

### Type Safety and Validation

| Adapter | Input Validation | Output Validation | IDE Support |
|---------|------------------|-------------------|-------------|
| PydanticAI | Pydantic | Pydantic | Excellent |
| LangGraph | TypedDict | TypedDict | Good |
| CrewAI | Pydantic | String | Good |
| LangChain | Varies | Varies | Good |
| DSPy | Signature | Signature | Good |
| LlamaIndex | Pydantic | Pydantic | Good |
| SemanticKernel | Any | Any | Good |
| AutoGen | Dict | Dict | Good |

## Streaming Events

### PydanticAI
```
token → token → tool_call → tool_result → token → checkpoint → done
```

### LangGraph
```
node_start → token → checkpoint → node_end → edge → node_start → ... → done
```

### CrewAI
```
agent_start → token → tool_call → tool_result → agent_end → checkpoint → agent_start → ... → done
```

### LangChain
```
chain_start → token → tool_call → tool_result → checkpoint → chain_end → done
```

### DSPy
```
token → token → tool_call → tool_result → checkpoint → done
```

### LlamaIndex
```
node_start → token → source → tool_call → tool_result → checkpoint → done
```

### SemanticKernel
```
tool_call → token → token → tool_result → checkpoint → done
```

### AutoGen
```
message → token → checkpoint → message → checkpoint → ... → done
```

## Checkpoint State Size

| Adapter | Typical Size | Contents |
|---------|--------------|----------|
| PydanticAI | 1-10 KB | Conversation, tool results |
| LangGraph | 10-100 KB | Full graph state per node |
| CrewAI | 10-50 KB | Task outputs, agent contexts |
| LangChain | 1-10 KB | Chain intermediate results |
| DSPy | 1-10 KB | Module result, trace |
| LlamaIndex | 5-20 KB | Chat history, sources |
| SemanticKernel | 1-10 KB | Chat history, function calls |
| AutoGen | 5-50 KB | Message buffer |

## Use Case Recommendations

### Chatbots and Assistants
**Recommended: PydanticAI or SemanticKernel**
- Type-safe responses
- Clean tool integration
- Lightweight checkpoints

### Complex Workflows
**Recommended: LangGraph**
- Conditional branching
- Cycles and loops
- Per-node visibility

### Research and Analysis
**Recommended: CrewAI or AutoGen**
- Role-based agents
- Task delegation
- Parallel execution

### RAG and Retrieval
**Recommended: LlamaIndex or LangChain**
- Native retriever integration
- Query engines and chat engines
- Document processing

### Prompt Optimization
**Recommended: DSPy**
- Automatic prompt tuning
- Few-shot learning
- Module composition

### Enterprise Microsoft Stack
**Recommended: SemanticKernel**
- Azure integration
- Plugins and planners
- .NET compatibility

### Multi-Agent Conversations
**Recommended: AutoGen**
- Conversational agents
- Code execution
- Group chats

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
| CrewAI | Slow | Medium | High |
| LangChain | Fast | Low | Low |
| DSPy | Fast | Low | Low |
| LlamaIndex | Medium | Low | Medium |
| SemanticKernel | Fast | Low | Low |
| AutoGen | Medium | Medium | Medium |

## Combining Adapters

Different endpoints can use different adapters:

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from fastagentic.adapters.langgraph import LangGraphAdapter
from fastagentic.adapters.crewai import CrewAIAdapter
from fastagentic.adapters.llamaindex import LlamaIndexAdapter
from fastagentic.adapters.dspy import DSPyAdapter

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

# RAG with LlamaIndex
@agent_endpoint(path="/query", runnable=LlamaIndexAdapter(query_engine=index.as_query_engine()))
async def query(question: str) -> Answer:
    pass

# Optimized prompts with DSPy
@agent_endpoint(path="/classify", runnable=DSPyAdapter(classifier_module))
async def classify(text: str) -> Classification:
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

## Builder Methods

All adapters support builder pattern for configuration:

```python
# PydanticAI
adapter = PydanticAIAdapter(agent).with_deps(deps).with_checkpoints(on_tool=True)

# LangGraph
adapter = LangGraphAdapter(graph).with_checkpoints(["node1", "node2"]).with_stream_mode("updates")

# CrewAI
adapter = CrewAIAdapter(crew).with_checkpoints(enabled=True).with_verbose()

# LangChain
adapter = LangChainAdapter(chain).with_checkpoints(["on_chain_end"]).with_stream_mode("events")

# DSPy
adapter = DSPyAdapter(module).with_trace(enabled=True).with_stream_fields(["answer"])

# LlamaIndex
adapter = LlamaIndexAdapter(query_engine=qe).with_checkpoints(enabled=True)

# SemanticKernel
adapter = SemanticKernelAdapter(kernel).with_function("chat").with_checkpoints()

# AutoGen
adapter = AutoGenAdapter(initiator, recipient).with_max_turns(10).with_checkpoints()
```

## Next Steps

- [PydanticAI Adapter](pydanticai.md) - Type-safe agents
- [LangGraph Adapter](langgraph.md) - Stateful workflows
- [CrewAI Adapter](crewai.md) - Multi-agent collaboration
- [LangChain Adapter](langchain.md) - Chain deployment
- [DSPy Adapter](dspy.md) - Prompt optimization
- [LlamaIndex Adapter](llamaindex.md) - RAG applications
- [SemanticKernel Adapter](semantic_kernel.md) - Microsoft ecosystem
- [AutoGen Adapter](autogen.md) - Multi-agent conversations
- [Custom Adapters](custom.md) - Build your own
