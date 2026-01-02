# Choosing an Adapter

This guide helps you choose the right FastAgentic adapter for your agent framework. We'll walk through your options based on what you're building.

---

## Start Here: What Are You Building?

### "I'm starting a new project"

If you're starting fresh without an existing codebase:

| Your Priority | Recommended Adapter | Why |
|---------------|---------------------|-----|
| Type safety, clean code | **PydanticAI** | Native Pydantic validation, excellent IDE support |
| Complex workflows, state machines | **LangGraph** | Graph-based, cycles, conditional branching |
| Multiple specialized agents | **CrewAI** | Role-based agents, task delegation |
| Rapid prototyping | **LangChain** | Large ecosystem, many examples |

### "I have existing code"

If you're adding FastAgentic to an existing agent:

| You Currently Use | Recommended Adapter |
|-------------------|---------------------|
| PydanticAI | `PydanticAIAdapter` |
| LangGraph | `LangGraphAdapter` |
| CrewAI | `CrewAIAdapter` |
| LangChain | `LangChainAdapter` |
| Custom/raw API calls | `BaseAdapter` (custom) |

**Key insight**: FastAgentic adapters wrap your existing code. You don't rewrite your agent — you deploy it.

---

## Decision Tree

```
What best describes your agent?
│
├─► "A chatbot or assistant that answers questions"
│   │
│   ├─► Need structured outputs (JSON, Pydantic models)?
│   │   └─► PydanticAI — Type-safe, validated responses
│   │
│   └─► Simple text responses are fine?
│       └─► LangChain or PydanticAI — Both work well
│
├─► "A workflow with multiple steps and decisions"
│   │
│   ├─► Steps can loop back or retry?
│   │   └─► LangGraph — Graph supports cycles
│   │
│   ├─► Linear pipeline, no cycles?
│   │   └─► LangChain or PydanticAI — Simpler is better
│   │
│   └─► Different paths based on conditions?
│       └─► LangGraph — Conditional edges, branching
│
├─► "Multiple agents working together"
│   │
│   ├─► Agents have different roles (researcher, writer, reviewer)?
│   │   └─► CrewAI — Role-based, hierarchical crews
│   │
│   └─► Agents talk to each other dynamically?
│       └─► LangGraph — Model agent communication as graph
│
└─► "Something custom or unique"
    │
    └─► BaseAdapter — Full control, minimal overhead
```

---

## Adapter Deep Dives

### PydanticAI Adapter

**Best for**: Type-safe assistants, structured outputs, clean code

**Why choose PydanticAI:**

1. **Type safety**: Input and output validation with Pydantic
2. **IDE support**: Autocomplete, type hints, refactoring
3. **Logfire integration**: Native observability via Logfire
4. **Minimal overhead**: Lightweight, fast cold starts

**Example use cases:**
- Customer support chatbot with structured responses
- Data extraction from documents
- API that returns validated JSON

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from fastagentic import agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter

class TicketResponse(BaseModel):
    priority: str
    category: str
    suggested_action: str

agent = Agent("openai:gpt-4o", result_type=TicketResponse)

@agent_endpoint(
    path="/triage",
    runnable=PydanticAIAdapter(agent),
)
async def triage(description: str) -> TicketResponse:
    pass
```

**When NOT to use:**
- Complex multi-step workflows with loops
- Multiple agents collaborating
- You need LangGraph-specific features

---

### LangGraph Adapter

**Best for**: Complex workflows, state machines, conditional logic

**Why choose LangGraph:**

1. **Graph structure**: Model any workflow topology
2. **Cycles**: Agents can loop back, retry, iterate
3. **Per-node checkpoints**: Resume from any node
4. **Conditional routing**: Branch based on state

**Example use cases:**
- Multi-step research with refinement loops
- Approval workflows with conditional paths
- Agents that iterate until a condition is met

```python
from langgraph.graph import StateGraph
from fastagentic import agent_endpoint
from fastagentic.adapters.langgraph import LangGraphAdapter

workflow = StateGraph(ResearchState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("refine", refine_node)
workflow.add_conditional_edges("analyze", should_refine, {
    True: "refine",
    False: END,
})
graph = workflow.compile()

@agent_endpoint(
    path="/research",
    runnable=LangGraphAdapter(graph),
    durable=True,  # Per-node checkpointing
)
async def research(topic: str) -> Report:
    pass
```

**When NOT to use:**
- Simple Q&A chatbots (overkill)
- If you don't need cycles or branching
- Performance-critical with minimal latency

---

### CrewAI Adapter

**Best for**: Multi-agent collaboration, role-based teams

**Why choose CrewAI:**

1. **Role-based agents**: Each agent has expertise
2. **Task delegation**: Agents hand off work
3. **Hierarchical structure**: Manager agents coordinate
4. **Parallel execution**: Agents work concurrently

**Example use cases:**
- Research crew (researcher, writer, editor)
- Analysis team (data analyst, strategist, presenter)
- Content creation pipeline

```python
from crewai import Agent, Task, Crew
from fastagentic import agent_endpoint
from fastagentic.adapters.crewai import CrewAIAdapter

researcher = Agent(role="Researcher", goal="Find relevant information")
writer = Agent(role="Writer", goal="Create compelling content")
editor = Agent(role="Editor", goal="Ensure quality and accuracy")

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
)

@agent_endpoint(
    path="/content",
    runnable=CrewAIAdapter(crew),
)
async def create_content(topic: str) -> Article:
    pass
```

**When NOT to use:**
- Single-agent use cases
- Simple chatbots
- When you need fine-grained control over execution

---

### LangChain Adapter

**Best for**: Existing LangChain code, RAG pipelines, ecosystem tools

**Why choose LangChain:**

1. **Ecosystem**: Retrievers, memory, tools, integrations
2. **LCEL**: Composable pipeline syntax
3. **Existing code**: Wrap existing chains
4. **RAG support**: First-class retrieval support

**Example use cases:**
- Document Q&A with RAG
- Migrating existing LangChain applications
- Using LangChain-specific integrations

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fastagentic import agent_endpoint
from fastagentic.adapters.langchain import LangChainAdapter

prompt = ChatPromptTemplate.from_messages([...])
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm | output_parser

@agent_endpoint(
    path="/analyze",
    runnable=LangChainAdapter(chain),
)
async def analyze(question: str) -> Answer:
    pass
```

**When NOT to use:**
- Starting fresh (consider PydanticAI for cleaner code)
- Complex workflows (use LangGraph instead)
- When you don't need LangChain's ecosystem

---

### Custom Adapter (BaseAdapter)

**Best for**: Unique requirements, full control, non-standard frameworks

**Why choose custom:**

1. **Full control**: Implement exactly what you need
2. **Any framework**: Wrap anything that runs Python
3. **Minimal overhead**: No adapter abstraction layers
4. **Special requirements**: Unique streaming, checkpointing

```python
from fastagentic.adapters.base import BaseAdapter, AdapterContext

class MyCustomAdapter(BaseAdapter):
    async def invoke(self, input: dict, ctx: AdapterContext) -> dict:
        # Your custom logic
        result = await my_custom_agent.run(input)
        return {"output": result}

    async def stream(self, input: dict, ctx: AdapterContext):
        async for chunk in my_custom_agent.stream(input):
            yield {"type": "token", "content": chunk}
```

**When NOT to use:**
- If a standard adapter exists for your framework
- You want community support and documentation

---

## Comparison Table

| Aspect | PydanticAI | LangGraph | CrewAI | LangChain |
|--------|------------|-----------|--------|-----------|
| **Complexity** | Simple | Complex | Medium | Medium |
| **Learning curve** | Easy | Steep | Medium | Medium |
| **Type safety** | Excellent | Good | Good | Varies |
| **Streaming** | Token-level | Node-level | Per-agent | Per-chain |
| **Checkpointing** | Per-run | Per-node | Per-task | Per-step |
| **Cycles** | No | Yes | Via delegation | No |
| **Cold start** | Fast | Medium | Slow | Fast |
| **Memory usage** | Low | Medium | High | Low |

---

## Mixing Adapters

You can use different adapters for different endpoints:

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from fastagentic.adapters.langgraph import LangGraphAdapter

app = App(title="Multi-Framework")

# Quick chat uses PydanticAI
@agent_endpoint(path="/chat", runnable=PydanticAIAdapter(chat_agent))
async def chat(message: str) -> str:
    pass

# Complex workflow uses LangGraph
@agent_endpoint(path="/workflow", runnable=LangGraphAdapter(workflow))
async def workflow(input: WorkflowInput) -> WorkflowOutput:
    pass
```

**Why mix adapters?**
- Different use cases have different needs
- Migrate incrementally from one framework to another
- Use the right tool for each job

---

## Migration Guide

### From Raw Framework to FastAgentic

Your existing agent code stays the same. FastAgentic wraps it:

```python
# Before: Manual deployment
from my_framework import agent
result = await agent.run(input)

# After: FastAgentic deployment
from fastagentic import agent_endpoint
from fastagentic.adapters.myframework import MyAdapter

@agent_endpoint(path="/agent", runnable=MyAdapter(agent))
async def run_agent(input: Input) -> Output:
    pass
```

**What you gain:**
- REST + MCP + A2A protocols
- Authentication and authorization
- Durability (checkpoints, resume)
- Observability hooks
- Rate limiting and cost control

### Switching Adapters

If you need to switch frameworks:

1. **Both endpoints coexist** — Deploy the new adapter alongside the old
2. **Migrate traffic gradually** — Route a percentage to the new endpoint
3. **Note**: Checkpoints are adapter-specific — active runs use the old adapter

---

## Quick Reference

| I want... | Use this adapter |
|-----------|------------------|
| Type-safe responses | PydanticAI |
| Complex state machines | LangGraph |
| Multiple agent collaboration | CrewAI |
| Existing LangChain code | LangChain |
| Full control | Custom (BaseAdapter) |

---

## Next Steps

- [PydanticAI Adapter Details](../adapters/pydanticai.md)
- [LangGraph Adapter Details](../adapters/langgraph.md)
- [CrewAI Adapter Details](../adapters/crewai.md)
- [LangChain Adapter Details](../adapters/langchain.md)
- [Custom Adapter Guide](../adapters/custom.md)
- [Choosing Integrations](choosing-integrations.md) — What to add for production
