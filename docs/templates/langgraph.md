# LangGraph Template

Production-ready starter for stateful graph workflows using [LangGraph](https://langchain-ai.github.io/langgraph/).

## Create Project

```bash
fastagentic new my-agent --template langgraph
cd my-agent
```

## Project Structure

```
my-agent/
├── app.py                    # FastAgentic entry point
├── graphs/
│   ├── __init__.py
│   └── research.py           # LangGraph workflow
├── nodes/
│   ├── __init__.py
│   ├── search.py             # Search node
│   ├── analyze.py            # Analysis node
│   └── synthesize.py         # Synthesis node
├── state/
│   └── __init__.py           # Graph state definition
├── models/
│   ├── inputs.py
│   └── outputs.py
├── config/
│   └── settings.yaml
├── tests/
│   ├── test_graph.py
│   └── test_contracts.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── k8s/
│   └── *.yaml
└── pyproject.toml
```

## Core Files

### app.py

```python
"""FastAgentic application entry point."""
import os
from fastagentic import App
from fastagentic.protocols import enable_mcp, enable_a2a
from fastagentic.adapters.langgraph import LangGraphAdapter

from graphs.research import research_graph
from models.inputs import ResearchQuery
from models.outputs import ResearchReport

app = App(
    title="Research Agent",
    version="1.0.0",
    description="Multi-step research workflow with human-in-the-loop",
    durable_store=os.getenv("DURABLE_STORE", "redis://localhost:6379"),
)

enable_mcp(app, tasks_enabled=True)
enable_a2a(app)


@app.agent_endpoint(
    path="/research",
    runnable=LangGraphAdapter(
        research_graph,
        stream_mode="values",      # Stream full state updates
        checkpoint_enabled=True,   # Enable checkpointing
    ),
    input_model=ResearchQuery,
    output_model=ResearchReport,
    stream=True,
    durable=True,
    mcp_tool="deep_research",
    a2a_skill="research-agent",
)
async def research(query: ResearchQuery) -> ResearchReport:
    """Execute deep research workflow."""
    ...


# Resume endpoint for interrupted workflows
@app.post("/research/{run_id}/resume")
async def resume_research(run_id: str, input: dict | None = None):
    """Resume an interrupted research workflow."""
    return await app.resume_run(run_id, input=input)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### state/__init__.py

```python
"""Graph state definition."""
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """State passed between nodes in the research graph."""

    # Input
    query: str
    depth: str  # "quick", "standard", "deep"

    # Accumulated during workflow
    messages: Annotated[list, add_messages]
    search_results: list[dict]
    analyzed_sources: list[dict]
    key_findings: list[str]

    # Output
    report: str | None
    confidence: float | None

    # Control flow
    needs_human_review: bool
    human_feedback: str | None
```

### graphs/research.py

```python
"""LangGraph research workflow."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from nodes.search import search_node
from nodes.analyze import analyze_node
from nodes.synthesize import synthesize_node
from nodes.review import human_review_node


def should_continue(state: ResearchState) -> str:
    """Determine next step based on state."""
    if state.get("needs_human_review"):
        return "human_review"
    if len(state.get("analyzed_sources", [])) < 3 and state["depth"] == "deep":
        return "search"  # Continue searching
    return "synthesize"


def after_review(state: ResearchState) -> str:
    """Route after human review."""
    feedback = state.get("human_feedback", "")
    if "more research" in feedback.lower():
        return "search"
    if "reject" in feedback.lower():
        return END
    return "synthesize"


# Build the graph
builder = StateGraph(ResearchState)

# Add nodes
builder.add_node("search", search_node)
builder.add_node("analyze", analyze_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("human_review", human_review_node)

# Add edges
builder.set_entry_point("search")
builder.add_edge("search", "analyze")
builder.add_conditional_edges("analyze", should_continue)
builder.add_conditional_edges("human_review", after_review)
builder.add_edge("synthesize", END)

# Compile with checkpointing
# Note: FastAgentic wraps this with its own checkpoint store
research_graph = builder.compile()
```

### nodes/search.py

```python
"""Search node implementation."""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from state import ResearchState


llm = ChatOpenAI(model="gpt-4o")


async def search_node(state: ResearchState) -> dict:
    """Execute web search for research query."""
    query = state["query"]
    existing_results = state.get("search_results", [])

    # Generate search queries
    response = await llm.ainvoke([
        HumanMessage(content=f"""Generate 3 search queries to research: {query}

Already searched: {[r['query'] for r in existing_results]}

Return as JSON array of strings.""")
    ])

    search_queries = parse_queries(response.content)

    # Execute searches (using your search tool)
    new_results = []
    for sq in search_queries:
        results = await web_search(sq)
        new_results.append({
            "query": sq,
            "results": results,
        })

    return {
        "search_results": existing_results + new_results,
        "messages": [AIMessage(content=f"Searched: {search_queries}")],
    }
```

### nodes/analyze.py

```python
"""Analysis node implementation."""
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from state import ResearchState


llm = ChatOpenAI(model="gpt-4o")


async def analyze_node(state: ResearchState) -> dict:
    """Analyze search results and extract key findings."""
    search_results = state["search_results"]
    query = state["query"]

    # Analyze each source
    analyzed = []
    for result in search_results:
        analysis = await llm.ainvoke([
            HumanMessage(content=f"""Analyze this source for: {query}

Source: {result}

Extract:
1. Key facts
2. Relevance score (0-1)
3. Credibility assessment""")
        ])
        analyzed.append({
            "source": result,
            "analysis": analysis.content,
        })

    # Determine if human review needed
    low_confidence = any(
        "low credibility" in a["analysis"].lower()
        for a in analyzed
    )

    return {
        "analyzed_sources": analyzed,
        "needs_human_review": low_confidence and state["depth"] == "deep",
        "messages": [AIMessage(content=f"Analyzed {len(analyzed)} sources")],
    }
```

### nodes/synthesize.py

```python
"""Synthesis node implementation."""
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from state import ResearchState


llm = ChatOpenAI(model="gpt-4o")


async def synthesize_node(state: ResearchState) -> dict:
    """Synthesize findings into final report."""
    query = state["query"]
    analyzed = state["analyzed_sources"]
    human_feedback = state.get("human_feedback")

    prompt = f"""Create a comprehensive research report on: {query}

Based on these analyzed sources:
{analyzed}

{"Human feedback to incorporate: " + human_feedback if human_feedback else ""}

Include:
1. Executive summary
2. Key findings
3. Supporting evidence
4. Confidence level
5. Limitations"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    return {
        "report": response.content,
        "confidence": calculate_confidence(analyzed),
        "messages": [AIMessage(content="Report generated")],
    }
```

### nodes/review.py

```python
"""Human review node with interrupt."""
from langgraph.types import interrupt

from state import ResearchState


async def human_review_node(state: ResearchState) -> dict:
    """Pause for human review and feedback."""
    # This creates an interrupt - workflow pauses here
    # FastAgentic handles this via durable runs
    feedback = interrupt({
        "type": "human_review",
        "message": "Please review the analyzed sources",
        "sources": state["analyzed_sources"],
        "options": ["approve", "more research", "reject"],
    })

    return {
        "human_feedback": feedback,
        "needs_human_review": False,
    }
```

### tests/test_graph.py

```python
"""Graph workflow tests."""
import pytest
from graphs.research import research_graph
from state import ResearchState


@pytest.mark.asyncio
async def test_quick_research_flow():
    """Test quick research doesn't require human review."""
    initial_state = ResearchState(
        query="What is Python?",
        depth="quick",
        messages=[],
        search_results=[],
        analyzed_sources=[],
        key_findings=[],
        report=None,
        confidence=None,
        needs_human_review=False,
        human_feedback=None,
    )

    result = await research_graph.ainvoke(initial_state)

    assert result["report"] is not None
    assert result["confidence"] is not None
    assert not result["needs_human_review"]


@pytest.mark.asyncio
async def test_deep_research_triggers_review():
    """Test deep research with uncertain sources triggers review."""
    # Test with mock that returns low-confidence sources
    ...
```

## Configuration

### .env.example

```bash
# LLM Provider
OPENAI_API_KEY=sk-...

# Search API
TAVILY_API_KEY=tvly-...

# Durable Store
DURABLE_STORE=redis://localhost:6379

# Application
LOG_LEVEL=INFO
```

### pyproject.toml

```toml
[project]
name = "my-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastagentic[langgraph]>=0.2.0",
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "tavily-python>=0.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

## Running

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env

# Run
fastagentic run --reload
```

## API Usage

### Start Research

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Impact of AI on software development",
    "depth": "deep"
  }'

# Returns run_id for tracking
# {"run_id": "run_abc123", "status": "running"}
```

### Stream Progress

```bash
curl -N http://localhost:8000/research/stream \
  -H "Accept: text/event-stream" \
  -d '{"query": "...", "depth": "deep"}'

# Streams node events:
# event: node_start
# data: {"node": "search", "timestamp": "..."}
#
# event: node_end
# data: {"node": "search", "result": {...}}
#
# event: interrupt
# data: {"type": "human_review", "sources": [...]}
```

### Resume After Review

```bash
# After human reviews and provides feedback
curl -X POST http://localhost:8000/research/run_abc123/resume \
  -H "Content-Type: application/json" \
  -d '{"feedback": "approve"}'
```

## Human-in-the-Loop

LangGraph's `interrupt()` integrates with FastAgentic's durable runs:

1. Graph reaches interrupt node
2. FastAgentic checkpoints state, returns `awaiting_input` status
3. Client polls or receives push notification
4. Client submits input via `/resume` endpoint
5. Graph continues from checkpoint

```python
# Client handling
response = await client.post("/research", json={"query": "..."})
run_id = response.json()["run_id"]

# Poll for status
while True:
    status = await client.get(f"/runs/{run_id}")
    if status["state"] == "awaiting_input":
        # Show review UI, get feedback
        feedback = await get_human_feedback(status["interrupt_data"])
        await client.post(f"/research/{run_id}/resume", json=feedback)
    elif status["state"] == "completed":
        break
```

## Next Steps

- [LangGraph Adapter](../adapters/langgraph.md) - Full adapter documentation
- [Protocol Support](../protocols/index.md) - MCP and A2A details
- [Durable Runs](../platform-services.md#durability) - Checkpoint management
