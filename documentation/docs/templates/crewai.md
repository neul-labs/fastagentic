# CrewAI Template

Production-ready starter for multi-agent collaboration using [CrewAI](https://www.crewai.com/).

## Create Project

```bash
fastagentic new my-agent --template crewai
cd my-agent
```

## Project Structure

```
my-agent/
├── app.py                    # FastAgentic entry point
├── crews/
│   ├── __init__.py
│   └── content.py            # CrewAI crew definition
├── agents/
│   ├── __init__.py
│   ├── researcher.py         # Researcher agent
│   ├── writer.py             # Writer agent
│   └── editor.py             # Editor agent
├── tasks/
│   ├── __init__.py
│   └── content_tasks.py      # Task definitions
├── tools/
│   ├── __init__.py
│   └── search.py             # Custom tools
├── models/
│   ├── inputs.py
│   └── outputs.py
├── config/
│   └── settings.yaml
├── tests/
│   ├── test_crew.py
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
from fastagentic.adapters.crewai import CrewAIAdapter

from crews.content import content_crew
from models.inputs import ContentRequest
from models.outputs import ContentDeliverable

app = App(
    title="Content Creation Crew",
    version="1.0.0",
    description="Multi-agent content creation pipeline",
    durable_store=os.getenv("DURABLE_STORE", "redis://localhost:6379"),
)

enable_mcp(app, tasks_enabled=True)
enable_a2a(app)


@app.agent_endpoint(
    path="/create",
    runnable=CrewAIAdapter(
        content_crew,
        stream_tasks=True,        # Stream task completions
        include_agent_thoughts=True,  # Include agent reasoning
    ),
    input_model=ContentRequest,
    output_model=ContentDeliverable,
    stream=True,
    durable=True,
    mcp_tool="create_content",
    a2a_skill="content-crew",
)
async def create_content(request: ContentRequest) -> ContentDeliverable:
    """Execute content creation workflow with multiple agents."""
    ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### agents/researcher.py

```python
"""Researcher agent definition."""
from crewai import Agent
from langchain_openai import ChatOpenAI

from tools.search import web_search_tool, academic_search_tool


researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover comprehensive, accurate information on any topic",
    backstory="""You are a seasoned research analyst with 15 years of experience
    in investigative journalism and academic research. You excel at finding
    credible sources, cross-referencing information, and identifying key insights
    that others might miss. You're known for your thoroughness and attention
    to detail.""",
    tools=[web_search_tool, academic_search_tool],
    llm=ChatOpenAI(model="gpt-4o"),
    verbose=True,
    allow_delegation=True,  # Can delegate to other agents
    max_iter=5,
)
```

### agents/writer.py

```python
"""Writer agent definition."""
from crewai import Agent
from langchain_openai import ChatOpenAI


writer = Agent(
    role="Content Writer",
    goal="Create engaging, well-structured content that resonates with the target audience",
    backstory="""You are an award-winning content writer who has written for
    major publications. You have a gift for transforming complex research into
    compelling narratives. You adapt your tone and style to match the target
    audience while maintaining accuracy and depth.""",
    llm=ChatOpenAI(model="gpt-4o"),
    verbose=True,
    allow_delegation=False,
)
```

### agents/editor.py

```python
"""Editor agent definition."""
from crewai import Agent
from langchain_openai import ChatOpenAI


editor = Agent(
    role="Senior Editor",
    goal="Ensure content is polished, accurate, and publication-ready",
    backstory="""You are a meticulous editor with experience at top-tier
    publications. You have an eye for detail and ensure every piece meets
    the highest standards of quality. You check for clarity, consistency,
    grammar, and factual accuracy.""",
    llm=ChatOpenAI(model="gpt-4o"),
    verbose=True,
    allow_delegation=True,  # Can send back to writer
)
```

### tasks/content_tasks.py

```python
"""Task definitions for content creation."""
from crewai import Task

from agents.researcher import researcher
from agents.writer import writer
from agents.editor import editor


def create_research_task(topic: str, audience: str) -> Task:
    """Create the research task."""
    return Task(
        description=f"""Research the following topic comprehensively: {topic}

        Target audience: {audience}

        Deliverables:
        1. Key facts and statistics
        2. Expert opinions and quotes
        3. Current trends and developments
        4. Potential angles for the content
        5. List of credible sources

        Ensure all information is current and from reputable sources.""",
        agent=researcher,
        expected_output="Comprehensive research brief with sources",
    )


def create_writing_task(topic: str, format: str, tone: str) -> Task:
    """Create the writing task."""
    return Task(
        description=f"""Write a {format} about: {topic}

        Tone: {tone}

        Requirements:
        1. Use the research provided by the researcher
        2. Include relevant statistics and quotes
        3. Structure content with clear headings
        4. Include a compelling introduction and conclusion
        5. Optimize for readability

        Format: {format}""",
        agent=writer,
        expected_output=f"Complete {format} draft",
    )


def create_editing_task() -> Task:
    """Create the editing task."""
    return Task(
        description="""Review and polish the content draft.

        Check for:
        1. Factual accuracy (cross-reference with research)
        2. Grammar and spelling
        3. Clarity and flow
        4. Tone consistency
        5. Proper attribution of sources

        If major issues found, delegate back to writer for revisions.""",
        agent=editor,
        expected_output="Publication-ready content",
    )
```

### crews/content.py

```python
"""CrewAI crew definition."""
from crewai import Crew, Process

from agents.researcher import researcher
from agents.writer import writer
from agents.editor import editor
from tasks.content_tasks import (
    create_research_task,
    create_writing_task,
    create_editing_task,
)


def create_content_crew(
    topic: str,
    audience: str,
    format: str = "blog post",
    tone: str = "professional",
) -> Crew:
    """Create a content creation crew."""

    # Create tasks
    research_task = create_research_task(topic, audience)
    writing_task = create_writing_task(topic, format, tone)
    editing_task = create_editing_task()

    # Tasks depend on each other
    writing_task.context = [research_task]
    editing_task.context = [research_task, writing_task]

    return Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,  # Or Process.hierarchical
        verbose=True,
        memory=True,  # Enable crew memory
        max_rpm=10,   # Rate limit
    )


# Default crew instance for adapter
# FastAgentic's CrewAIAdapter will call kickoff() with inputs
content_crew = create_content_crew(
    topic="{topic}",      # Placeholder - replaced at runtime
    audience="{audience}",
    format="{format}",
    tone="{tone}",
)
```

### models/inputs.py

```python
"""Input models."""
from pydantic import BaseModel, Field
from typing import Literal


class ContentRequest(BaseModel):
    """Request to create content."""

    topic: str = Field(description="The topic to create content about")
    audience: str = Field(
        description="Target audience for the content",
        examples=["developers", "executives", "general public"],
    )
    format: Literal["blog post", "article", "whitepaper", "social thread"] = Field(
        default="blog post",
        description="Content format",
    )
    tone: Literal["professional", "casual", "technical", "conversational"] = Field(
        default="professional",
        description="Writing tone",
    )
    max_length: int | None = Field(
        default=None,
        description="Maximum word count",
    )
```

### models/outputs.py

```python
"""Output models."""
from pydantic import BaseModel, Field


class ContentDeliverable(BaseModel):
    """Final content deliverable."""

    title: str = Field(description="Content title")
    content: str = Field(description="Final content body")
    summary: str = Field(description="Brief summary")
    word_count: int = Field(description="Total word count")
    sources: list[str] = Field(description="List of sources used")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )
```

### tools/search.py

```python
"""Custom tools for the crew."""
from crewai_tools import BaseTool
from tavily import TavilyClient


class WebSearchTool(BaseTool):
    """Web search tool using Tavily."""

    name: str = "web_search"
    description: str = "Search the web for current information on any topic"

    def _run(self, query: str) -> str:
        client = TavilyClient()
        results = client.search(query, max_results=5)
        return "\n\n".join([
            f"**{r['title']}**\n{r['content']}\nSource: {r['url']}"
            for r in results["results"]
        ])


class AcademicSearchTool(BaseTool):
    """Academic paper search tool."""

    name: str = "academic_search"
    description: str = "Search for academic papers and research"

    def _run(self, query: str) -> str:
        # Implementation using Semantic Scholar, arXiv, etc.
        ...


web_search_tool = WebSearchTool()
academic_search_tool = AcademicSearchTool()
```

### tests/test_crew.py

```python
"""Crew tests."""
import pytest
from unittest.mock import patch, AsyncMock

from crews.content import create_content_crew
from models.inputs import ContentRequest


@pytest.mark.asyncio
async def test_crew_creation():
    """Test crew is created with correct agents."""
    crew = create_content_crew(
        topic="AI in healthcare",
        audience="healthcare executives",
    )
    assert len(crew.agents) == 3
    assert len(crew.tasks) == 3


@pytest.mark.asyncio
@patch("langchain_openai.ChatOpenAI")
async def test_crew_execution(mock_llm):
    """Test crew execution flow."""
    mock_llm.return_value.ainvoke = AsyncMock(
        return_value="Mocked response"
    )

    crew = create_content_crew(
        topic="Test topic",
        audience="Test audience",
    )

    # Test task dependencies
    assert crew.tasks[1].context == [crew.tasks[0]]
    assert crew.tasks[2].context == [crew.tasks[0], crew.tasks[1]]
```

## Configuration

### .env.example

```bash
# LLM Provider
OPENAI_API_KEY=sk-...

# Search APIs
TAVILY_API_KEY=tvly-...

# Durable Store
DURABLE_STORE=redis://localhost:6379

# CrewAI Settings
CREWAI_MEMORY_STORE=redis://localhost:6379

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
    "fastagentic[crewai]>=0.2.0",
    "crewai>=0.80.0",
    "crewai-tools>=0.14.0",
    "tavily-python>=0.5.0",
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

### Create Content

```bash
curl -X POST http://localhost:8000/create \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The future of remote work",
    "audience": "HR executives",
    "format": "whitepaper",
    "tone": "professional"
  }'
```

### Stream Progress

```bash
curl -N http://localhost:8000/create/stream \
  -H "Accept: text/event-stream" \
  -d '{"topic": "...", ...}'

# Streams agent/task events:
# event: task_start
# data: {"task": "Research", "agent": "Senior Research Analyst"}
#
# event: agent_thought
# data: {"agent": "researcher", "thought": "I'll start by..."}
#
# event: task_end
# data: {"task": "Research", "output": "..."}
```

## Streaming Events

The CrewAI adapter emits these events:

| Event | Description |
|-------|-------------|
| `task_start` | Task begins execution |
| `task_end` | Task completes |
| `agent_thought` | Agent reasoning (if enabled) |
| `delegation` | Agent delegates to another |
| `tool_use` | Agent uses a tool |
| `checkpoint` | State checkpointed |
| `cost` | Token/cost update |

## Cost Attribution

FastAgentic tracks costs per agent:

```json
{
  "event": "cost",
  "data": {
    "run_id": "run_abc123",
    "agent": "researcher",
    "task": "Research",
    "tokens": {"input": 1500, "output": 800},
    "cost": {"amount": 0.045, "currency": "USD"}
  }
}
```

## Next Steps

- [CrewAI Adapter](../adapters/crewai.md) - Full adapter documentation
- [Protocol Support](../protocols/index.md) - MCP and A2A details
- [Cost Tracking](../platform-services.md#cost-tracking) - FinOps features
