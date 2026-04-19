# CrewAI Adapter

The CrewAI adapter wraps [CrewAI](https://www.crewai.com/) crews for deployment through FastAgentic. Deploy multi-agent collaboration workflows with role-based observability and durability.

## TL;DR

Wrap any CrewAI `Crew` and get REST + MCP + per-agent streaming + task-level checkpoints.

## Before FastAgentic

```python
from fastapi import FastAPI
from crewai import Agent, Task, Crew
import json

app = FastAPI()

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information",
    backstory="Expert researcher with access to multiple sources"
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content",
    backstory="Skilled writer who transforms research into articles"
)

research_task = Task(
    description="Research {topic}",
    agent=researcher
)

write_task = Task(
    description="Write article based on research",
    agent=writer
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])

@app.post("/research")
async def run_research(topic: str):
    result = crew.kickoff(inputs={"topic": topic})
    return {"result": result}

# No streaming, no durability, no auth, no cost tracking per agent...
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.crewai import CrewAIAdapter
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information",
    backstory="Expert researcher"
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content",
    backstory="Skilled writer"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        Task(description="Research {topic}", agent=researcher),
        Task(description="Write article", agent=writer),
    ]
)

app = App(
    title="Research Crew",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost",
)

@agent_endpoint(
    path="/research",
    runnable=CrewAIAdapter(crew),
    stream=True,
    durable=True,
)
async def research(topic: str) -> str:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /research` | Run crew synchronously |
| `POST /research/stream` | Run with per-agent streaming |
| `GET /research/{run_id}` | Get run status and result |
| `GET /research/{run_id}/events` | Replay event stream |
| `POST /research/{run_id}/resume` | Resume from checkpoint |

### CrewAI-Specific Features

**Per-Agent Streaming**

Events show which agent is working:

```
agent_start: {agent: "Research Analyst", task: "Research AI trends"}
token: {content: "Searching...", agent: "Research Analyst"}
tool_call: {tool: "search", agent: "Research Analyst"}
tool_result: {tool: "search", output: [...]}
agent_end: {agent: "Research Analyst", result: "..."}
task_complete: {task: "Research AI trends", agent: "Research Analyst"}
agent_start: {agent: "Content Writer", task: "Write article"}
...
```

**Task Delegation Tracking**

When agents delegate:
```
delegation: {from: "Research Analyst", to: "Content Writer", task: "..."}
```

**Role-Based Cost Attribution**

Cost tracking per agent:
```json
{
  "run_id": "run-123",
  "total_cost": 0.45,
  "by_agent": {
    "Research Analyst": 0.30,
    "Content Writer": 0.15
  }
}
```

**Crew Composition Visibility**

MCP schema includes crew structure:
```json
{
  "name": "research",
  "metadata": {
    "agents": [
      {"role": "Research Analyst", "goal": "..."},
      {"role": "Content Writer", "goal": "..."}
    ],
    "tasks": ["Research {topic}", "Write article"]
  }
}
```

## Configuration Options

### CrewAIAdapter Constructor

```python
CrewAIAdapter(
    crew: Crew,
    stream_agent_output: bool = True,
    stream_task_output: bool = True,
    checkpoint_tasks: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crew` | `Crew` | required | CrewAI crew instance |
| `stream_agent_output` | `bool` | `True` | Stream per-agent output |
| `stream_task_output` | `bool` | `True` | Stream per-task output |
| `checkpoint_tasks` | `bool` | `True` | Checkpoint after each task |

### Builder Methods

```python
# Configure checkpointing
adapter = CrewAIAdapter(crew).with_checkpoints(enabled=True)

# Enable verbose mode
adapter = CrewAIAdapter(crew).with_verbose()

# Configure streaming
adapter = CrewAIAdapter(crew).with_streaming(agent_output=True, task_output=True)
```

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `node_start` | Crew/agent begins | `{name, agents, tasks}` |
| `node_end` | Agent completes | `{name, task_index}` |
| `token` | LLM output (native streaming) | `{content}` |
| `tool_call` | Tool invoked | `{name, input}` or `{name, agent, task_index}` |
| `tool_result` | Tool returns | `{name, output}` |
| `checkpoint` | State saved | `{task_index, agent}` or `{step: "completed"}` |
| `done` | Crew complete | `{result, raw}` |

## Checkpoint State

The adapter automatically creates checkpoints containing:

```python
{
    "state": {
        "completed_tasks": int,  # Number of completed tasks
        "current_agent": str,    # Current agent role
        "completed": bool,       # Whether crew finished
    },
    "task_outputs": [
        {"task_index": 0, "agent": "Research Analyst"},
        {"task_index": 1, "agent": "Content Writer"},
    ],
    "context": {"result": "..."},  # Final result (on completion)
}
```

**Checkpoint Triggers:**
- After each task completion
- On crew completion

**Resume Behavior:**
When resuming, the adapter:
1. Restores completed task count and outputs
2. Sets `_is_resumed = True` on the run context
3. Emits a `NODE_START` event with `name: "__resume__"`

## Native Streaming (CrewAI 0.30+)

For CrewAI versions with event bus support, the adapter uses native token streaming:

```python
# Automatic detection - uses event bus if available
adapter = CrewAIAdapter(crew)

# Events from native streaming:
# node_start → token → token → tool_call → tool_result → checkpoint → node_end → done
```

**Event Bus Features:**
- Real-time token streaming via `LLMStreamChunkEvent`
- Tool events via `ToolUseEvent` and `ToolResultEvent`
- Falls back to polling-based streaming for older versions

## Common Patterns

### Research and Writing Crew

```python
from crewai import Agent, Task, Crew, Process

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
)

@agent_endpoint(
    path="/content",
    runnable=CrewAIAdapter(crew),
    stream=True,
    durable=True,
)
async def create_content(topic: str) -> Article:
    pass
```

### Hierarchical Crew

```python
crew = Crew(
    agents=[manager, analyst, developer],
    tasks=[planning_task, analysis_task, implementation_task],
    process=Process.hierarchical,
    manager_llm="openai:gpt-4",
)

@agent_endpoint(
    path="/project",
    runnable=CrewAIAdapter(crew),
    stream=True,
)
async def run_project(requirements: str) -> ProjectPlan:
    pass
```

### With Memory

```python
from crewai.memory import LongTermMemory

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    long_term_memory=LongTermMemory(),
)
```

FastAgentic preserves memory across checkpoints.

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [LangGraph Adapter](langgraph.md) - For stateful workflows
- [Custom Adapters](custom.md) - Build your own
