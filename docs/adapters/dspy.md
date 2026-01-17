# DSPy Adapter

The DSPy adapter wraps [DSPy](https://dspy-docs.vercel.app/) modules for deployment through FastAgentic. Deploy optimized prompt pipelines with automatic checkpointing and tool event streaming.

## TL;DR

Wrap any DSPy `Module` and get REST + MCP + streaming + durability + tool events from ReAct traces.

## Why DSPy + FastAgentic?

DSPy excels at prompt optimization and module composition. FastAgentic adds production deployment:

| Capability | DSPy | FastAgentic |
|------------|------|-------------|
| Prompt optimization | Built-in | Inherited |
| Module composition | Built-in | Inherited |
| Few-shot learning | Built-in | Inherited |
| Signatures | Built-in | Inherited |
| REST API | Manual | Automatic |
| MCP Protocol | No | Yes |
| Token streaming | Via streamify | Automatic fallback |
| Checkpoints | No | Built-in |
| Tool event tracking | Via trace | Automatic |

**Optimize with DSPy. Deploy with FastAgentic.**

## Before FastAgentic

```python
from fastapi import FastAPI
import dspy

app = FastAPI()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa_module = dspy.ChainOfThought(QA)

@app.post("/qa")
async def answer(question: str):
    result = qa_module(question=question)
    return {"answer": result.answer}

# No streaming, no checkpoints, no tool tracking, no auth...
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.dspy import DSPyAdapter
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa_module = dspy.ChainOfThought(QA)

app = App(
    title="QA Agent",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost",
)

@agent_endpoint(
    path="/qa",
    runnable=DSPyAdapter(qa_module),
    stream=True,
    durable=True,
)
async def answer(question: str) -> str:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /qa` | Run module synchronously |
| `POST /qa/stream` | Run with token streaming |
| `GET /qa/{run_id}` | Get run status and result |
| `POST /qa/{run_id}/resume` | Resume from checkpoint |

### DSPy-Specific Features

**Native Streaming (with dspy.streamify)**

When available, uses DSPy's native streaming:

```python
adapter = DSPyAdapter(module, stream_fields=["answer"])

# Native streaming events:
# token → token → token → checkpoint → done
```

**Tool Events from ReAct Traces**

For ReAct modules, automatically extracts tool calls:

```python
react_module = dspy.ReAct(QA, tools=[search_tool])
adapter = DSPyAdapter(react_module)

# Events emitted:
# token → tool_call → tool_result → token → checkpoint → done
```

**Trace Integration**

Enable tracing for debugging:

```python
adapter = DSPyAdapter(module, trace=True)

# Checkpoint includes trace data:
# {"state": {...}, "context": {"trace": {...}}}
```

## Configuration Options

### DSPyAdapter Constructor

```python
DSPyAdapter(
    module: dspy.Module,
    trace: bool = False,
    stream_fields: list[str] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | DSPy module instance |
| `trace` | `bool` | `False` | Enable trace in checkpoints |
| `stream_fields` | `list[str]` | `None` | Signature fields to stream (for native streaming) |

### Builder Methods

```python
# Enable tracing
adapter = DSPyAdapter(module).with_trace(enabled=True)

# Configure streaming fields
adapter = DSPyAdapter(module).with_stream_fields(["answer", "reasoning"])
```

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `token` | Output token (native or simulated) | `{content}` or `{content, field}` |
| `tool_call` | Tool invoked (ReAct) | `{name, input}` |
| `tool_result` | Tool returns (ReAct) | `{name, output}` |
| `checkpoint` | State saved | `{step: "completed"}` |
| `done` | Module complete | `{result}` |

## Checkpoint State

The adapter automatically creates checkpoints containing:

```python
{
    "state": {
        "result": {...},    # Module output fields
        "completed": True,
    },
    "context": {
        "trace": {...},     # If trace=True
    },
}
```

**Checkpoint Triggers:**
- On module completion

**Resume Behavior:**
When resuming, the adapter:
1. Returns cached result if completed
2. Sets `_is_resumed = True` on the run context
3. Emits a `NODE_START` event with `name: "__resume__"`

## Tool Event Extraction

For ReAct and tool-using modules, the adapter extracts tool events from the action history:

```python
def _extract_tool_events(self, result):
    # Looks for result.actions or result.action_history
    for action in actions:
        yield {"type": "tool_call", "name": action.tool, "input": action.tool_input}
        yield {"type": "tool_result", "name": action.tool, "output": action.observation}
```

## Common Patterns

### Chain of Thought

```python
class Analysis(dspy.Signature):
    text: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()

cot = dspy.ChainOfThought(Analysis)

@agent_endpoint(
    path="/analyze",
    runnable=DSPyAdapter(cot, stream_fields=["reasoning", "sentiment"]),
    stream=True,
)
async def analyze(text: str) -> dict:
    pass
```

### ReAct Agent

```python
from dspy.predict.react import Tool

search = Tool(
    name="search",
    desc="Search the web",
    func=lambda query: web_search(query),
)

react = dspy.ReAct(QA, tools=[search])

@agent_endpoint(
    path="/research",
    runnable=DSPyAdapter(react),
    stream=True,
    durable=True,
)
async def research(question: str) -> str:
    pass
```

### Optimized Module

```python
# Train and optimize your module
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(qa_module, trainset=train_data)

# Deploy the optimized version
@agent_endpoint(
    path="/qa",
    runnable=DSPyAdapter(optimized),
)
async def qa(question: str) -> str:
    pass
```

## Streaming Modes

### Native Streaming (Preferred)

Uses `dspy.streamify()` when available:

```python
adapter = DSPyAdapter(module, stream_fields=["answer"])

# Real-time token streaming per field
```

### Simulated Streaming (Fallback)

When native streaming unavailable, simulates by chunking output:

```python
# Chunks output text into ~20 character pieces
# Still provides progressive output experience
```

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [PydanticAI Adapter](pydanticai.md) - For type-safe agents
- [LlamaIndex Adapter](llamaindex.md) - For RAG applications
