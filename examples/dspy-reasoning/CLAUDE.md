# DSPy Reasoning Agent - Claude Code Guide

This is a FastAgentic example using DSPy for chain-of-thought reasoning.

## Project Structure

```
dspy-reasoning/
├── CLAUDE.md           # This file - instructions for Claude Code
├── app.py              # Main FastAgentic application with DSPy module
├── pyproject.toml      # Dependencies
├── .env.example        # Environment variables template
└── README.md           # User documentation
```

## Key Commands

```bash
# Install dependencies
uv sync

# Run the server
uv run fastagentic run

# Test with CLI
uv run fastagentic agent chat --endpoint /ask

# Run tests
uv run pytest tests/ -v
```

## Architecture

- **FastAgentic App** (`app.py`): Wraps the DSPy module with REST/MCP interfaces
- **DSPy Module**: ChainOfThought for reasoning with typed signatures
- **Streaming**: Field-level streaming via `stream_fields` parameter

## Key Components

### DSPy Signature

```python
class ReasoningQA(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()
```

### DSPyAdapter

```python
@agent_endpoint(
    path="/ask",
    runnable=DSPyAdapter(reasoning_module, stream_fields=["reasoning", "answer"]),
    stream=True,
    durable=True,
)
```

## When Modifying

1. **Change signature**: Add/modify fields in the `dspy.Signature` class
2. **Change module**: Use `dspy.ReAct`, `dspy.ProgramOfThought`, etc.
3. **Add tools**: Create `Tool` objects and pass to `dspy.ReAct`
4. **Enable tracing**: Use `DSPyAdapter(module, trace=True)`

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models
- `FASTAGENTIC_ENV`: Environment (dev/staging/prod)
- `REDIS_URL`: Optional durable store

## Testing

```bash
# Run tests
uv run pytest

# Test endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2 + 2?"}'
```

## Common Tasks

### Add a new output field

```python
class EnhancedQA(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    confidence: str = dspy.OutputField()
    answer: str = dspy.OutputField()
```

### Use ReAct with tools

```python
from dspy.predict.react import Tool

search = Tool(
    name="search",
    desc="Search the web for information",
    func=lambda q: web_search(q),
)

react_module = dspy.ReAct(ReasoningQA, tools=[search])
adapter = DSPyAdapter(react_module)
```

### Enable trace in checkpoint

```python
@agent_endpoint(
    path="/ask",
    runnable=DSPyAdapter(module, trace=True),
    durable=True,
)
```
