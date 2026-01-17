# DSPy Reasoning Agent

A chain-of-thought reasoning agent built with DSPy and deployed with FastAgentic.

## Features

- DSPy ChainOfThought module for step-by-step reasoning
- Native streaming with field-level output
- Checkpointing for durability
- Trace integration for debugging
- REST, MCP, and A2A interfaces

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the server
uv run fastagentic run

# 4. Test with CLI
uv run fastagentic agent chat --endpoint /ask
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question with reasoning |
| `/ask/stream` | POST | Ask with streaming response |
| `/ask/{run_id}` | GET | Get run status |
| `/ask/{run_id}/resume` | POST | Resume from checkpoint |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/mcp/schema` | GET | MCP schema |

## Testing

```bash
# Simple question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 15% of 80?"}'

# Complex reasoning
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "If all cats are mammals and some mammals are pets, can we conclude that some cats are pets?"}'

# Streaming response
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "Explain why the sky is blue"}'

# Using the Agent CLI
uv run fastagentic agent chat --endpoint /ask
```

## Response Format

The agent returns structured responses with reasoning:

```json
{
  "reasoning": "Step 1: Calculate 15% as a decimal: 15/100 = 0.15\nStep 2: Multiply 0.15 by 80: 0.15 * 80 = 12",
  "answer": "15% of 80 is 12"
}
```

## DSPy Modules

### ChainOfThought

The default module encourages step-by-step reasoning:

```python
class ReasoningQA(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()

module = dspy.ChainOfThought(ReasoningQA)
```

### ReAct (with tools)

Add tools for more complex tasks:

```python
from dspy.predict.react import Tool

search = Tool(name="search", desc="Search the web", func=search_web)
module = dspy.ReAct(ReasoningQA, tools=[search])
```

## Project Structure

```
dspy-reasoning/
├── app.py              # FastAgentic application with DSPy module
├── pyproject.toml      # Dependencies
├── .env.example        # Environment template
├── CLAUDE.md           # Claude Code instructions
└── README.md           # This file
```

## Optimization

DSPy supports prompt optimization:

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(module, trainset=examples)

# Deploy optimized module
@agent_endpoint(
    path="/ask",
    runnable=DSPyAdapter(optimized),
)
```

## Learn More

- [FastAgentic Documentation](https://github.com/neul-labs/fastagentic)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy Adapter Guide](../../docs/adapters/dspy.md)
