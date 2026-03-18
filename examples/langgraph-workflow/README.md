# LangGraph Research Workflow

A stateful research workflow built with LangGraph and deployed with FastAgentic.

## Features

- Multi-step research workflow
- Iterative research with conditional routing
- State persistence (with Redis)
- Streaming progress updates

## Workflow

```
[Plan] → [Research] → [Analyze] → [Summarize]
              ↓
         [Continue?] → [Research]
```

## Quick Start

```bash
# Install
uv sync

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY

# Run
uv run fastagentic run

# Test
uv run fastagentic agent chat --endpoint /research
```

## API

```bash
# Start research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the benefits of renewable energy?"}'
```

## Customization

See `CLAUDE.md` for detailed modification instructions.

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAgentic LangGraph Adapter](../../docs/adapters/langgraph.md)
