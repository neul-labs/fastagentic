# CrewAI Multi-Agent System

Multi-agent collaboration using CrewAI and FastAgentic.

## Agents

- **Researcher**: Gathers information
- **Analyst**: Analyzes data
- **Writer**: Creates summaries

## Quick Start

```bash
uv sync
cp .env.example .env  # Add OPENAI_API_KEY
uv run fastagentic run
```

## Test

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"topic": "artificial intelligence"}'
```

See `CLAUDE.md` for detailed instructions.
