# FastAgentic Examples

Working examples demonstrating FastAgentic features.

## Examples

| Example | Description | Framework |
|---------|-------------|-----------|
| [pydanticai-chat](pydanticai-chat/) | Simple chat agent | PydanticAI |
| [langgraph-workflow](langgraph-workflow/) | Stateful research workflow | LangGraph |
| [crewai-multi-agent](crewai-multi-agent/) | Multi-agent collaboration | CrewAI |
| [full-production](full-production/) | Complete production setup | PydanticAI |

## Quick Start

```bash
# Choose an example
cd examples/pydanticai-chat

# Install dependencies
uv sync

# Set API key
export OPENAI_API_KEY="sk-..."

# Run
uv run fastagentic run

# Test
uv run fastagentic agent chat
```

## Example Structure

Each example includes:

```
example-name/
├── CLAUDE.md          # Claude Code instructions
├── app.py             # FastAgentic application
├── pyproject.toml     # Dependencies
├── .env.example       # Environment template
└── README.md          # Documentation
```

## Using Examples as Templates

```bash
# Copy an example as starting point
cp -r examples/pydanticai-chat my-new-agent
cd my-new-agent

# Customize and run
uv sync
uv run fastagentic run
```

## Local Development

Start the infrastructure stack:

```bash
# From project root
docker-compose up -d

# This starts:
# - Redis (localhost:6379)
# - PostgreSQL (localhost:5432)
# - OTEL Collector (localhost:4317)
# - Jaeger UI (localhost:16686)
# - Prometheus (localhost:9090)
# - Grafana (localhost:3000)
```

## Contributing Examples

1. Create new directory in `examples/`
2. Include `CLAUDE.md` with Claude Code instructions
3. Add working `app.py` with minimal dependencies
4. Include `pyproject.toml` and `README.md`
5. Test that it runs with `uv sync && uv run fastagentic run`
