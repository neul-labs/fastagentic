# FastAgentic - Claude Code Guide

FastAgentic is the deployment layer for agentic applications. Build agents with anything. Ship them with FastAgentic.

## Project Overview

```
fastagentic/
├── src/fastagentic/       # Main source code
│   ├── adapters/          # Framework adapters (PydanticAI, LangGraph, etc.)
│   ├── cli/               # CLI commands
│   ├── protocols/         # MCP, A2A protocol implementations
│   ├── integrations/      # Third-party integrations
│   └── ...                # Core modules
├── tests/                 # Test suite (608+ tests)
├── docs/                  # Documentation (66 files)
├── examples/              # Working examples
└── templates/             # Project templates
```

## Quick Commands

```bash
# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check src/

# Run type checking
uv run mypy src/

# Start development server
uv run fastagentic run --reload

# Test with Agent CLI
uv run fastagentic agent chat
```

## Key Architecture

### Decorators
- `@tool` - Expose function as MCP tool
- `@resource` - Expose data as MCP resource
- `@prompt` - Expose prompt template
- `@agent_endpoint` - Create agent API endpoint

### Adapters
Connect any agent framework to FastAgentic:
- `PydanticAIAdapter` - PydanticAI agents
- `LangGraphAdapter` - LangGraph workflows
- `CrewAIAdapter` - CrewAI crews
- `LangChainAdapter` - LangChain runnables

### Core Modules
| Module | Purpose |
|--------|---------|
| `app.py` | Main App class |
| `decorators.py` | Decorator implementations |
| `context.py` | Request context |
| `reliability.py` | Retry, circuit breaker |
| `policy.py` | RBAC, scopes, budgets |
| `compliance.py` | PII detection |
| `sdk/` | Python client SDK |
| `cli/` | CLI commands |

## When Modifying Code

### Adding a new adapter

1. Create `src/fastagentic/adapters/new_adapter.py`
2. Implement `BaseAdapter` interface
3. Add to `src/fastagentic/adapters/__init__.py`
4. Add tests in `tests/test_adapters_new.py`
5. Add documentation in `docs/adapters/`

### Adding a new CLI command

1. Edit `src/fastagentic/cli/main.py`
2. Add command with `@app.command()` or create subgroup
3. Update `docs/cli-reference.md`

### Adding a new integration

1. Create `src/fastagentic/integrations/new_integration.py`
2. Implement hook or provider interface
3. Add to `src/fastagentic/integrations/__init__.py`
4. Add documentation in `docs/integrations/`

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_app.py -v

# Run with coverage
uv run pytest tests/ --cov=src/fastagentic --cov-report=html

# Run only fast tests
uv run pytest tests/ -m "not slow"
```

## Code Style

- Python 3.10+ with full type hints
- Pydantic models for data validation
- Async by default
- Docstrings for all public APIs
- ruff for linting, mypy for type checking

## Version History

| Version | Features |
|---------|----------|
| v1.2 | Agent CLI, examples, templates |
| v1.1 | New adapters, template ecosystem |
| v1.0 | SDK, PII detection, dashboard |
| v0.5 | Cluster orchestration |
| v0.4 | HITL, prompt management |
| v0.3 | Policy, integrations |

## Common Tasks

### Run the development server

```bash
cd examples/pydanticai-chat
uv sync
uv run fastagentic run --reload
```

### Add a new example

1. Create directory in `examples/`
2. Add `CLAUDE.md`, `app.py`, `pyproject.toml`, `README.md`
3. Update `examples/README.md`

### Update documentation

1. Edit markdown files in `docs/`
2. Update `docs/index.md` for new pages
3. Update README.md Learn More section

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `FASTAGENTIC_ENV` | Environment (dev/staging/prod) |
| `REDIS_URL` | Redis connection URL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTEL collector endpoint |

## Links

- [Documentation](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Examples](examples/)
- [API Reference](docs/decorators.md)
