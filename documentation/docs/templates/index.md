# Project Templates

FastAgentic provides production-ready starter templates for each supported agent framework. Templates are hosted in the [fastagentic-templates](https://github.com/fastagentic/fastagentic-templates) repository, enabling community contributions and independent versioning.

Each template includes:

- Pre-configured `App` with MCP + A2A protocols
- Framework-specific adapter setup
- Sample agent implementation
- Docker and Kubernetes manifests
- GitHub Actions CI/CD
- Test scaffolding

## Quick Start

```bash
# List available templates (fetched from remote repository)
fastagentic templates list

# Create new project with template
fastagentic new my-agent --template pydanticai

# Or interactively choose
fastagentic new my-agent
# ? Select framework:
#   > PydanticAI (recommended)     [official]
#     LangGraph                     [official]
#     CrewAI                        [official]
#     LangChain                     [official]
#     AutoGen                       [community] ✓ verified
#     DSPy                          [community]
#     Minimal (no framework)        [official]
```

## Template Discovery

Templates are discovered from the remote index at runtime:

```bash
# View all templates with details
fastagentic templates list --verbose

# Filter by category
fastagentic templates list --category official
fastagentic templates list --category community

# Search by tag
fastagentic templates search "multi-agent"

# View template details
fastagentic templates info pydanticai
```

### Remote Index

The CLI fetches `index.json` from the templates repository:

```
https://github.com/fastagentic/fastagentic-templates/blob/main/index.json
```

This index is auto-generated when templates are added or updated, ensuring the CLI always has access to the latest templates.

### Local Templates

You can also use local templates that override remote ones:

```bash
# Use local template directory
fastagentic new my-agent --template /path/to/my-template

# Add to local template cache
mkdir -p ~/.fastagentic/templates/my-custom
# ... add template files ...
fastagentic new my-agent --template my-custom
```

## Available Templates

| Template | Best For | Framework Strength |
|----------|----------|-------------------|
| [**pydanticai**](pydanticai.md) | Type-safe agents, structured outputs | Pydantic validation, dependency injection |
| [**langgraph**](langgraph.md) | Stateful workflows, cycles | Graph-based control flow, interrupts |
| [**crewai**](crewai.md) | Multi-agent collaboration | Role-based agents, task delegation |
| [**langchain**](langchain.md) | Existing chains, RAG | LCEL composition, vast integrations |
| **minimal** | Custom frameworks | Bare FastAgentic setup |

## Template Structure

All templates follow the same directory structure:

```
my-agent/
├── app.py                    # FastAgentic App entry point
├── agents/
│   └── main.py               # Agent definition (framework-specific)
├── tools/
│   └── __init__.py           # Shared tools
├── models/
│   ├── inputs.py             # Pydantic input models
│   └── outputs.py            # Pydantic output models
├── config/
│   ├── settings.yaml         # Environment-specific settings
│   └── prompts/              # Prompt templates
├── tests/
│   ├── test_agent.py         # Agent tests
│   ├── test_tools.py         # Tool tests
│   └── test_contracts.py     # MCP/A2A schema parity
├── .env.example              # Environment variable template
├── Dockerfile                # Production container
├── docker-compose.yml        # Local development stack
├── k8s/
│   ├── deployment.yaml       # Kubernetes deployment
│   ├── service.yaml          # Service definition
│   └── ingress.yaml          # Ingress configuration
├── .github/
│   └── workflows/
│       ├── ci.yml            # CI pipeline
│       └── deploy.yml        # CD pipeline
├── pyproject.toml            # Dependencies
└── README.md                 # Project documentation
```

## What Each Template Includes

### Core Files

**app.py** - FastAgentic application:
```python
from fastagentic import App
from fastagentic.protocols import enable_mcp, enable_a2a
from agents.main import agent
from fastagentic.adapters.{framework} import {Framework}Adapter

app = App(
    title="My Agent",
    version="1.0.0",
    durable_store=os.getenv("DURABLE_STORE", "redis://localhost:6379"),
)

enable_mcp(app)
enable_a2a(app)

@app.agent_endpoint(
    path="/run",
    runnable={Framework}Adapter(agent),
    stream=True,
    durable=True,
    a2a_skill="my-agent",
)
async def run(input: AgentInput) -> AgentOutput:
    ...
```

**agents/main.py** - Framework-specific agent definition (varies by template)

### Configuration

**config/settings.yaml**:
```yaml
default:
  log_level: INFO
  telemetry: false

development:
  durable_store: redis://localhost:6379
  reload: true

production:
  durable_store: ${DURABLE_STORE}
  telemetry: true
  otel_endpoint: ${OTEL_ENDPOINT}
```

### Docker

**Dockerfile**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install .

COPY . .
EXPOSE 8000

CMD ["fastagentic", "run", "--host", "0.0.0.0"]
```

**docker-compose.yml**:
```yaml
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DURABLE_STORE=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Testing

**tests/test_contracts.py**:
```python
from fastagentic.testing import validate_mcp_schema, validate_a2a_card

def test_mcp_schema_valid():
    """Ensure MCP schema matches implementation."""
    validate_mcp_schema(app)

def test_a2a_card_valid():
    """Ensure A2A Agent Card is valid."""
    validate_a2a_card(app)

def test_schema_parity():
    """Ensure OpenAPI, MCP, and A2A schemas align."""
    from fastagentic.testing import assert_schema_parity
    assert_schema_parity(app)
```

## Creating a Project

### Interactive Mode

```bash
fastagentic new my-agent
```

Prompts for:
1. Framework selection
2. LLM provider (OpenAI, Anthropic, Google, etc.)
3. Durable store (Redis, PostgreSQL, none)
4. Authentication (OIDC, API key, none)
5. Observability (OTEL, none)

### Non-Interactive Mode

```bash
fastagentic new my-agent \
  --template pydanticai \
  --llm openai \
  --store redis \
  --auth oidc \
  --otel
```

### Minimal Setup

```bash
# Bare bones - just FastAgentic, no framework
fastagentic new my-agent --template minimal
```

## Running Templates

```bash
cd my-agent

# Install dependencies
pip install -e .

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
fastagentic run --reload

# Or with Docker
docker-compose up
```

## Customizing Templates

### Adding Tools

```python
# tools/search.py
from fastagentic import tool

@tool(name="web_search")
async def search(query: str) -> list[dict]:
    """Search the web for information."""
    # Implementation
    ...
```

### Adding Prompts

```python
# config/prompts/triage.py
from fastagentic import prompt

@prompt(name="triage_system")
def triage_prompt() -> str:
    return """You are a support triage assistant.
    Analyze tickets and assign priority and category."""
```

### Adding Resources

```python
# resources/knowledge.py
from fastagentic import resource

@resource(uri="knowledge/{topic}")
async def get_knowledge(topic: str) -> dict:
    """Fetch knowledge base entry."""
    ...
```

## Template Comparison

| Feature | PydanticAI | LangGraph | CrewAI | LangChain |
|---------|------------|-----------|--------|-----------|
| Type safety | Native | Manual | Manual | Manual |
| Streaming | Token-level | Node-level | Task-level | Token-level |
| State management | Deps injection | Graph state | Shared memory | Chain state |
| Multi-agent | Via A2A | Subgraphs | Built-in | Via A2A |
| Checkpointing | FastAgentic | Native + FA | FastAgentic | FastAgentic |
| Learning curve | Low | Medium | Medium | Low |

## Next Steps

Choose your template:
- [PydanticAI Template](pydanticai.md) - Recommended for new projects
- [LangGraph Template](langgraph.md) - For complex workflows
- [CrewAI Template](crewai.md) - For multi-agent systems
- [LangChain Template](langchain.md) - For existing LangChain users

Or learn more:
- [Adapters Guide](../adapters/index.md) - Framework adapter details
- [Protocol Support](../protocols/index.md) - MCP and A2A
- [Operations Guide](../operations/index.md) - Production deployment

## Template Repository

Templates are managed in a separate repository to enable:

- **Community Contributions**: Anyone can submit templates via PR
- **Independent Versioning**: Templates can update without core releases
- **Dynamic Discovery**: CLI fetches latest templates at runtime
- **Quality Control**: Automated validation and maintainer review

### Repository Structure

```
fastagentic-templates/
├── index.json              # Auto-generated template registry
├── templates/
│   ├── official/           # FastAgentic team maintained
│   │   ├── pydanticai/
│   │   ├── langgraph/
│   │   └── ...
│   └── community/          # Community contributed
│       ├── autogen/
│       ├── dspy/
│       └── ...
└── scripts/
    ├── build-index.py      # Generates index.json
    └── validate.py         # Validates templates
```

### Contributing

Want to add a template for your favorite framework?

1. [Repository Structure](repository.md) - Technical specification
2. [Contributing Guide](contributing.md) - Step-by-step instructions
3. [Template Examples](https://github.com/fastagentic/fastagentic-templates) - Reference implementations

### Template Index

The `index.json` file is automatically generated when templates are added or updated:

```json
{
  "version": "1.0.0",
  "templates": [
    {
      "name": "pydanticai",
      "category": "official",
      "version": "0.2.0",
      "features": {"streaming": true, "mcp": true, "a2a": true}
    }
  ]
}
```

See [Repository Specification](repository.md) for the complete schema.
