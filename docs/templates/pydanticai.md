# PydanticAI Template

Production-ready starter for type-safe agents using [PydanticAI](https://ai.pydantic.dev/).

## Create Project

```bash
fastagentic new my-agent --template pydanticai
cd my-agent
```

## Project Structure

```
my-agent/
├── app.py                    # FastAgentic entry point
├── agents/
│   └── main.py               # PydanticAI agent
├── tools/
│   ├── __init__.py
│   └── search.py             # Example tools
├── models/
│   ├── inputs.py             # Input models
│   └── outputs.py            # Output models
├── deps/
│   └── __init__.py           # Dependency injection
├── config/
│   └── settings.yaml
├── tests/
│   ├── test_agent.py
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
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from fastagentic.auth import configure_oidc
from fastagentic.telemetry import configure_otel

from agents.main import support_agent
from models.inputs import TicketInput
from models.outputs import TriageResult
from deps import create_deps

# Initialize app
app = App(
    title="Support Triage Agent",
    version="1.0.0",
    description="AI-powered support ticket triage",
    durable_store=os.getenv("DURABLE_STORE", "redis://localhost:6379"),
)

# Enable protocols
enable_mcp(app, tasks_enabled=True)
enable_a2a(app)

# Optional: Authentication
if os.getenv("OIDC_ISSUER"):
    configure_oidc(
        app,
        issuer=os.getenv("OIDC_ISSUER"),
        audience=os.getenv("OIDC_AUDIENCE", "support-triage"),
    )

# Optional: Observability
if os.getenv("OTEL_ENDPOINT"):
    configure_otel(
        app,
        service_name="support-triage",
        endpoint=os.getenv("OTEL_ENDPOINT"),
    )


# Register agent endpoint
@app.agent_endpoint(
    path="/triage",
    runnable=PydanticAIAdapter(
        support_agent,
        deps_factory=create_deps,
        stream_tokens=True,
    ),
    input_model=TicketInput,
    output_model=TriageResult,
    stream=True,
    durable=True,
    # Protocol exposure
    mcp_tool="triage_ticket",
    a2a_skill="support-triage",
)
async def triage_ticket(ticket: TicketInput) -> TriageResult:
    """Triage a support ticket."""
    ...


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### agents/main.py

```python
"""PydanticAI agent definition."""
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from models.inputs import TicketInput
from models.outputs import TriageResult
from deps import AgentDeps

# Initialize model
model = OpenAIModel("gpt-4o")

# Create agent
support_agent = Agent(
    model=model,
    result_type=TriageResult,
    deps_type=AgentDeps,
    system_prompt="""You are a support triage assistant.

Your job is to analyze support tickets and:
1. Determine the priority (low, medium, high, urgent)
2. Assign a category (billing, technical, account, feature_request, other)
3. Suggest the best team to handle the ticket
4. Provide a brief summary

Be concise and accurate. Use the available tools to look up customer information.""",
)


@support_agent.tool
async def get_customer_tier(ctx, customer_id: str) -> str:
    """Look up customer's subscription tier."""
    # Access deps for database/API calls
    customer = await ctx.deps.customer_service.get(customer_id)
    return customer.tier if customer else "unknown"


@support_agent.tool
async def search_knowledge_base(ctx, query: str) -> list[str]:
    """Search the knowledge base for relevant articles."""
    results = await ctx.deps.knowledge_base.search(query, limit=3)
    return [r.title for r in results]


@support_agent.tool
async def get_recent_tickets(ctx, customer_id: str, limit: int = 5) -> list[dict]:
    """Get customer's recent support tickets."""
    tickets = await ctx.deps.ticket_service.get_recent(customer_id, limit)
    return [{"id": t.id, "subject": t.subject, "status": t.status} for t in tickets]
```

### models/inputs.py

```python
"""Input models for the agent."""
from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """Support ticket to triage."""

    ticket_id: str = Field(description="Unique ticket identifier")
    customer_id: str = Field(description="Customer identifier")
    subject: str = Field(description="Ticket subject line")
    description: str = Field(description="Full ticket description")
    channel: str = Field(
        default="web",
        description="Source channel",
        json_schema_extra={"enum": ["web", "email", "chat", "phone"]},
    )
```

### models/outputs.py

```python
"""Output models for the agent."""
from pydantic import BaseModel, Field
from typing import Literal


class TriageResult(BaseModel):
    """Triage decision for a support ticket."""

    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Ticket priority level"
    )
    category: Literal["billing", "technical", "account", "feature_request", "other"] = Field(
        description="Ticket category"
    )
    team: str = Field(description="Suggested team to handle the ticket")
    summary: str = Field(description="Brief summary of the issue")
    suggested_response: str | None = Field(
        default=None,
        description="Optional suggested initial response",
    )
    escalation_needed: bool = Field(
        default=False,
        description="Whether immediate escalation is recommended",
    )
```

### deps/__init__.py

```python
"""Dependency injection for the agent."""
from dataclasses import dataclass
from fastapi import Request

from services.customer import CustomerService
from services.knowledge import KnowledgeBaseService
from services.tickets import TicketService


@dataclass
class AgentDeps:
    """Dependencies available to the agent."""

    customer_service: CustomerService
    knowledge_base: KnowledgeBaseService
    ticket_service: TicketService
    user_id: str | None = None


def create_deps(request: Request) -> AgentDeps:
    """Create dependencies from request context."""
    return AgentDeps(
        customer_service=request.app.state.customer_service,
        knowledge_base=request.app.state.knowledge_base,
        ticket_service=request.app.state.ticket_service,
        user_id=getattr(request.state, "user_id", None),
    )
```

### tests/test_agent.py

```python
"""Agent tests."""
import pytest
from pydantic_ai.models.test import TestModel
from agents.main import support_agent
from models.inputs import TicketInput
from deps import AgentDeps


@pytest.fixture
def test_model():
    """Use test model for deterministic responses."""
    return TestModel()


@pytest.fixture
def mock_deps():
    """Mock dependencies."""
    return AgentDeps(
        customer_service=MockCustomerService(),
        knowledge_base=MockKnowledgeBase(),
        ticket_service=MockTicketService(),
    )


@pytest.mark.asyncio
async def test_triage_basic(test_model, mock_deps):
    """Test basic triage flow."""
    with support_agent.override(model=test_model):
        result = await support_agent.run(
            "Triage this ticket",
            deps=mock_deps,
        )
        assert result.data.priority in ["low", "medium", "high", "urgent"]
        assert result.data.category is not None


@pytest.mark.asyncio
async def test_triage_urgent_detection(test_model, mock_deps):
    """Test urgent ticket detection."""
    test_model.seed_response(
        '{"priority": "urgent", "category": "technical", ...}'
    )
    with support_agent.override(model=test_model):
        result = await support_agent.run(
            "CRITICAL: Production is down!",
            deps=mock_deps,
        )
        assert result.data.priority == "urgent"
        assert result.data.escalation_needed is True
```

### tests/test_contracts.py

```python
"""Protocol contract tests."""
from fastagentic.testing import (
    validate_mcp_schema,
    validate_a2a_card,
    assert_schema_parity,
)
from app import app


def test_mcp_schema():
    """MCP schema is valid."""
    errors = validate_mcp_schema(app)
    assert not errors, f"MCP schema errors: {errors}"


def test_a2a_card():
    """A2A Agent Card is valid."""
    errors = validate_a2a_card(app)
    assert not errors, f"A2A card errors: {errors}"


def test_schema_parity():
    """OpenAPI, MCP, and A2A schemas align."""
    assert_schema_parity(app)
```

## Configuration

### .env.example

```bash
# LLM Provider
OPENAI_API_KEY=sk-...

# Durable Store
DURABLE_STORE=redis://localhost:6379

# Authentication (optional)
OIDC_ISSUER=https://auth.example.com
OIDC_AUDIENCE=support-triage

# Observability (optional)
OTEL_ENDPOINT=http://localhost:4318

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
    "fastagentic[pydanticai]>=0.2.0",
    "pydantic-ai>=0.1.0",
    "redis>=5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

## Running

### Local Development

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run with reload
fastagentic run --reload
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or production image
docker build -t my-agent .
docker run -p 8000:8000 --env-file .env my-agent
```

### Kubernetes

```bash
# Create secrets
kubectl create secret generic agent-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY

# Deploy
kubectl apply -f k8s/
```

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=agents --cov=models

# Contract tests only
pytest tests/test_contracts.py
```

## API Usage

### REST

```bash
# Triage a ticket
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "T-123",
    "customer_id": "C-456",
    "subject": "Cannot login",
    "description": "Getting 401 error when trying to login..."
  }'
```

### Streaming

```bash
# Stream triage response
curl -N http://localhost:8000/triage/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"ticket_id": "T-123", ...}'
```

### MCP

```bash
# Via MCP tool
fastagentic mcp call triage_ticket --input '{"ticket_id": "T-123", ...}'
```

### A2A

```bash
# Via A2A skill
fastagentic a2a invoke support-triage --input '{"ticket_id": "T-123", ...}'
```

## Extending

### Add a Tool

```python
# In agents/main.py

@support_agent.tool
async def check_service_status(ctx, service: str) -> dict:
    """Check if a service is experiencing issues."""
    status = await ctx.deps.status_page.get_service(service)
    return {
        "service": service,
        "status": status.state,
        "incidents": [i.title for i in status.active_incidents],
    }
```

### Add Validation

```python
# In models/inputs.py

from pydantic import field_validator

class TicketInput(BaseModel):
    ...

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v
```

### Add Streaming Events

```python
# In app.py

@app.agent_endpoint(
    path="/triage",
    runnable=PydanticAIAdapter(
        support_agent,
        stream_tokens=True,
        include_tool_calls=True,  # Stream tool invocations
        include_cost=True,        # Stream cost updates
    ),
    ...
)
```

## Next Steps

- [PydanticAI Adapter](../adapters/pydanticai.md) - Full adapter documentation
- [Protocol Support](../protocols/index.md) - MCP and A2A details
- [Operations Guide](../operations/index.md) - Production deployment
