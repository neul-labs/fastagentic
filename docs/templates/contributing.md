# Contributing Templates

FastAgentic welcomes community template contributions. This guide explains how to create, test, and submit templates.

## Quick Start

```bash
# 1. Fork the templates repository
gh repo fork fastagentic/fastagentic-templates

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/fastagentic-templates
cd fastagentic-templates

# 3. Create your template
mkdir -p templates/community/my-framework
cd templates/community/my-framework

# 4. Use the template scaffolder
python scripts/scaffold-template.py my-framework

# 5. Customize and test
# ... edit files ...
python scripts/validate.py templates/community/my-framework
python scripts/test-templates.py my-framework

# 6. Submit PR
git add .
git commit -m "feat: add my-framework template"
git push origin main
gh pr create
```

## Template Requirements

### Required Files

| File | Purpose |
|------|---------|
| `template.json` | Metadata and configuration |
| `app.py` | FastAgentic entry point |
| `.env.example` | Environment variable template |
| `pyproject.toml` | Dependencies |
| `Dockerfile` | Container build |
| `README.md` | Documentation |
| `tests/test_*.py` | At least one test file |

### Recommended Files

| File | Purpose |
|------|---------|
| `agents/main.py` | Agent definition |
| `models/inputs.py` | Input Pydantic models |
| `models/outputs.py` | Output Pydantic models |
| `tools/__init__.py` | Custom tools |
| `docker-compose.yml` | Local development stack |
| `k8s/*.yaml` | Kubernetes manifests |
| `.github/workflows/*.yml` | CI/CD pipelines |

## Creating `template.json`

The metadata file defines your template:

```json
{
  "$schema": "https://raw.githubusercontent.com/fastagentic/fastagentic-templates/main/schema/template.schema.json",
  "name": "my-framework",
  "display_name": "My Framework",
  "description": "Brief description of what this template provides",
  "version": "0.1.0",
  "min_fastagentic_version": "0.2.0",
  "framework": "my-framework",
  "author": {
    "name": "Your Name",
    "github": "your-github-username",
    "url": "https://your-website.com"
  },
  "license": "MIT",
  "tags": ["tag1", "tag2", "tag3"],
  "features": {
    "streaming": true,
    "checkpointing": true,
    "mcp": true,
    "a2a": true
  },
  "prompts": [
    {
      "name": "project_name",
      "message": "Project name",
      "default": "my-agent",
      "validate": "^[a-z][a-z0-9-]*$"
    }
  ],
  "post_create": [
    "pip install -e .",
    "cp .env.example .env"
  ],
  "post_create_message": "Success! Edit .env, then run: fastagentic run --reload"
}
```

### Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier (lowercase, hyphens only) |
| `display_name` | Yes | Human-readable name |
| `description` | Yes | Brief description (max 200 chars) |
| `version` | Yes | Semantic version |
| `min_fastagentic_version` | Yes | Minimum FastAgentic version |
| `framework` | Yes | Framework name (e.g., "autogen", "dspy") |
| `author` | Yes | Author information |
| `license` | Yes | SPDX license identifier |
| `tags` | No | Searchable tags (max 5) |
| `features` | No | Feature flags |
| `prompts` | No | Interactive prompts during creation |
| `post_create` | No | Commands to run after creation |
| `post_create_message` | No | Message shown after creation |

### Feature Flags

| Flag | Description |
|------|-------------|
| `streaming` | Supports token/event streaming |
| `checkpointing` | Supports durable checkpoints |
| `mcp` | Exposed via MCP protocol |
| `a2a` | Exposed via A2A protocol |
| `interrupts` | Supports human-in-the-loop |

## Creating `app.py`

The entry point must follow this pattern:

```python
"""FastAgentic application entry point."""
import os
from fastagentic import App
from fastagentic.protocols import enable_mcp, enable_a2a
from fastagentic.adapters.base import BaseAdapter  # Or your framework's adapter

# Your framework imports
from my_framework import MyAgent

from models.inputs import MyInput
from models.outputs import MyOutput


# Create custom adapter if needed
class MyFrameworkAdapter(BaseAdapter):
    """Adapter for My Framework."""

    def __init__(self, agent):
        self.agent = agent

    async def invoke(self, input: dict, config: dict | None = None):
        return await self.agent.run(input)

    async def stream(self, input: dict, config: dict | None = None):
        async for event in self.agent.stream(input):
            yield self.map_event(event)


# Initialize app
app = App(
    title="{{ project_name }}",  # Template variable
    version="1.0.0",
    durable_store=os.getenv("DURABLE_STORE", "redis://localhost:6379"),
)

# Enable protocols
enable_mcp(app)
enable_a2a(app)


# Define agent
my_agent = MyAgent(...)


# Register endpoint
@app.agent_endpoint(
    path="/run",
    runnable=MyFrameworkAdapter(my_agent),
    input_model=MyInput,
    output_model=MyOutput,
    stream=True,
    durable=True,
    mcp_tool="my_agent",
    a2a_skill="my-agent",
)
async def run_agent(input: MyInput) -> MyOutput:
    """Run the agent."""
    ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Template Variables

Use `{{ variable_name }}` for values that should be replaced during project creation:

| Variable | Description |
|----------|-------------|
| `{{ project_name }}` | User-provided project name |
| `{{ project_name_snake }}` | Snake_case version |
| `{{ project_name_pascal }}` | PascalCase version |
| `{{ author_name }}` | Git user name |
| `{{ author_email }}` | Git user email |
| `{{ llm_provider }}` | Selected LLM provider |
| `{{ durable_store }}` | Selected durable store |

## Writing Tests

Include meaningful tests:

```python
# tests/test_agent.py
"""Agent tests."""
import pytest
from app import app
from models.inputs import MyInput


@pytest.mark.asyncio
async def test_agent_endpoint_exists():
    """Verify agent endpoint is registered."""
    routes = [r.path for r in app.routes]
    assert "/run" in routes


@pytest.mark.asyncio
async def test_agent_input_validation():
    """Test input validation."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Missing required field should fail
    response = client.post("/run", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_basic_run():
    """Test basic agent execution."""
    # Mock the LLM to avoid API calls in tests
    ...


# tests/test_contracts.py
"""Protocol contract tests."""
from fastagentic.testing import validate_mcp_schema, validate_a2a_card
from app import app


def test_mcp_schema():
    """MCP schema is valid."""
    errors = validate_mcp_schema(app)
    assert not errors


def test_a2a_card():
    """A2A Agent Card is valid."""
    errors = validate_a2a_card(app)
    assert not errors
```

## Documentation

Write a clear README:

```markdown
# My Framework Template

Brief description of what this template provides.

## Features

- Feature 1
- Feature 2
- Feature 3

## Prerequisites

- Python 3.10+
- Redis (for durable storage)
- My Framework API key

## Quick Start

\`\`\`bash
# Create project
fastagentic new my-project --template my-framework

# Install dependencies
cd my-project
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
fastagentic run --reload
\`\`\`

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MY_API_KEY` | API key for My Framework | Required |
| `DURABLE_STORE` | Redis/Postgres URL | `redis://localhost:6379` |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run` | POST | Run the agent |
| `/run/stream` | POST | Stream agent output |

## Example Usage

\`\`\`bash
curl -X POST http://localhost:8000/run \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello"}'
\`\`\`

## License

MIT
```

## Testing Locally

Before submitting:

```bash
# 1. Validate structure
python scripts/validate.py templates/community/my-framework

# 2. Build and test
cd templates/community/my-framework
pip install -e ".[dev]"
pytest

# 3. Test template creation
cd /tmp
fastagentic new test-project --template /path/to/templates/community/my-framework
cd test-project
fastagentic run --reload

# 4. Test protocols
curl http://localhost:8000/mcp/schema
curl http://localhost:8000/.well-known/agent.json
```

## Submission Checklist

Before opening a PR:

- [ ] `template.json` has all required fields
- [ ] All required files present
- [ ] Tests pass locally
- [ ] README is complete
- [ ] No hardcoded API keys or secrets
- [ ] `.env.example` documents all environment variables
- [ ] pyproject.toml includes fastagentic dependency
- [ ] Dockerfile builds successfully
- [ ] MCP and A2A schemas validate

## Review Process

1. **Automated Checks**: CI validates structure and runs tests
2. **Code Review**: Maintainer reviews for quality and security
3. **Testing**: Maintainer tests template creation
4. **Verification**: For complex templates, may require demo
5. **Merge**: Added to community templates

### Verification Badge

After review, templates can receive a "verified" badge indicating:
- Code review passed
- Security check passed
- Works with current FastAgentic version
- Documentation is complete

## Maintaining Your Template

After merge:
- You're responsible for updates when FastAgentic releases new versions
- Open issues in the templates repo for bugs
- PRs for updates follow the same process

### Version Updates

When FastAgentic releases a new version:

```bash
# Update min_fastagentic_version in template.json
# Update fastagentic dependency in pyproject.toml
# Test with new version
# Submit PR with version bump
```

## Getting Help

- [GitHub Discussions](https://github.com/fastagentic/fastagentic-templates/discussions) - Questions
- [Discord](https://discord.gg/fastagentic) - Community chat
- [Template Examples](https://github.com/fastagentic/fastagentic-templates/tree/main/templates/official) - Reference implementations
