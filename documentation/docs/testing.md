# Testing Guide

How to test your FastAgentic agents effectively.

## Quick Start

```bash
# Run tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=. --cov-report=html
```

## Testing Approaches

### 1. Agent CLI Testing (Manual)

The fastest way to test during development:

```bash
# Start your server
uv run fastagentic run --reload

# In another terminal, test interactively
uv run fastagentic agent chat --endpoint /chat
```

### 2. Unit Tests

Test individual components:

```python
# tests/test_tools.py
import pytest
from tools import calculate, search_kb

@pytest.mark.asyncio
async def test_calculate():
    result = await calculate("2 + 2")
    assert result == "4"

@pytest.mark.asyncio
async def test_search_kb():
    results = await search_kb("test query")
    assert len(results) > 0
```

### 3. Integration Tests

Test the full endpoint:

```python
# tests/test_endpoints.py
import pytest
from httpx import AsyncClient
from app import app

@pytest.fixture
async def client():
    async with AsyncClient(
        app=app.fastapi,
        base_url="http://test"
    ) as client:
        yield client

@pytest.mark.asyncio
async def test_chat_endpoint(client):
    response = await client.post(
        "/chat",
        json={"message": "Hello"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
```

### 4. Contract Tests

Verify schema parity:

```bash
uv run fastagentic test contract
```

## Mocking LLM Calls

### Option 1: Mock the adapter

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_chat_with_mock():
    mock_response = {"response": "Mocked response"}

    with patch.object(
        PydanticAIAdapter,
        "invoke",
        new_callable=AsyncMock,
        return_value=mock_response
    ):
        # Your test code
        pass
```

### Option 2: Use a test model

```python
# In test configuration
from pydantic_ai import Agent

test_agent = Agent(
    model="test",  # Uses mock responses
    system_prompt="Test agent",
)
```

### Option 3: Environment-based

```python
# conftest.py
import os

@pytest.fixture(autouse=True)
def mock_llm():
    os.environ["MOCK_LLM"] = "true"
    yield
    del os.environ["MOCK_LLM"]

# In your agent
import os
if os.environ.get("MOCK_LLM"):
    # Return mock response
    pass
```

## Testing Streaming

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_streaming_response(client):
    async with client.stream(
        "POST",
        "/chat",
        json={"message": "Hello", "stream": True}
    ) as response:
        assert response.status_code == 200

        chunks = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line[6:])

        assert len(chunks) > 0
```

## Testing Tools

```python
@pytest.mark.asyncio
async def test_tool_called():
    """Verify a tool is called during agent execution."""

    tool_called = False
    original_tool = my_tool

    async def tracked_tool(*args, **kwargs):
        nonlocal tool_called
        tool_called = True
        return await original_tool(*args, **kwargs)

    with patch("tools.my_tool", tracked_tool):
        # Run agent
        pass

    assert tool_called
```

## Testing Policies

```python
@pytest.mark.asyncio
async def test_rate_limit():
    """Test rate limiting works."""
    # Make many requests quickly
    for i in range(100):
        response = await client.post("/chat", json={"message": "test"})

    # Should eventually get rate limited
    assert response.status_code == 429

@pytest.mark.asyncio
async def test_unauthorized():
    """Test auth is required."""
    response = await client.post(
        "/admin/analyze",
        json={"message": "test"}
    )
    assert response.status_code == 401
```

## Test Configuration

### pytest.ini / pyproject.toml

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-ra -q"
markers = [
    "slow: marks tests as slow",
    "integration: marks as integration test",
]
```

### conftest.py

```python
import pytest
from httpx import AsyncClient
from app import app

@pytest.fixture
async def client():
    """HTTP client for testing."""
    async with AsyncClient(
        app=app.fastapi,
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set a mock API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
```

## CI/CD Testing

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Run tests
        run: uv run pytest tests/ -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Best Practices

1. **Mock LLM calls in CI** - Don't make real API calls in automated tests
2. **Use fixtures** - Share setup code across tests
3. **Test edge cases** - Empty input, long input, special characters
4. **Test error handling** - Verify errors are handled gracefully
5. **Test streaming** - If you use streaming, test it works
6. **Run contract tests** - Ensure schema consistency

## Example Test Suite

```bash
tests/
├── conftest.py           # Shared fixtures
├── test_app.py           # App configuration tests
├── test_endpoints.py     # API endpoint tests
├── test_tools.py         # Tool function tests
├── test_streaming.py     # Streaming tests
├── test_auth.py          # Authentication tests
└── test_contract.py      # Schema parity tests
```

## Debugging Tests

```bash
# Run with verbose output
uv run pytest tests/ -v -s

# Run single test
uv run pytest tests/test_app.py::test_health -v

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Show locals on failure
uv run pytest tests/ -l
```
