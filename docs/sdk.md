# Python SDK

FastAgentic provides a Python SDK for interacting with deployed agent endpoints.

## Installation

```bash
pip install fastagentic
```

## Quick Start

```python
from fastagentic import FastAgenticClient

# Create client
client = FastAgenticClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
)

# Run an agent endpoint
response = client.run("/chat", input={"message": "Hello!"})
print(response.output)
```

## Configuration

```python
from fastagentic import ClientConfig, FastAgenticClient

config = ClientConfig(
    base_url="http://api.example.com",
    api_key="your-api-key",
    timeout=60.0,           # Request timeout in seconds
    max_retries=3,          # Number of retries on failure
    retry_delay=1.0,        # Initial delay between retries
)

client = FastAgenticClient(config=config)
```

## Synchronous Client

```python
from fastagentic import FastAgenticClient

client = FastAgenticClient(base_url="http://localhost:8000")

# Simple run
response = client.run("/chat", input={"message": "Hello"})

# With options
response = client.run(
    "/chat",
    input={"message": "Hello"},
    timeout=30.0,
    metadata={"user_id": "123"},
)

# Check status
print(response.status)      # RunStatus.COMPLETED
print(response.is_success)  # True
print(response.output)      # {"reply": "Hi there!"}

# Get a specific run
run = client.get_run("run-abc123")
```

## Async Client

```python
import asyncio
from fastagentic import AsyncFastAgenticClient

async def main():
    client = AsyncFastAgenticClient(base_url="http://localhost:8000")

    # Run endpoint
    response = await client.run("/chat", input={"message": "Hello"})
    print(response.output)

    # Close client when done
    await client.close()

asyncio.run(main())
```

### Context Manager

```python
async with AsyncFastAgenticClient(base_url="http://localhost:8000") as client:
    response = await client.run("/chat", input={"message": "Hello"})
```

## Streaming

```python
from fastagentic import FastAgenticClient, StreamEventType

client = FastAgenticClient()

# Stream responses
for event in client.stream("/chat", input={"message": "Tell me a story"}):
    if event.type == StreamEventType.TOKEN:
        print(event.data, end="", flush=True)
    elif event.type == StreamEventType.TOOL_CALL:
        print(f"\n[Tool: {event.data['name']}]")
    elif event.type == StreamEventType.DONE:
        print("\n[Complete]")
```

### Async Streaming

```python
async for event in client.stream("/chat", input={"message": "Hello"}):
    if event.type == StreamEventType.TOKEN:
        print(event.data, end="")
```

## Request/Response Models

### RunRequest

```python
from fastagentic import RunRequest

request = RunRequest(
    endpoint="/chat",
    input={"message": "Hello"},
    stream=False,
    timeout=60.0,
    metadata={"session_id": "abc"},
)
```

### RunResponse

```python
from fastagentic import RunResponse, RunStatus

# Response properties
response.run_id         # Unique run identifier
response.status         # RunStatus enum
response.output         # Response data
response.error          # Error message (if failed)
response.usage          # Token usage stats
response.duration_ms    # Request duration

# Status checks
response.is_complete    # True if finished
response.is_success     # True if completed successfully
```

### StreamEvent

```python
from fastagentic import StreamEvent, StreamEventType

# Event types
StreamEventType.TOKEN       # Text token
StreamEventType.TOOL_CALL   # Tool invocation
StreamEventType.TOOL_RESULT # Tool response
StreamEventType.MESSAGE     # Complete message
StreamEventType.ERROR       # Error occurred
StreamEventType.DONE        # Stream complete
```

## Tool Handling

```python
from fastagentic import ToolCall, ToolResult

# Tool call from response
call = ToolCall(
    id="call-123",
    name="search",
    arguments={"query": "weather"},
)

# Tool result to send back
result = ToolResult(
    call_id="call-123",
    result={"temperature": 72},
)

# Or error result
error_result = ToolResult(
    call_id="call-123",
    error="Search service unavailable",
)
```

## Error Handling

```python
from fastagentic import (
    FastAgenticClient,
    FastAgenticError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    TimeoutError,
    ServerError,
)

client = FastAgenticClient()

try:
    response = client.run("/chat", input={"message": "Hello"})
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except ValidationError as e:
    print(f"Invalid input: {e.errors}")
except TimeoutError as e:
    print(f"Request timed out after {e.timeout}s")
except ServerError as e:
    print(f"Server error: {e}")
except FastAgenticError as e:
    print(f"API error: {e}")
```

## Retry Configuration

The client automatically retries failed requests:

```python
from fastagentic import ClientConfig

config = ClientConfig(
    max_retries=3,          # Max retry attempts
    retry_delay=1.0,        # Initial delay (seconds)
    retry_backoff=2.0,      # Exponential backoff multiplier
    retry_codes=[429, 503], # HTTP codes to retry
)
```

## Custom Headers

```python
from fastagentic import ClientConfig

config = ClientConfig(
    api_key="your-key",
    headers={
        "X-Custom-Header": "value",
        "X-Request-ID": "req-123",
    },
)
```

## Usage Statistics

```python
response = client.run("/chat", input={"message": "Hello"})

if response.usage:
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Cost: ${response.usage.cost:.4f}")
    print(f"Model: {response.usage.model}")
```
