"""FastAgentic Python SDK.

Provides a client for interacting with FastAgentic services.
"""

from fastagentic.sdk.client import (
    FastAgenticClient,
    AsyncFastAgenticClient,
    ClientConfig,
)
from fastagentic.sdk.models import (
    RunRequest,
    RunResponse,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)
from fastagentic.sdk.exceptions import (
    FastAgenticError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    TimeoutError,
    ServerError,
)

__all__ = [
    # Client
    "FastAgenticClient",
    "AsyncFastAgenticClient",
    "ClientConfig",
    # Models
    "RunRequest",
    "RunResponse",
    "RunStatus",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "ToolResult",
    # Exceptions
    "FastAgenticError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "TimeoutError",
    "ServerError",
]
