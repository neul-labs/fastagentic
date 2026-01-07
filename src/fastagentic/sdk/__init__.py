"""FastAgentic Python SDK.

Provides a client for interacting with FastAgentic services.
"""

from fastagentic.sdk.client import AsyncFastAgenticClient, ClientConfig, FastAgenticClient
from fastagentic.sdk.exceptions import (
    AuthenticationError,
    FastAgenticError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
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
