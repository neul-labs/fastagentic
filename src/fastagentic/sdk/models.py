"""SDK data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """Status of a run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StreamEventType(str, Enum):
    """Types of stream events."""

    START = "start"
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE = "message"
    ERROR = "error"
    END = "end"


@dataclass
class ToolCall:
    """A tool call made during a run.

    Attributes:
        id: Unique tool call ID
        name: Tool name
        arguments: Tool arguments
        timestamp: When the call was made
    """

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call-{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Create from dictionary."""
        return cls(
            id=data.get("id", f"call-{uuid.uuid4().hex[:8]}"),
            name=data["name"],
            arguments=data.get("arguments", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ToolResult:
    """Result of a tool call.

    Attributes:
        call_id: ID of the tool call
        result: Tool result data
        error: Error message if failed
        duration_ms: Execution duration in milliseconds
    """

    call_id: str
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None

    @property
    def is_error(self) -> bool:
        """Check if result is an error."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        """Create from dictionary."""
        return cls(
            call_id=data["call_id"],
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
        )


@dataclass
class StreamEvent:
    """An event from a streaming run.

    Attributes:
        type: Event type
        data: Event data
        timestamp: Event timestamp
    """

    type: StreamEventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamEvent:
        """Create from dictionary."""
        return cls(
            type=StreamEventType(data["type"]),
            data=data.get("data"),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: Message role (user, assistant, system, tool)
        content: Message content
        tool_calls: Tool calls in this message
        tool_call_id: ID if this is a tool result message
        name: Name for tool messages
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class RunRequest:
    """Request to start a run.

    Attributes:
        endpoint: Endpoint path or name
        input: Input data
        stream: Whether to stream the response
        timeout: Request timeout in seconds
        metadata: Additional metadata
    """

    endpoint: str
    input: dict[str, Any]
    stream: bool = False
    timeout: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "endpoint": self.endpoint,
            "input": self.input,
            "stream": self.stream,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


@dataclass
class UsageStats:
    """Token and cost usage statistics.

    Attributes:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens used
        cost: Estimated cost in USD
        model: Model used
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageStats:
        """Create from dictionary."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cost=data.get("cost", 0.0),
            model=data.get("model"),
        )


@dataclass
class RunResponse:
    """Response from a run.

    Attributes:
        run_id: Unique run ID
        status: Run status
        output: Run output
        messages: Conversation messages
        tool_calls: Tool calls made
        usage: Token and cost usage
        error: Error message if failed
        started_at: When the run started
        completed_at: When the run completed
        duration_ms: Total duration in milliseconds
        metadata: Additional metadata
    """

    run_id: str
    status: RunStatus
    output: Any = None
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if run is complete."""
        return self.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMEOUT,
        )

    @property
    def is_success(self) -> bool:
        """Check if run completed successfully."""
        return self.status == RunStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "output": self.output,
            "messages": [m.to_dict() for m in self.messages],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage.to_dict(),
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResponse:
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            output=data.get("output"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            usage=UsageStats.from_dict(data.get("usage", {})),
            error=data.get("error"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata", {}),
        )
