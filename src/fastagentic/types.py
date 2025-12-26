"""Core type definitions for FastAgentic."""

from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")
AsyncCallable = Callable[..., Any]


class RunStatus(str, Enum):
    """Status of a durable run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    NODE_START = "node_start"
    NODE_END = "node_end"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    DONE = "done"
    # Additional types for adapter compatibility
    MESSAGE = "message"
    SOURCE = "source"
    TRACE = "trace"


class StreamEvent(BaseModel):
    """A streaming event from an agent run."""

    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    timestamp: float | None = None


class ToolDefinition(BaseModel):
    """Definition of a tool for MCP/OpenAPI schemas."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    scopes: list[str] = Field(default_factory=list)


class ResourceDefinition(BaseModel):
    """Definition of a resource for MCP/OpenAPI schemas."""

    name: str
    uri: str
    description: str = ""
    mime_type: str = "application/json"
    scopes: list[str] = Field(default_factory=list)
    cache_ttl: int | None = None


class PromptDefinition(BaseModel):
    """Definition of a prompt template for MCP schemas."""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)


class EndpointDefinition(BaseModel):
    """Definition of an agent endpoint."""

    path: str
    name: str
    description: str = ""
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    stream: bool = False
    durable: bool = False
    mcp_tool: str | None = None
    a2a_skill: str | None = None
    scopes: list[str] = Field(default_factory=list)


class MCPCapabilities(BaseModel):
    """MCP server capabilities."""

    tools: bool = True
    resources: bool = True
    prompts: bool = True
    sampling: bool = False


class AgentCard(BaseModel):
    """A2A Agent Card definition."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    skills: list[dict[str, Any]] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=lambda: ["a2a/v0.3"])
    security: dict[str, Any] = Field(default_factory=dict)
