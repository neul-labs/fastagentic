"""Pydantic models for chat request and response.

These models define the API schema and are automatically exposed
in OpenAPI, MCP, and A2A interfaces.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ...,
        description="The user's message to the chat agent",
        min_length=1,
        max_length=10000,
        examples=["Hello!", "What time is it?", "Calculate 25 * 4"],
    )

    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for multi-turn conversations",
    )

    stream: bool = Field(
        default=True,
        description="Whether to stream the response",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(
        ...,
        description="The agent's response message",
    )

    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID for continuing the conversation",
    )

    tool_calls: list[dict] | None = Field(
        default=None,
        description="List of tool calls made during the response",
    )

    usage: dict | None = Field(
        default=None,
        description="Token usage statistics",
    )
