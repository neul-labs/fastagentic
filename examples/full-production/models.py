"""API models for production application."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None
    stream: bool = True


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str | None = None
    usage: dict | None = None
