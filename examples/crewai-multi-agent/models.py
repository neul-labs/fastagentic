"""API models for CrewAI multi-agent system."""

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request model for content analysis."""

    topic: str = Field(..., description="Topic to research and analyze")


class AnalysisResponse(BaseModel):
    """Response model for content analysis."""

    summary: str = Field(..., description="Final written summary")
    research: str = Field(default="", description="Research findings")
    analysis: str = Field(default="", description="Analysis insights")
