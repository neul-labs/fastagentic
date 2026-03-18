"""API models for the research workflow."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""

    query: str = Field(
        ...,
        description="The research query or question",
        min_length=1,
        max_length=1000,
        examples=[
            "What are the latest developments in AI?",
            "How does climate change affect agriculture?",
        ],
    )

    max_iterations: int = Field(
        default=3,
        description="Maximum research iterations",
        ge=1,
        le=5,
    )


class ResearchResponse(BaseModel):
    """Response model for research endpoint."""

    query: str = Field(description="The original query")
    summary: str = Field(description="Final research summary")
    findings: list[str] = Field(description="Research findings")
    analysis: str = Field(description="Analysis of findings")
    iterations: int = Field(description="Number of research iterations")
