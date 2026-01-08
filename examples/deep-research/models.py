"""Data models for Deep Research Agent."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request to start a research task."""

    topic: str = Field(..., description="The topic to research")
    depth: str = Field(
        default="medium",
        description="Research depth: shallow, medium, or deep",
    )


class SearchResult(BaseModel):
    """A single search result."""

    title: str
    snippet: str
    source: str


class SearchFindings(BaseModel):
    """Results from the search phase."""

    query: str
    results: list[SearchResult]
    tokens_used: int = 0


class Analysis(BaseModel):
    """Analysis of search findings."""

    key_themes: list[str]
    main_points: list[str]
    gaps: list[str]
    tokens_used: int = 0


class ResearchResult(BaseModel):
    """Final research output."""

    topic: str
    findings: SearchFindings
    analysis: Analysis
    synthesis: str
    total_tokens: int = 0
    total_duration_ms: int = 0


class ResearchResponse(BaseModel):
    """API response for research endpoint."""

    run_id: str
    result: ResearchResult
    status: str = "completed"
