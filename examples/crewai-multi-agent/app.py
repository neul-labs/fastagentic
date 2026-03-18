"""FastAgentic Multi-Agent Application with CrewAI."""

from fastagentic import App, agent_endpoint, resource
from fastagentic.adapters.crewai import CrewAIAdapter

from crew import analysis_crew
from models import AnalysisRequest, AnalysisResponse

app = App(
    title="CrewAI Multi-Agent System",
    version="1.0.0",
    description="Multi-agent collaboration powered by CrewAI",
)


@resource(name="crew-info", uri="info")
async def get_crew_info() -> dict:
    """Return crew information."""
    return {
        "name": "Content Analysis Crew",
        "agents": ["Researcher", "Analyst", "Writer"],
        "process": "sequential",
    }


@agent_endpoint(
    path="/analyze",
    runnable=CrewAIAdapter(analysis_crew),
    input_model=AnalysisRequest,
    output_model=AnalysisResponse,
    stream=True,
    mcp_tool="analyze_content",
    a2a_skill="content-analysis",
)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze a topic using the multi-agent crew."""
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
