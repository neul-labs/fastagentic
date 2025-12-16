"""FastAgentic Research Workflow with LangGraph.

A stateful research workflow deployed with FastAgentic.
"""

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.langgraph import LangGraphAdapter

from workflow import research_workflow
from models import ResearchRequest, ResearchResponse

# Create the FastAgentic app
app = App(
    title="LangGraph Research Workflow",
    version="1.0.0",
    description="A stateful research workflow powered by LangGraph",
    # Enable durability for workflow persistence
    # durable_store="redis://localhost:6379",
)


# Expose the workflow status as a resource
@resource(
    name="workflow-info",
    uri="info",
    description="Get workflow metadata",
)
async def get_workflow_info() -> dict:
    """Return workflow information."""
    return {
        "name": "Research Workflow",
        "version": "1.0.0",
        "nodes": ["plan", "research", "analyze", "summarize"],
        "description": "Multi-step research workflow with iterative research",
    }


# Research endpoint
@agent_endpoint(
    path="/research",
    runnable=LangGraphAdapter(research_workflow),
    input_model=ResearchRequest,
    output_model=ResearchResponse,
    stream=True,
    durable=False,  # Set True with Redis for persistence
    mcp_tool="research",
    a2a_skill="deep-research",
    description="Conduct research on a topic",
)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Research a topic using the multi-step workflow."""
    pass  # Handler provided by adapter


# Tool to check workflow status (for long-running workflows)
@tool(
    name="get_research_status",
    description="Get the status of a research workflow run",
)
async def get_research_status(run_id: str) -> dict:
    """Get status of a running research workflow.

    Args:
        run_id: The workflow run ID

    Returns:
        Current status of the workflow
    """
    # In production, this would query the durable store
    return {
        "run_id": run_id,
        "status": "unknown",
        "message": "Enable durable_store for run tracking",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
