"""FastAgentic Deep Research Application.

Exposes the research agent as REST, MCP, and A2A endpoints
with streaming support and checkpointing.

Run with:
    fastagentic run
    # or
    uvicorn app:app.fastapi --reload
"""

from fastagentic import App, agent_endpoint, resource, tool
from fastagentic.adapters.pydanticai import PydanticAIAdapter

from agent import research_agent
from models import ResearchRequest, ResearchResponse, ResearchResult


# Create the FastAgentic app
app = App(
    title="Deep Research Agent",
    version="1.0.0",
    description="Production-safe deep research agent with checkpointing",
    # Enable checkpointing for resume capability
    # durable_store="redis://localhost:6379",  # Uncomment for Redis
)


# Expose agent info as MCP resource
@resource(
    name="agent-info",
    uri="info",
    description="Get information about the research agent",
)
async def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "name": "Deep Research Agent",
        "version": "1.0.0",
        "capabilities": ["research", "analysis", "synthesis", "checkpointing"],
        "model": research_agent.model,
        "phases": ["search", "analyze", "synthesize"],
    }


# Expose research tool for MCP
@tool(
    name="research",
    description="Conduct deep research on a topic with automatic checkpointing",
)
async def research_tool(topic: str) -> str:
    """Research a topic and return synthesis."""
    from fastagentic import FileCheckpointStore
    from fastagentic.checkpoint import CheckpointManager
    from fastagentic.runner import StepTracker
    import uuid

    store = FileCheckpointStore(".checkpoints")
    manager = CheckpointManager(store)
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    tracker = StepTracker(run_id, manager)

    result = await research_agent.run(topic, tracker)
    await manager.mark_completed(run_id)

    return result.synthesis


# Main research endpoint
@agent_endpoint(
    path="/research",
    runnable=PydanticAIAdapter(research_agent._search_agent),  # Use for basic streaming
    input_model=ResearchRequest,
    output_model=ResearchResponse,
    stream=True,
    mcp_tool="deep_research",
    a2a_skill="deep-research",
    description="Conduct deep research with checkpointing and streaming",
)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run deep research on a topic.

    This endpoint supports:
    - Streaming progress updates via SSE
    - Checkpoint-based resume
    - Full observability
    """
    from fastagentic import FileCheckpointStore
    from fastagentic.checkpoint import CheckpointManager
    from fastagentic.runner import StepTracker
    import uuid

    store = FileCheckpointStore(".checkpoints")
    manager = CheckpointManager(store)
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    tracker = StepTracker(run_id, manager)

    result = await research_agent.run(request.topic, tracker)
    await manager.mark_completed(run_id)

    return ResearchResponse(
        run_id=run_id,
        result=result,
    )


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "deep-research"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
