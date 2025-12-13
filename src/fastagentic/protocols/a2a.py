"""A2A (Agent-to-Agent) protocol implementation.

Implements the A2A v0.3 specification for agent interoperability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse

from fastagentic.decorators import get_endpoints

if TYPE_CHECKING:
    from fastagentic.app import App

# A2A Protocol Version
A2A_VERSION = "0.3"


def configure_a2a(
    app: App,
    *,
    enabled: bool = True,
    require_auth: bool = False,
    protocols: list[str] | None = None,
) -> None:
    """Configure A2A protocol routes on an App.

    This function adds A2A-compliant endpoints for agent discovery
    and task delegation.

    Args:
        app: The FastAgentic App instance
        enabled: Whether to enable A2A routes
        require_auth: Whether to require authentication
        protocols: List of supported protocols

    Example:
        from fastagentic import App
        from fastagentic.protocols.a2a import configure_a2a

        app = App(title="My Agent")
        configure_a2a(app)
    """
    if not enabled:
        return

    fastapi = app.fastapi
    supported_protocols = protocols or [f"a2a/v{A2A_VERSION}"]

    # Agent Card is already registered in App._register_a2a_routes
    # This function adds additional A2A-specific endpoints

    # Task creation endpoint
    @fastapi.post("/a2a/tasks")
    async def a2a_create_task(request: Request) -> JSONResponse:
        """Create a new A2A task."""
        try:
            body = await request.json()
            skill_name = body.get("skill")
            task_input = body.get("input", {})

            # Find the endpoint for this skill
            endpoints = get_endpoints()
            target_endpoint = None
            target_func = None

            for path, (defn, func) in endpoints.items():
                if defn.a2a_skill == skill_name:
                    target_endpoint = defn
                    target_func = func
                    break

            if not target_endpoint:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": f"Skill '{skill_name}' not found",
                        "available_skills": [
                            defn.a2a_skill
                            for defn, _ in endpoints.values()
                            if defn.a2a_skill
                        ],
                    },
                )

            # TODO: Create async task and return task ID
            # For now, execute synchronously
            import uuid

            task_id = str(uuid.uuid4())

            return JSONResponse(
                content={
                    "task_id": task_id,
                    "status": "pending",
                    "skill": skill_name,
                }
            )

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    # Task status endpoint
    @fastapi.get("/a2a/tasks/{task_id}")
    async def a2a_get_task(task_id: str) -> JSONResponse:
        """Get the status of an A2A task."""
        # TODO: Implement task storage and retrieval
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "unknown",
                "message": "Task storage not yet implemented",
            }
        )

    # Task cancellation
    @fastapi.delete("/a2a/tasks/{task_id}")
    async def a2a_cancel_task(task_id: str) -> JSONResponse:
        """Cancel an A2A task."""
        # TODO: Implement task cancellation
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "cancelled",
            }
        )

    # Agent registry (for internal agent discovery)
    @fastapi.get("/a2a/agents")
    async def a2a_list_agents() -> JSONResponse:
        """List registered agents in the registry."""
        # TODO: Implement agent registry
        return JSONResponse(
            content={
                "agents": [
                    {
                        "name": app.config.title,
                        "url": "/",
                        "version": app.config.version,
                    }
                ]
            }
        )

    # Ping endpoint for health checks
    @fastapi.get("/a2a/ping")
    async def a2a_ping() -> JSONResponse:
        """A2A ping endpoint."""
        return JSONResponse(
            content={
                "status": "ok",
                "version": A2A_VERSION,
                "agent": app.config.title,
            }
        )
