"""A2A (Agent-to-Agent) protocol implementation.

Implements the A2A v0.3 specification for agent interoperability.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse

from fastagentic.decorators import get_endpoints

if TYPE_CHECKING:
    from fastagentic.app import App

# A2A Protocol Version
A2A_VERSION = "0.3"


class TaskStatus(str, Enum):
    """A2A task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class A2ATask:
    """Represents an A2A task."""

    task_id: str
    skill: str
    input: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    cancelled: bool = False


class InMemoryTaskStore:
    """In-memory task store for A2A tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, A2ATask] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def create(self, task: A2ATask) -> None:
        """Create a new task."""
        self._tasks[task.task_id] = task
        self._locks[task.task_id] = asyncio.Lock()

    async def get(self, task_id: str) -> A2ATask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    async def update(self, task: A2ATask) -> None:
        """Update a task."""
        task.updated_at = datetime.utcnow()
        self._tasks[task.task_id] = task

    async def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        async with self._locks.get(task_id, asyncio.Lock()):
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.status = TaskStatus.CANCELLED
                task.cancelled = True
                task.updated_at = datetime.utcnow()
                return True
            return False

    async def list_by_skill(self, skill: str) -> list[A2ATask]:
        """List tasks by skill."""
        return [t for t in self._tasks.values() if t.skill == skill]

    async def delete(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            if task_id in self._locks:
                del self._locks[task_id]
            return True
        return False


# Global task store
_task_store = InMemoryTaskStore()


def get_task_store() -> InMemoryTaskStore:
    """Get the global task store."""
    return _task_store


def configure_a2a(
    app: App,
    *,
    enabled: bool = True,
    _require_auth: bool = False,
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

    # Task creation endpoint
    @fastapi.post("/a2a/tasks")
    async def a2a_create_task(request: Request) -> JSONResponse:
        """Create a new A2A task."""
        try:
            body = await request.json()
            skill_name = body.get("skill")
            task_input = body.get("input", {})

            if not skill_name:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing required field: skill"},
                )

            # Find the endpoint for this skill
            endpoints = get_endpoints()
            target_endpoint = None
            target_func = None

            for _path, (defn, func) in endpoints.items():
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
                            defn.a2a_skill for defn, _ in endpoints.values() if defn.a2a_skill
                        ],
                    },
                )

            # Create async task
            task_id = str(uuid.uuid4())
            task = A2ATask(
                task_id=task_id,
                skill=skill_name,
                input=task_input,
            )

            await _task_store.create(task)

            # Start task execution in background
            asyncio.create_task(_execute_task(task_id, target_func, task_input))

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
        task = await _task_store.get(task_id)

        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task '{task_id}' not found"},
            )

        response = {
            "task_id": task_id,
            "skill": task.skill,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

        if task.status == TaskStatus.COMPLETED:
            response["result"] = task.result
        elif task.status == TaskStatus.FAILED:
            response["error"] = task.error

        return JSONResponse(content=response)

    # Task cancellation
    @fastapi.delete("/a2a/tasks/{task_id}")
    async def a2a_cancel_task(task_id: str) -> JSONResponse:
        """Cancel an A2A task."""
        task = await _task_store.get(task_id)

        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task '{task_id}' not found"},
            )

        cancelled = await _task_store.cancel(task_id)

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "cancelled" if cancelled else task.status.value,
            }
        )

    # Agent registry (for internal agent discovery)
    @fastapi.get("/a2a/agents")
    async def a2a_list_agents() -> JSONResponse:
        """List registered agents in the registry."""
        return JSONResponse(
            content={
                "agents": [
                    {
                        "name": app.config.title,
                        "url": "http://localhost:8000",  # Default URL since host is not configured
                        "version": app.config.version,
                        "protocols": supported_protocols,
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


async def _execute_task(
    task_id: str,
    func: Any,
    input: dict[str, Any],
) -> None:
    """Execute an A2A task in the background."""
    task = await _task_store.get(task_id)
    if not task:
        return

    # Update status to running
    task.status = TaskStatus.RUNNING
    await _task_store.update(task)

    try:
        # Call the endpoint function
        if asyncio.iscoroutinefunction(func):
            result = await func(**input)
        else:
            result = func(**input)

        task.result = result
        task.status = TaskStatus.COMPLETED
    except Exception as e:
        task.error = str(e)
        task.status = TaskStatus.FAILED
    finally:
        await _task_store.update(task)
