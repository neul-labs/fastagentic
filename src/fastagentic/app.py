"""FastAgentic App - The core application container."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Sequence

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import structlog

from fastagentic.context import AgentContext, RunContext, UserInfo
from fastagentic.decorators import get_endpoints, get_prompts, get_resources, get_tools
from fastagentic.types import RunStatus, StreamEvent, StreamEventType

if TYPE_CHECKING:
    from fastagentic.adapters.base import BaseAdapter
    from fastagentic.hooks.base import Hook
    from fastagentic.memory import MemoryProvider

logger = structlog.get_logger()


class AppConfig(BaseModel):
    """Configuration for the FastAgentic App."""

    title: str = "FastAgentic App"
    version: str = "1.0.0"
    description: str = ""

    # Auth
    oidc_issuer: str | None = None
    oidc_audience: str | None = None

    # Telemetry
    telemetry: bool = False

    # Durability
    durable_store: str | None = None

    # MCP
    mcp_enabled: bool = True
    mcp_path_prefix: str = "/mcp"

    # A2A
    a2a_enabled: bool = True


class App:
    """The main FastAgentic application container.

    App manages the lifecycle of your agent application, including:
    - ASGI server configuration
    - MCP and A2A protocol exposure
    - Hook registration and execution
    - Memory provider configuration
    - Durable run management

    Example:
        from fastagentic import App

        app = App(
            title="Support Triage",
            version="1.0.0",
            oidc_issuer="https://auth.example.com",
            durable_store="redis://localhost:6379",
        )
    """

    def __init__(
        self,
        title: str = "FastAgentic App",
        version: str = "1.0.0",
        description: str = "",
        *,
        oidc_issuer: str | None = None,
        oidc_audience: str | None = None,
        telemetry: bool = False,
        durable_store: str | None = None,
        mcp_enabled: bool = True,
        mcp_path_prefix: str = "/mcp",
        a2a_enabled: bool = True,
        hooks: Sequence[Hook] | None = None,
        memory: MemoryProvider | None = None,
        session_memory: MemoryProvider | None = None,
    ) -> None:
        self.config = AppConfig(
            title=title,
            version=version,
            description=description,
            oidc_issuer=oidc_issuer,
            oidc_audience=oidc_audience,
            telemetry=telemetry,
            durable_store=durable_store,
            mcp_enabled=mcp_enabled,
            mcp_path_prefix=mcp_path_prefix,
            a2a_enabled=a2a_enabled,
        )

        self._hooks: list[Hook] = list(hooks) if hooks else []
        self._memory = memory
        self._session_memory = session_memory
        self._durable_store: Any = None  # Will be initialized on startup

        # Create the FastAPI app with lifespan
        self._fastapi = FastAPI(
            title=title,
            version=version,
            description=description,
            lifespan=self._lifespan,
        )

        # Register built-in routes
        self._register_health_routes()
        self._register_mcp_routes()
        self._register_a2a_routes()

    @property
    def fastapi(self) -> FastAPI:
        """Get the underlying FastAPI application."""
        return self._fastapi

    @property
    def memory(self) -> MemoryProvider | None:
        """Get the configured memory provider."""
        return self._memory

    @property
    def session_memory(self) -> MemoryProvider | None:
        """Get the configured session memory provider."""
        return self._session_memory

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifespan events."""
        logger.info("Starting FastAgentic application", title=self.config.title)

        # Initialize durable store if configured
        if self.config.durable_store:
            await self._init_durable_store()

        # Initialize hooks
        for hook in self._hooks:
            if hasattr(hook, "on_startup"):
                await hook.on_startup(self)

        # Register decorated endpoints
        self._register_agent_endpoints()

        yield

        # Cleanup
        logger.info("Shutting down FastAgentic application")
        for hook in self._hooks:
            if hasattr(hook, "on_shutdown"):
                await hook.on_shutdown(self)

        if self._durable_store:
            await self._close_durable_store()

    async def _init_durable_store(self) -> None:
        """Initialize the durable store connection."""
        store_url = self.config.durable_store
        if not store_url:
            return

        if store_url.startswith("redis://"):
            try:
                import redis.asyncio as redis

                self._durable_store = redis.from_url(store_url)
                logger.info("Connected to Redis durable store")
            except ImportError:
                logger.warning("Redis not installed, durable runs disabled")
        elif store_url.startswith("postgres://"):
            logger.warning("PostgreSQL durable store not yet implemented")

    async def _close_durable_store(self) -> None:
        """Close the durable store connection."""
        if self._durable_store:
            await self._durable_store.close()

    def _register_health_routes(self) -> None:
        """Register health check endpoints."""

        @self._fastapi.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "version": self.config.version,
                "title": self.config.title,
            }

        @self._fastapi.get("/ready")
        async def ready() -> dict[str, Any]:
            # Check dependencies
            checks = {"app": True}
            if self.config.durable_store:
                checks["durable_store"] = self._durable_store is not None
            return {"ready": all(checks.values()), "checks": checks}

    def _register_mcp_routes(self) -> None:
        """Register MCP protocol routes."""
        if not self.config.mcp_enabled:
            return

        prefix = self.config.mcp_path_prefix

        @self._fastapi.get(f"{prefix}/schema")
        async def mcp_schema() -> dict[str, Any]:
            """Return MCP schema with tools, resources, and prompts."""
            tools = get_tools()
            resources = get_resources()
            prompts = get_prompts()

            return {
                "protocolVersion": "2025-11-25",
                "capabilities": {
                    "tools": len(tools) > 0,
                    "resources": len(resources) > 0,
                    "prompts": len(prompts) > 0,
                },
                "tools": [
                    {
                        "name": defn.name,
                        "description": defn.description,
                        "inputSchema": defn.parameters,
                    }
                    for defn, _ in tools.values()
                ],
                "resources": [
                    {
                        "name": defn.name,
                        "uri": defn.uri,
                        "description": defn.description,
                        "mimeType": defn.mime_type,
                    }
                    for defn, _ in resources.values()
                ],
                "prompts": [
                    {
                        "name": defn.name,
                        "description": defn.description,
                        "arguments": defn.arguments,
                    }
                    for defn, _ in prompts.values()
                ],
            }

        @self._fastapi.get(f"{prefix}/health")
        async def mcp_health() -> dict[str, str]:
            return {"status": "ok"}

    def _register_a2a_routes(self) -> None:
        """Register A2A protocol routes."""
        if not self.config.a2a_enabled:
            return

        @self._fastapi.get("/.well-known/agent.json")
        async def agent_card() -> dict[str, Any]:
            """Return the A2A Agent Card."""
            endpoints = get_endpoints()

            skills = []
            for path, (defn, _) in endpoints.items():
                if defn.a2a_skill:
                    skills.append(
                        {
                            "name": defn.a2a_skill,
                            "description": defn.description,
                            "endpoint": path,
                            "inputSchema": (
                                defn.input_model.model_json_schema()
                                if defn.input_model
                                else {}
                            ),
                            "outputSchema": (
                                defn.output_model.model_json_schema()
                                if defn.output_model
                                else {}
                            ),
                        }
                    )

            return {
                "name": self.config.title,
                "description": self.config.description,
                "version": self.config.version,
                "protocols": ["a2a/v0.3"],
                "skills": skills,
                "security": (
                    {"type": "oidc", "issuer": self.config.oidc_issuer}
                    if self.config.oidc_issuer
                    else {}
                ),
            }

    def _register_agent_endpoints(self) -> None:
        """Register all decorated agent endpoints as FastAPI routes."""
        endpoints = get_endpoints()

        for path, (defn, func) in endpoints.items():
            self._create_endpoint_route(path, defn, func)

    def _create_endpoint_route(
        self,
        path: str,
        defn: Any,
        func: Callable[..., Any],
    ) -> None:
        """Create a FastAPI route for an agent endpoint."""
        runnable = getattr(func, "_fastagentic_runnable", None)

        if defn.stream:
            # Streaming endpoint returns SSE
            @self._fastapi.post(path, name=defn.name)
            async def stream_endpoint(
                request: Request,
                body: defn.input_model if defn.input_model else dict,  # type: ignore
                _func: Callable[..., Any] = func,
                _runnable: Any = runnable,
                _defn: Any = defn,
            ) -> EventSourceResponse:
                run_id = str(uuid.uuid4())
                ctx = self._create_context(run_id, path, request)

                async def event_generator() -> AsyncIterator[dict[str, Any]]:
                    try:
                        if _runnable and hasattr(_runnable, "stream"):
                            async for event in _runnable.stream(body, ctx):
                                yield {"event": event.type.value, "data": event.model_dump_json()}
                        else:
                            # Run the function directly
                            result = await _func(body, ctx=ctx)
                            yield {
                                "event": "done",
                                "data": (
                                    result.model_dump_json()
                                    if hasattr(result, "model_dump_json")
                                    else str(result)
                                ),
                            }
                    except Exception as e:
                        logger.exception("Error in stream endpoint", error=str(e))
                        yield {"event": "error", "data": str(e)}

                return EventSourceResponse(event_generator())
        else:
            # Non-streaming endpoint returns JSON
            @self._fastapi.post(path, name=defn.name)
            async def invoke_endpoint(
                request: Request,
                body: defn.input_model if defn.input_model else dict,  # type: ignore
                _func: Callable[..., Any] = func,
                _runnable: Any = runnable,
                _defn: Any = defn,
            ) -> Response:
                run_id = str(uuid.uuid4())
                ctx = self._create_context(run_id, path, request)

                try:
                    if _runnable and hasattr(_runnable, "invoke"):
                        result = await _runnable.invoke(body, ctx)
                    else:
                        result = await _func(body, ctx=ctx)

                    if hasattr(result, "model_dump"):
                        return JSONResponse(content=result.model_dump())
                    return JSONResponse(content={"result": result})
                except Exception as e:
                    logger.exception("Error in endpoint", error=str(e))
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(e)},
                    )

    def _create_context(
        self,
        run_id: str,
        endpoint: str,
        request: Request,
    ) -> AgentContext:
        """Create an AgentContext for a request."""
        # TODO: Extract user from auth headers
        user = None

        run_ctx = RunContext(
            run_id=run_id,
            endpoint=endpoint,
            user=user,
        )

        return AgentContext(
            run=run_ctx,
            app=self,
            request=request,
        )

    def add_hook(self, hook: Hook) -> None:
        """Add a hook to the application."""
        self._hooks.append(hook)

    # FastAPI method proxies for convenience
    def get(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Proxy to FastAPI.get()."""
        return self._fastapi.get(path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Proxy to FastAPI.post()."""
        return self._fastapi.post(path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Proxy to FastAPI.put()."""
        return self._fastapi.put(path, **kwargs)

    def delete(
        self, path: str, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Proxy to FastAPI.delete()."""
        return self._fastapi.delete(path, **kwargs)
