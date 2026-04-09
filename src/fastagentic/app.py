"""FastAgentic App - The core application container."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from fastagentic.context import AgentContext, RunContext
from fastagentic.decorators import get_endpoints, get_prompts, get_resources, get_tools

if TYPE_CHECKING:
    from fastagentic.context import UserInfo
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

    # Concurrency
    max_concurrent: int | None = None

    # Cluster identification
    instance_id: str | None = None

    # Connection pools
    redis_pool_size: int = 10
    redis_pool_timeout: float = 5.0
    db_pool_size: int = 5
    db_max_overflow: int = 10


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
        max_concurrent: int | None = None,
        instance_id: str | None = None,
        redis_pool_size: int | None = None,
        db_pool_size: int | None = None,
    ) -> None:
        # Get pool config from environment if not specified
        _redis_pool = redis_pool_size or int(os.environ.get("FASTAGENTIC_REDIS_POOL_SIZE", "10"))
        _db_pool = db_pool_size or int(os.environ.get("FASTAGENTIC_DB_POOL_SIZE", "5"))
        _db_overflow = int(os.environ.get("FASTAGENTIC_DB_MAX_OVERFLOW", "10"))
        _max_concurrent = max_concurrent or (
            int(mc) if (mc := os.environ.get("FASTAGENTIC_MAX_CONCURRENT")) else None
        )
        _instance_id = instance_id or os.environ.get("FASTAGENTIC_INSTANCE_ID")

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
            max_concurrent=_max_concurrent,
            instance_id=_instance_id,
            redis_pool_size=_redis_pool,
            db_pool_size=_db_pool,
            db_max_overflow=_db_overflow,
        )

        self._hooks: list[Hook] = list(hooks) if hooks else []
        self._memory = memory
        self._session_memory = session_memory
        self._durable_store: Any = None  # Will be initialized on startup
        self._instance_id = _instance_id or self._generate_instance_id()

        # Create the FastAPI app with lifespan
        self._fastapi = FastAPI(
            title=title,
            version=version,
            description=description,
            lifespan=self._lifespan,
        )

        # Add middleware
        self._configure_middleware()

        # Register built-in routes
        self._register_health_routes()
        self._register_mcp_routes()
        self._register_a2a_routes()

    def _generate_instance_id(self) -> str:
        """Generate a unique instance ID."""
        import socket

        hostname = socket.gethostname()
        pid = os.getpid()
        return f"{hostname}-{pid}"

    def _configure_middleware(self) -> None:
        """Configure production middleware."""
        # Add concurrency limit middleware if configured
        if self.config.max_concurrent:
            from fastagentic.server.middleware import ConcurrencyLimitMiddleware

            self._fastapi.add_middleware(
                ConcurrencyLimitMiddleware,
                max_concurrent=self.config.max_concurrent,
            )
            logger.info(
                "Concurrency limit enabled",
                max_concurrent=self.config.max_concurrent,
            )

        # Add instance metrics middleware
        from fastagentic.server.middleware import InstanceMetricsMiddleware

        self._fastapi.add_middleware(
            InstanceMetricsMiddleware,
            instance_id=self._instance_id,
        )

    @property
    def instance_id(self) -> str:
        """Get the instance ID for this app instance."""
        return self._instance_id

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
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
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
        """Initialize the durable store connection with pooling."""
        store_url = self.config.durable_store
        if not store_url:
            return

        if store_url.startswith("redis://"):
            try:
                import redis.asyncio as redis
                from redis.asyncio.connection import ConnectionPool

                # Create connection pool with configured settings
                pool = ConnectionPool.from_url(
                    store_url,
                    max_connections=self.config.redis_pool_size,
                    socket_timeout=self.config.redis_pool_timeout,
                    socket_connect_timeout=self.config.redis_pool_timeout,
                    retry_on_timeout=True,
                )
                self._durable_store = redis.Redis(connection_pool=pool)
                logger.info(
                    "Connected to Redis durable store",
                    pool_size=self.config.redis_pool_size,
                    instance_id=self._instance_id,
                )
            except ImportError:
                logger.warning("Redis not installed, durable runs disabled")
        elif store_url.startswith("postgres://") or store_url.startswith("postgresql://"):
            try:
                from sqlalchemy.ext.asyncio import create_async_engine

                # Create async engine with pool configuration
                self._db_engine = create_async_engine(
                    store_url.replace("postgres://", "postgresql+asyncpg://").replace(
                        "postgresql://", "postgresql+asyncpg://"
                    ),
                    pool_size=self.config.db_pool_size,
                    max_overflow=self.config.db_max_overflow,
                    pool_timeout=30.0,
                    pool_recycle=1800,
                    pool_pre_ping=True,
                )
                logger.info(
                    "Connected to PostgreSQL durable store",
                    pool_size=self.config.db_pool_size,
                    max_overflow=self.config.db_max_overflow,
                    instance_id=self._instance_id,
                )
            except ImportError:
                logger.warning("SQLAlchemy/asyncpg not installed, PostgreSQL disabled")

    async def _close_durable_store(self) -> None:
        """Close the durable store connection."""
        if self._durable_store:
            await self._durable_store.close()
        if hasattr(self, "_db_engine") and self._db_engine:
            await self._db_engine.dispose()

    def _register_health_routes(self) -> None:
        """Register health check endpoints."""

        @self._fastapi.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "version": self.config.version,
                "title": self.config.title,
                "instance_id": self._instance_id,
            }

        @self._fastapi.get("/ready")
        async def ready() -> dict[str, Any]:
            # Check dependencies
            checks = {"app": True}
            if self.config.durable_store:
                checks["durable_store"] = self._durable_store is not None
            return {
                "ready": all(checks.values()),
                "checks": checks,
                "instance_id": self._instance_id,
            }

        @self._fastapi.get("/metrics")
        async def metrics() -> dict[str, Any]:
            """Return instance metrics for monitoring."""
            metrics_data = {
                "instance_id": self._instance_id,
                "version": self.config.version,
                "config": {
                    "max_concurrent": self.config.max_concurrent,
                    "redis_pool_size": self.config.redis_pool_size,
                    "db_pool_size": self.config.db_pool_size,
                },
            }

            # Include middleware metrics if available
            for middleware in self._fastapi.middleware_stack.app.__dict__.get("_middleware", []):
                if hasattr(middleware, "get_metrics"):
                    metrics_data["middleware"] = middleware.get_metrics()

            return metrics_data

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
                                defn.input_model.model_json_schema() if defn.input_model else {}
                            ),
                            "outputSchema": (
                                defn.output_model.model_json_schema() if defn.output_model else {}
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
        # Extract user from Authorization header
        user = self._extract_user_from_request(request)

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

    def _extract_user_from_request(self, request: Request) -> UserInfo | None:
        """Extract user information from Authorization header.

        Supports Bearer token authentication. Override this method
        to implement custom auth logic.

        Args:
            request: The incoming request

        Returns:
            UserInfo if authenticated, None otherwise
        """
        from fastagentic.context import UserInfo

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        # Support Bearer token format
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            if token:
                return UserInfo(
                    id=token,  # Use token as user ID (can be replaced with decoded JWT)
                    token=token,
                )

        return None

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
