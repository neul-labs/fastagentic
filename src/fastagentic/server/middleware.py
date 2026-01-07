"""Server middleware for production deployments."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger()


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit concurrent requests.

    Prevents server overload by rejecting requests when the
    concurrent request limit is reached. Returns HTTP 503
    with a Retry-After header.

    Example:
        from fastapi import FastAPI
        from fastagentic.server.middleware import ConcurrencyLimitMiddleware

        app = FastAPI()
        app.add_middleware(ConcurrencyLimitMiddleware, max_concurrent=100)
    """

    def __init__(
        self,
        app: Callable[..., Any],
        max_concurrent: int = 100,
        retry_after: int = 1,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application
            max_concurrent: Maximum concurrent requests allowed
            retry_after: Retry-After header value in seconds
        """
        super().__init__(app)
        self.max_concurrent = max_concurrent
        self.retry_after = retry_after
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_requests = 0

    async def dispatch(  # type: ignore[override]
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request with concurrency limiting."""
        # Skip health checks from concurrency limiting
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        # Try to acquire semaphore with non-blocking attempt
        acquired = self._semaphore.locked()
        if not acquired:
            # Quick check passed, try to acquire (still racy but minimized window)
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=0.001)
            except asyncio.TimeoutError:
                logger.warning(
                    "Request rejected due to concurrency limit",
                    current=self._current_requests,
                    max=self.max_concurrent,
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Service temporarily unavailable",
                        "message": "Too many concurrent requests",
                        "retry_after": self.retry_after,
                    },
                    headers={"Retry-After": str(self.retry_after)},
                )
        else:
            # Semaphore is fully exhausted, reject immediately
            logger.warning(
                "Request rejected due to concurrency limit",
                current=self._current_requests,
                max=self.max_concurrent,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "Too many concurrent requests",
                    "retry_after": self.retry_after,
                },
                headers={"Retry-After": str(self.retry_after)},
            )

        self._current_requests += 1
        try:
            return await call_next(request)
        finally:
            self._current_requests -= 1
            self._semaphore.release()

    @property
    def current_requests(self) -> int:
        """Get current number of in-flight requests."""
        return self._current_requests


class InstanceMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to add instance labels to metrics.

    Adds instance identification headers and tracks metrics
    per instance for cluster deployments.

    Example:
        app.add_middleware(
            InstanceMetricsMiddleware,
            instance_id="worker-1",
        )
    """

    def __init__(
        self,
        app: Callable[..., Any],
        instance_id: str | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application
            instance_id: Unique identifier for this instance
        """
        super().__init__(app)
        self.instance_id = instance_id or os.environ.get("FASTAGENTIC_INSTANCE_ID", "unknown")
        self._request_count = 0
        self._error_count = 0
        self._lock = asyncio.Lock()

    async def dispatch(  # type: ignore[override]
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process request and add instance headers."""
        async with self._lock:
            self._request_count += 1

        response = await call_next(request)

        # Add instance identification header
        response.headers["X-FastAgentic-Instance"] = self.instance_id

        # Track errors
        if response.status_code >= 500:
            async with self._lock:
                self._error_count += 1

        return response

    def get_metrics(self) -> dict[str, Any]:
        """Get instance metrics."""
        return {
            "instance_id": self.instance_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
        }
