"""Dashboard API for FastAgentic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastagentic.dashboard.metrics import MetricsRegistry, PrometheusExporter, default_registry
from fastagentic.dashboard.stats import StatsCollector


@dataclass
class DashboardConfig:
    """Configuration for dashboard API.

    Attributes:
        enabled: Whether dashboard is enabled
        path_prefix: URL path prefix for dashboard endpoints
        require_auth: Require authentication for dashboard
        allowed_roles: Roles allowed to access dashboard
        metrics_enabled: Enable Prometheus metrics endpoint
        stats_enabled: Enable stats endpoints
        runs_enabled: Enable run introspection endpoints
    """

    enabled: bool = True
    path_prefix: str = "/dashboard"
    require_auth: bool = True
    allowed_roles: list[str] = field(default_factory=lambda: ["admin"])
    metrics_enabled: bool = True
    stats_enabled: bool = True
    runs_enabled: bool = True


class DashboardAPI:
    """Dashboard API for operational visibility.

    Example:
        dashboard = DashboardAPI()

        # Record runs
        await dashboard.record_run_start("run-123", "/chat", input_tokens=100)
        await dashboard.record_run_end("run-123", "completed", output_tokens=500)

        # Get data
        summary = dashboard.get_summary()
        metrics = dashboard.get_metrics()
        runs = dashboard.get_recent_runs()

        # Integration with FastAPI
        app = FastAPI()
        dashboard.register_routes(app)
    """

    def __init__(
        self,
        config: DashboardConfig | None = None,
        stats_collector: StatsCollector | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize dashboard API.

        Args:
            config: Dashboard configuration
            stats_collector: Stats collector instance
            metrics_registry: Metrics registry instance
        """
        self.config = config or DashboardConfig()
        self._stats = stats_collector or StatsCollector()
        self._registry = metrics_registry or default_registry
        self._exporter = PrometheusExporter(self._registry)

    async def record_run_start(
        self,
        run_id: str,
        endpoint: str,
        input_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the start of a run.

        Args:
            run_id: Run identifier
            endpoint: Endpoint being called
            input_tokens: Input token count
            metadata: Additional metadata
        """
        await self._stats.start_run(run_id, endpoint, input_tokens, metadata)

        # Update metrics
        from fastagentic.dashboard.metrics import active_requests

        active_requests.inc()

    async def record_run_end(
        self,
        run_id: str,
        status: str,
        output_tokens: int = 0,
        cost: float = 0.0,
        model: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record the end of a run.

        Args:
            run_id: Run identifier
            status: Final status
            output_tokens: Output token count
            cost: Run cost
            model: Model used
            error: Error message if failed
        """
        run = await self._stats.complete_run(run_id, status, output_tokens, cost, error)

        # Update metrics
        from fastagentic.dashboard.metrics import (
            active_requests,
            cost_total,
            errors_total,
            request_duration,
            requests_total,
            tokens_total,
        )

        active_requests.dec()

        if run:
            requests_total.inc(
                endpoint=run.endpoint,
                method="POST",
                status=status,
            )

            if run.duration_ms:
                request_duration.observe(
                    run.duration_ms / 1000,
                    endpoint=run.endpoint,
                )

            if run.input_tokens > 0:
                tokens_total.inc(
                    run.input_tokens,
                    model=model or "unknown",
                    type="input",
                )

            if output_tokens > 0:
                tokens_total.inc(
                    output_tokens,
                    model=model or "unknown",
                    type="output",
                )

            if cost > 0:
                cost_total.inc(cost, model=model or "unknown")

            if status != "completed":
                error_type = "timeout" if status == "timeout" else "error"
                errors_total.inc(
                    endpoint=run.endpoint,
                    error_type=error_type,
                )

    async def record_tool_call(self, run_id: str, tool_name: str) -> None:
        """Record a tool call.

        Args:
            run_id: Run identifier
            tool_name: Tool that was called
        """
        await self._stats.record_tool_call(run_id)

        from fastagentic.dashboard.metrics import tool_calls_total

        tool_calls_total.inc(tool=tool_name)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all statistics.

        Returns:
            Summary dictionary
        """
        return self._stats.get_summary()

    def get_system_stats(self) -> dict[str, Any]:
        """Get system-level statistics.

        Returns:
            System stats dictionary
        """
        return self._stats.get_system_stats().to_dict()

    def get_endpoint_stats(self, endpoint: str | None = None) -> dict[str, Any]:
        """Get endpoint statistics.

        Args:
            endpoint: Specific endpoint or None for all

        Returns:
            Endpoint stats dictionary
        """
        if endpoint:
            stats = self._stats.get_endpoint_stats(endpoint)
            return stats.to_dict() if stats else {}
        else:
            return {ep.endpoint: ep.to_dict() for ep in self._stats.get_all_endpoint_stats()}

    def get_recent_runs(
        self,
        limit: int = 100,
        endpoint: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent runs.

        Args:
            limit: Maximum runs to return
            endpoint: Filter by endpoint
            status: Filter by status

        Returns:
            List of run dictionaries
        """
        runs = self._stats.get_recent_runs(limit, endpoint, status)
        return [r.to_dict() for r in runs]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Run dictionary or None
        """
        run = self._stats.get_run(run_id)
        return run.to_dict() if run else None

    def get_time_series(
        self,
        metric: str,
        endpoint: str | None = None,
        duration_seconds: int = 3600,
        bucket_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Get time series data.

        Args:
            metric: Metric name
            endpoint: Filter by endpoint
            duration_seconds: Time range
            bucket_seconds: Bucket size

        Returns:
            List of time series points
        """
        points = self._stats.get_time_series(metric, endpoint, duration_seconds, bucket_seconds)
        return [p.to_dict() for p in points]

    def get_metrics(self) -> str:
        """Get Prometheus metrics.

        Returns:
            Prometheus text format metrics
        """
        return self._exporter.export()

    def get_health(self) -> dict[str, Any]:
        """Get health status.

        Returns:
            Health status dictionary
        """
        system = self._stats.get_system_stats()
        return {
            "status": "healthy",
            "uptime_seconds": system.uptime_seconds,
            "active_requests": system.active_requests,
            "error_rate": system.error_rate,
        }

    def get_routes(self) -> list[dict[str, Any]]:
        """Get dashboard route definitions.

        Returns:
            List of route definitions for framework integration
        """
        routes = []
        prefix = self.config.path_prefix

        if self.config.stats_enabled:
            routes.extend(
                [
                    {
                        "path": f"{prefix}/summary",
                        "method": "GET",
                        "handler": "get_summary",
                        "description": "Get dashboard summary",
                    },
                    {
                        "path": f"{prefix}/stats/system",
                        "method": "GET",
                        "handler": "get_system_stats",
                        "description": "Get system statistics",
                    },
                    {
                        "path": f"{prefix}/stats/endpoints",
                        "method": "GET",
                        "handler": "get_endpoint_stats",
                        "description": "Get endpoint statistics",
                    },
                    {
                        "path": f"{prefix}/stats/timeseries",
                        "method": "GET",
                        "handler": "get_time_series",
                        "description": "Get time series data",
                    },
                ]
            )

        if self.config.runs_enabled:
            routes.extend(
                [
                    {
                        "path": f"{prefix}/runs",
                        "method": "GET",
                        "handler": "get_recent_runs",
                        "description": "Get recent runs",
                    },
                    {
                        "path": f"{prefix}/runs/{{run_id}}",
                        "method": "GET",
                        "handler": "get_run",
                        "description": "Get specific run",
                    },
                ]
            )

        if self.config.metrics_enabled:
            routes.append(
                {
                    "path": f"{prefix}/metrics",
                    "method": "GET",
                    "handler": "get_metrics",
                    "description": "Get Prometheus metrics",
                }
            )

        routes.append(
            {
                "path": f"{prefix}/health",
                "method": "GET",
                "handler": "get_health",
                "description": "Get health status",
            }
        )

        return routes
