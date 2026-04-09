"""Statistics collection for FastAgentic dashboard."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeSeriesPoint:
    """A point in a time series.

    Attributes:
        timestamp: Unix timestamp
        value: Metric value
        labels: Optional labels
    """

    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class RunStats:
    """Statistics for a single run.

    Attributes:
        run_id: Run identifier
        endpoint: Endpoint called
        status: Run status
        started_at: Start timestamp
        completed_at: Completion timestamp
        duration_ms: Duration in milliseconds
        input_tokens: Input token count
        output_tokens: Output token count
        total_tokens: Total tokens
        cost: Estimated cost
        tool_calls: Number of tool calls
        error: Error message if failed
        metadata: Additional metadata
    """

    run_id: str
    endpoint: str
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    tool_calls: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        status: str,
        output_tokens: int = 0,
        cost: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Mark run as complete."""
        self.completed_at = time.time()
        self.duration_ms = (self.completed_at - self.started_at) * 1000
        self.status = status
        self.output_tokens = output_tokens
        self.total_tokens = self.input_tokens + output_tokens
        self.cost = cost
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "endpoint": self.endpoint,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class EndpointStats:
    """Aggregated statistics for an endpoint.

    Attributes:
        endpoint: Endpoint path
        total_requests: Total request count
        successful_requests: Successful request count
        failed_requests: Failed request count
        total_duration_ms: Sum of all durations
        total_tokens: Total tokens used
        total_cost: Total cost
        avg_duration_ms: Average duration
        p50_duration_ms: 50th percentile duration
        p95_duration_ms: 95th percentile duration
        p99_duration_ms: 99th percentile duration
        error_rate: Error rate (0-1)
    """

    endpoint: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    durations: list[float] = field(default_factory=list)

    @property
    def avg_duration_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_duration_ms / self.total_requests

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def _percentile(self, p: float) -> float:
        if not self.durations:
            return 0.0
        sorted_d = sorted(self.durations)
        idx = int(len(sorted_d) * p)
        return sorted_d[min(idx, len(sorted_d) - 1)]

    @property
    def p50_duration_ms(self) -> float:
        return self._percentile(0.5)

    @property
    def p95_duration_ms(self) -> float:
        return self._percentile(0.95)

    @property
    def p99_duration_ms(self) -> float:
        return self._percentile(0.99)

    def record(self, run: RunStats) -> None:
        """Record a completed run."""
        self.total_requests += 1
        if run.status == "completed":
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if run.duration_ms:
            self.total_duration_ms += run.duration_ms
            self.durations.append(run.duration_ms)
            # Keep only last 1000 for percentiles
            if len(self.durations) > 1000:
                self.durations = self.durations[-1000:]

        self.total_tokens += run.total_tokens
        self.total_cost += run.cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_duration_ms": self.avg_duration_ms,
            "p50_duration_ms": self.p50_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error_rate": self.error_rate,
        }


@dataclass
class SystemStats:
    """System-level statistics.

    Attributes:
        uptime_seconds: Server uptime
        total_requests: Total requests handled
        active_requests: Currently active requests
        total_errors: Total error count
        avg_response_time_ms: Average response time
        requests_per_second: Current request rate
        tokens_per_minute: Current token rate
        cost_per_hour: Current cost rate
    """

    started_at: float = field(default_factory=time.time)
    total_requests: int = 0
    active_requests: int = 0
    total_errors: int = 0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    request_times: list[float] = field(default_factory=list)
    token_times: list[tuple[float, int]] = field(default_factory=list)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def avg_response_time_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_duration_ms / self.total_requests

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second over last minute."""
        now = time.time()
        cutoff = now - 60
        recent = [t for t in self.request_times if t > cutoff]
        if not recent:
            return 0.0
        return len(recent) / 60.0

    @property
    def tokens_per_minute(self) -> int:
        """Calculate tokens per minute over last minute."""
        now = time.time()
        cutoff = now - 60
        recent = [(t, tokens) for t, tokens in self.token_times if t > cutoff]
        return sum(tokens for _, tokens in recent)

    @property
    def cost_per_hour(self) -> float:
        """Estimate hourly cost based on recent usage."""
        uptime_hours = self.uptime_seconds / 3600
        if uptime_hours < 0.001:
            return 0.0
        return self.total_cost / uptime_hours

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    def record_request_start(self) -> None:
        """Record request start."""
        self.active_requests += 1
        self.request_times.append(time.time())
        # Keep last 1000 timestamps
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]

    def record_request_end(
        self,
        duration_ms: float,
        tokens: int,
        cost: float,
        is_error: bool,
    ) -> None:
        """Record request completion."""
        self.active_requests = max(0, self.active_requests - 1)
        self.total_requests += 1
        self.total_duration_ms += duration_ms
        self.total_tokens += tokens
        self.total_cost += cost

        if is_error:
            self.total_errors += 1

        if tokens > 0:
            self.token_times.append((time.time(), tokens))
            if len(self.token_times) > 1000:
                self.token_times = self.token_times[-1000:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uptime_seconds": self.uptime_seconds,
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "total_errors": self.total_errors,
            "avg_response_time_ms": self.avg_response_time_ms,
            "requests_per_second": self.requests_per_second,
            "tokens_per_minute": self.tokens_per_minute,
            "cost_per_hour": self.cost_per_hour,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error_rate": self.error_rate,
        }


class StatsCollector:
    """Collect and aggregate statistics.

    Example:
        collector = StatsCollector()

        # Record a run
        run = collector.start_run("run-123", "/chat")
        # ... process request ...
        collector.complete_run("run-123", "completed", tokens=1500, cost=0.05)

        # Get stats
        system = collector.get_system_stats()
        endpoint = collector.get_endpoint_stats("/chat")
        recent = collector.get_recent_runs(limit=10)
    """

    def __init__(self, max_runs: int = 1000) -> None:
        """Initialize stats collector.

        Args:
            max_runs: Maximum recent runs to keep
        """
        self._max_runs = max_runs
        self._system = SystemStats()
        self._endpoints: dict[str, EndpointStats] = {}
        self._runs: dict[str, RunStats] = {}
        self._run_order: list[str] = []
        self._lock = asyncio.Lock()

    async def start_run(
        self,
        run_id: str,
        endpoint: str,
        input_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> RunStats:
        """Start tracking a run.

        Args:
            run_id: Run identifier
            endpoint: Endpoint being called
            input_tokens: Input token count
            metadata: Additional metadata

        Returns:
            RunStats for the run
        """
        async with self._lock:
            run = RunStats(
                run_id=run_id,
                endpoint=endpoint,
                input_tokens=input_tokens,
                metadata=metadata or {},
            )
            self._runs[run_id] = run
            self._run_order.append(run_id)

            # Prune old runs
            while len(self._run_order) > self._max_runs:
                old_id = self._run_order.pop(0)
                self._runs.pop(old_id, None)

            self._system.record_request_start()
            return run

    async def complete_run(
        self,
        run_id: str,
        status: str,
        output_tokens: int = 0,
        cost: float = 0.0,
        error: str | None = None,
    ) -> RunStats | None:
        """Complete a run.

        Args:
            run_id: Run identifier
            status: Final status
            output_tokens: Output token count
            cost: Run cost
            error: Error message if failed

        Returns:
            Completed RunStats or None
        """
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None

            run.complete(status, output_tokens, cost, error)

            # Update endpoint stats
            if run.endpoint not in self._endpoints:
                self._endpoints[run.endpoint] = EndpointStats(run.endpoint)
            self._endpoints[run.endpoint].record(run)

            # Update system stats
            self._system.record_request_end(
                duration_ms=run.duration_ms or 0,
                tokens=run.total_tokens,
                cost=run.cost,
                is_error=status != "completed",
            )

            return run

    async def record_tool_call(self, run_id: str) -> None:
        """Record a tool call for a run."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.tool_calls += 1

    def get_run(self, run_id: str) -> RunStats | None:
        """Get stats for a specific run."""
        return self._runs.get(run_id)

    def get_recent_runs(
        self,
        limit: int = 100,
        endpoint: str | None = None,
        status: str | None = None,
    ) -> list[RunStats]:
        """Get recent runs.

        Args:
            limit: Maximum runs to return
            endpoint: Filter by endpoint
            status: Filter by status

        Returns:
            List of RunStats
        """
        runs = list(self._runs.values())

        # Filter
        if endpoint:
            runs = [r for r in runs if r.endpoint == endpoint]
        if status:
            runs = [r for r in runs if r.status == status]

        # Sort by start time descending
        runs.sort(key=lambda r: r.started_at, reverse=True)

        return runs[:limit]

    def get_endpoint_stats(self, endpoint: str) -> EndpointStats | None:
        """Get stats for an endpoint."""
        return self._endpoints.get(endpoint)

    def get_all_endpoint_stats(self) -> list[EndpointStats]:
        """Get stats for all endpoints."""
        return list(self._endpoints.values())

    def get_system_stats(self) -> SystemStats:
        """Get system-level stats."""
        return self._system

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all stats."""
        return {
            "system": self._system.to_dict(),
            "endpoints": {ep: stats.to_dict() for ep, stats in self._endpoints.items()},
            "recent_errors": [r.to_dict() for r in self.get_recent_runs(limit=10, status="failed")],
        }

    def get_time_series(
        self,
        metric: str,
        endpoint: str | None = None,
        duration_seconds: int = 3600,
        bucket_seconds: int = 60,
    ) -> list[TimeSeriesPoint]:
        """Get time series data for a metric.

        Args:
            metric: Metric name (requests, tokens, cost, errors)
            endpoint: Filter by endpoint
            duration_seconds: Time range
            bucket_seconds: Bucket size

        Returns:
            List of time series points
        """
        now = time.time()
        start = now - duration_seconds
        num_buckets = duration_seconds // bucket_seconds

        # Initialize buckets
        buckets: dict[int, dict[str, float]] = {
            i: {"requests": 0, "tokens": 0, "cost": 0, "errors": 0, "duration": 0}
            for i in range(num_buckets)
        }

        # Fill buckets from runs
        for run in self._runs.values():
            if run.started_at < start:
                continue
            if endpoint and run.endpoint != endpoint:
                continue

            bucket_idx = int((run.started_at - start) / bucket_seconds)
            if 0 <= bucket_idx < num_buckets:
                buckets[bucket_idx]["requests"] += 1
                buckets[bucket_idx]["tokens"] += run.total_tokens
                buckets[bucket_idx]["cost"] += run.cost
                if run.status != "completed":
                    buckets[bucket_idx]["errors"] += 1
                if run.duration_ms:
                    buckets[bucket_idx]["duration"] += run.duration_ms

        # Convert to time series
        points = []
        for i in range(num_buckets):
            ts = start + (i * bucket_seconds)
            value = buckets[i].get(metric, 0)
            points.append(
                TimeSeriesPoint(
                    timestamp=ts,
                    value=value,
                    labels={"endpoint": endpoint} if endpoint else {},
                )
            )

        return points
