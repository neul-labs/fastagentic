"""Tests for dashboard module."""

import time

import pytest

from fastagentic.dashboard import (
    Counter,
    DashboardAPI,
    DashboardConfig,
    EndpointStats,
    Gauge,
    Histogram,
    MetricsRegistry,
    PrometheusExporter,
    RunStats,
    StatsCollector,
    SystemStats,
)

# ============================================================================
# Stats Tests
# ============================================================================


class TestRunStats:
    """Tests for RunStats."""

    def test_create_run_stats(self):
        """Test creating run stats."""
        stats = RunStats(run_id="run-123", endpoint="/chat")
        assert stats.run_id == "run-123"
        assert stats.endpoint == "/chat"
        assert stats.status == "pending"

    def test_complete_run(self):
        """Test completing a run."""
        stats = RunStats(run_id="run-123", endpoint="/chat")
        stats.complete(
            status="completed",
            output_tokens=100,
            cost=0.05,
        )

        assert stats.status == "completed"
        assert stats.output_tokens == 100
        assert stats.cost == 0.05
        assert stats.duration_ms is not None

    def test_to_dict(self):
        """Test serialization."""
        stats = RunStats(run_id="run-123", endpoint="/chat")
        data = stats.to_dict()

        assert data["run_id"] == "run-123"
        assert data["endpoint"] == "/chat"


class TestEndpointStats:
    """Tests for EndpointStats."""

    def test_create_endpoint_stats(self):
        """Test creating endpoint stats."""
        stats = EndpointStats(endpoint="/chat")
        assert stats.endpoint == "/chat"
        assert stats.total_requests == 0

    def test_record_run(self):
        """Test recording a run."""
        stats = EndpointStats(endpoint="/chat")
        run = RunStats(run_id="run-1", endpoint="/chat")
        run.complete("completed", output_tokens=100, cost=0.05)

        stats.record(run)

        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.total_tokens == 100
        assert stats.total_cost == 0.05

    def test_error_rate(self):
        """Test error rate calculation."""
        stats = EndpointStats(endpoint="/chat")

        # Add success
        run1 = RunStats(run_id="run-1", endpoint="/chat")
        run1.complete("completed")
        stats.record(run1)

        # Add failure
        run2 = RunStats(run_id="run-2", endpoint="/chat")
        run2.complete("failed", error="Error")
        stats.record(run2)

        assert stats.error_rate == 0.5

    def test_percentiles(self):
        """Test percentile calculations."""
        stats = EndpointStats(endpoint="/chat")

        # Add runs with different durations
        for i, duration in enumerate([100, 200, 300, 400, 500]):
            run = RunStats(run_id=f"run-{i}", endpoint="/chat")
            run.duration_ms = duration
            stats.durations.append(duration)

        assert stats.p50_duration_ms == 300
        assert stats.p95_duration_ms >= 400


class TestSystemStats:
    """Tests for SystemStats."""

    def test_create_system_stats(self):
        """Test creating system stats."""
        stats = SystemStats()
        assert stats.total_requests == 0
        assert stats.active_requests == 0

    def test_uptime(self):
        """Test uptime calculation."""
        stats = SystemStats()
        time.sleep(0.1)
        assert stats.uptime_seconds >= 0.1

    def test_request_tracking(self):
        """Test request tracking."""
        stats = SystemStats()

        stats.record_request_start()
        assert stats.active_requests == 1

        stats.record_request_end(
            duration_ms=100,
            tokens=500,
            cost=0.05,
            is_error=False,
        )
        assert stats.active_requests == 0
        assert stats.total_requests == 1
        assert stats.total_tokens == 500


class TestStatsCollector:
    """Tests for StatsCollector."""

    @pytest.mark.asyncio
    async def test_start_and_complete_run(self):
        """Test starting and completing a run."""
        collector = StatsCollector()

        run = await collector.start_run("run-123", "/chat", input_tokens=50)
        assert run.run_id == "run-123"

        completed = await collector.complete_run(
            "run-123",
            "completed",
            output_tokens=100,
            cost=0.05,
        )
        assert completed.status == "completed"

    @pytest.mark.asyncio
    async def test_get_recent_runs(self):
        """Test getting recent runs."""
        collector = StatsCollector()

        for i in range(5):
            await collector.start_run(f"run-{i}", "/chat")
            await collector.complete_run(f"run-{i}", "completed")

        runs = collector.get_recent_runs(limit=3)
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_get_endpoint_stats(self):
        """Test getting endpoint stats."""
        collector = StatsCollector()

        await collector.start_run("run-1", "/chat")
        await collector.complete_run("run-1", "completed")

        stats = collector.get_endpoint_stats("/chat")
        assert stats is not None
        assert stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_get_system_stats(self):
        """Test getting system stats."""
        collector = StatsCollector()

        await collector.start_run("run-1", "/chat")
        await collector.complete_run("run-1", "completed")

        stats = collector.get_system_stats()
        assert stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test getting summary."""
        collector = StatsCollector()

        await collector.start_run("run-1", "/chat")
        await collector.complete_run("run-1", "completed")

        summary = collector.get_summary()
        assert "system" in summary
        assert "endpoints" in summary


# ============================================================================
# Metrics Tests
# ============================================================================


class TestCounter:
    """Tests for Counter metric."""

    def test_increment(self):
        """Test incrementing counter."""
        counter = Counter("requests", "Total requests")
        counter.inc()
        counter.inc()

        assert counter.get() == 2

    def test_increment_with_labels(self):
        """Test incrementing with labels."""
        counter = Counter("requests", "Total requests", ["endpoint"])
        counter.inc(endpoint="/chat")
        counter.inc(endpoint="/chat")
        counter.inc(endpoint="/search")

        assert counter.get(endpoint="/chat") == 2
        assert counter.get(endpoint="/search") == 1

    def test_add(self):
        """Test adding to counter."""
        counter = Counter("tokens", "Total tokens")
        counter.add(100)
        counter.add(50)

        assert counter.get() == 150


class TestGauge:
    """Tests for Gauge metric."""

    def test_set(self):
        """Test setting gauge."""
        gauge = Gauge("active", "Active connections")
        gauge.set(5)

        assert gauge.get() == 5

    def test_inc_dec(self):
        """Test incrementing and decrementing."""
        gauge = Gauge("active", "Active connections")
        gauge.inc()
        gauge.inc()
        gauge.dec()

        assert gauge.get() == 1


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Test observing values."""
        hist = Histogram("duration", "Request duration")
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(1.0)

        assert hist.get_count() == 3
        assert hist.get_sum() == 1.6

    def test_custom_buckets(self):
        """Test custom buckets."""
        hist = Histogram(
            "duration",
            "Request duration",
            buckets=[0.1, 0.5, 1.0],
        )
        assert hist.buckets == (0.1, 0.5, 1.0)


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_create_counter(self):
        """Test creating counter."""
        registry = MetricsRegistry(prefix="test")
        counter = registry.counter("requests", "Total requests")

        assert counter.name == "test_requests"

    def test_create_gauge(self):
        """Test creating gauge."""
        registry = MetricsRegistry(prefix="test")
        gauge = registry.gauge("active", "Active")

        assert gauge.name == "test_active"

    def test_create_histogram(self):
        """Test creating histogram."""
        registry = MetricsRegistry(prefix="test")
        hist = registry.histogram("duration", "Duration")

        assert hist.name == "test_duration"

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        registry = MetricsRegistry()
        registry.counter("a", "A")
        registry.gauge("b", "B")

        metrics = registry.get_all_metrics()
        assert len(metrics) == 2


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_export_counter(self):
        """Test exporting counter."""
        registry = MetricsRegistry(prefix="test")
        counter = registry.counter("requests", "Total requests")
        counter.inc()

        exporter = PrometheusExporter(registry)
        output = exporter.export()

        assert "test_requests" in output
        assert "# TYPE test_requests counter" in output

    def test_export_gauge(self):
        """Test exporting gauge."""
        registry = MetricsRegistry(prefix="test")
        gauge = registry.gauge("active", "Active")
        gauge.set(5)

        exporter = PrometheusExporter(registry)
        output = exporter.export()

        assert "test_active" in output
        assert "# TYPE test_active gauge" in output


# ============================================================================
# Dashboard API Tests
# ============================================================================


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DashboardConfig()
        assert config.enabled is True
        assert config.path_prefix == "/dashboard"

    def test_custom_config(self):
        """Test custom configuration."""
        config = DashboardConfig(
            path_prefix="/admin",
            require_auth=False,
        )
        assert config.path_prefix == "/admin"
        assert config.require_auth is False


class TestDashboardAPI:
    """Tests for DashboardAPI."""

    @pytest.mark.asyncio
    async def test_record_run(self):
        """Test recording a run."""
        dashboard = DashboardAPI()

        await dashboard.record_run_start("run-1", "/chat", input_tokens=50)
        await dashboard.record_run_end("run-1", "completed", output_tokens=100)

        run = dashboard.get_run("run-1")
        assert run is not None
        assert run["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test getting summary."""
        dashboard = DashboardAPI()

        await dashboard.record_run_start("run-1", "/chat")
        await dashboard.record_run_end("run-1", "completed")

        summary = dashboard.get_summary()
        assert "system" in summary
        assert "endpoints" in summary

    @pytest.mark.asyncio
    async def test_get_system_stats(self):
        """Test getting system stats."""
        dashboard = DashboardAPI()

        stats = dashboard.get_system_stats()
        assert "uptime_seconds" in stats

    @pytest.mark.asyncio
    async def test_get_recent_runs(self):
        """Test getting recent runs."""
        dashboard = DashboardAPI()

        await dashboard.record_run_start("run-1", "/chat")
        await dashboard.record_run_end("run-1", "completed")

        runs = dashboard.get_recent_runs(limit=10)
        assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting Prometheus metrics."""
        dashboard = DashboardAPI()

        metrics = dashboard.get_metrics()
        assert "fastagentic_" in metrics

    @pytest.mark.asyncio
    async def test_get_health(self):
        """Test getting health status."""
        dashboard = DashboardAPI()

        health = dashboard.get_health()
        assert health["status"] == "healthy"

    def test_get_routes(self):
        """Test getting route definitions."""
        dashboard = DashboardAPI()

        routes = dashboard.get_routes()
        assert len(routes) > 0

        paths = [r["path"] for r in routes]
        assert any("/health" in p for p in paths)
        assert any("/metrics" in p for p in paths)
