"""Dashboard module for FastAgentic.

Provides operational visibility, metrics, and run introspection.
"""

from fastagentic.dashboard.stats import (
    StatsCollector,
    RunStats,
    EndpointStats,
    SystemStats,
    TimeSeriesPoint,
)
from fastagentic.dashboard.metrics import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    MetricExporter,
    PrometheusExporter,
)
from fastagentic.dashboard.api import (
    DashboardAPI,
    DashboardConfig,
)

__all__ = [
    # Stats
    "StatsCollector",
    "RunStats",
    "EndpointStats",
    "SystemStats",
    "TimeSeriesPoint",
    # Metrics
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricExporter",
    "PrometheusExporter",
    # API
    "DashboardAPI",
    "DashboardConfig",
]
