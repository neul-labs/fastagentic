"""Dashboard module for FastAgentic.

Provides operational visibility, metrics, and run introspection.
"""

from fastagentic.dashboard.api import DashboardAPI, DashboardConfig
from fastagentic.dashboard.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricExporter,
    MetricsRegistry,
    PrometheusExporter,
)
from fastagentic.dashboard.stats import (
    EndpointStats,
    RunStats,
    StatsCollector,
    SystemStats,
    TimeSeriesPoint,
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
