"""Metrics collection and export for FastAgentic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MetricLabels:
    """Labels for a metric."""

    labels: dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.labels.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricLabels):
            return False
        return self.labels == other.labels

    def to_prometheus(self) -> str:
        """Convert to Prometheus label format."""
        if not self.labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(pairs) + "}"


class Metric(ABC):
    """Base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[MetricLabels, float] = {}

    def _get_labels(self, **kwargs: str) -> MetricLabels:
        """Get labels from kwargs."""
        labels = {k: kwargs.get(k, "") for k in self.label_names}
        return MetricLabels(labels)

    @abstractmethod
    def get_type(self) -> str:
        """Get Prometheus metric type."""
        ...

    def get_samples(self) -> list[tuple[MetricLabels, float]]:
        """Get all samples."""
        return list(self._values.items())


class Counter(Metric):
    """A counter metric that only increases.

    Example:
        requests = Counter("requests_total", "Total requests", ["endpoint"])
        requests.inc(endpoint="/chat")
        requests.add(5, endpoint="/chat")
    """

    def get_type(self) -> str:
        return "counter"

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment counter.

        Args:
            value: Amount to increment (must be positive)
            **labels: Label values
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")

        key = self._get_labels(**labels)
        self._values[key] = self._values.get(key, 0) + value

    def add(self, value: float, **labels: str) -> None:
        """Add to counter (alias for inc)."""
        self.inc(value, **labels)

    def get(self, **labels: str) -> float:
        """Get current value."""
        key = self._get_labels(**labels)
        return self._values.get(key, 0)


class Gauge(Metric):
    """A gauge metric that can increase or decrease.

    Example:
        active = Gauge("active_requests", "Currently active requests")
        active.inc()
        active.dec()
        active.set(10)
    """

    def get_type(self) -> str:
        return "gauge"

    def set(self, value: float, **labels: str) -> None:
        """Set gauge value.

        Args:
            value: New value
            **labels: Label values
        """
        key = self._get_labels(**labels)
        self._values[key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment gauge."""
        key = self._get_labels(**labels)
        self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement gauge."""
        self.inc(-value, **labels)

    def get(self, **labels: str) -> float:
        """Get current value."""
        key = self._get_labels(**labels)
        return self._values.get(key, 0)


class Histogram(Metric):
    """A histogram metric for measuring distributions.

    Example:
        duration = Histogram(
            "request_duration_seconds",
            "Request duration",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )
        duration.observe(0.35)
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__(name, description, labels)
        self.buckets = tuple(sorted(buckets or self.DEFAULT_BUCKETS))
        self._bucket_counts: dict[MetricLabels, dict[float, int]] = {}
        self._sums: dict[MetricLabels, float] = {}
        self._counts: dict[MetricLabels, int] = {}

    def get_type(self) -> str:
        return "histogram"

    def observe(self, value: float, **labels: str) -> None:
        """Observe a value.

        Args:
            value: Observed value
            **labels: Label values
        """
        key = self._get_labels(**labels)

        # Initialize if needed
        if key not in self._bucket_counts:
            self._bucket_counts[key] = dict.fromkeys(self.buckets, 0)
            self._sums[key] = 0
            self._counts[key] = 0

        # Update buckets
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[key][bucket] += 1

        self._sums[key] += value
        self._counts[key] += 1

    def get_samples(self) -> list[tuple[MetricLabels, float]]:
        """Get all samples including bucket metrics."""
        samples = []

        for key in self._bucket_counts:
            # Bucket samples
            for bucket, count in self._bucket_counts[key].items():
                bucket_labels = MetricLabels(
                    {
                        **key.labels,
                        "le": str(bucket),
                    }
                )
                samples.append((bucket_labels, count))

            # +Inf bucket
            inf_labels = MetricLabels({**key.labels, "le": "+Inf"})
            samples.append((inf_labels, self._counts.get(key, 0)))

        return samples

    def get_sum(self, **labels: str) -> float:
        """Get sum of all observations."""
        key = self._get_labels(**labels)
        return self._sums.get(key, 0)

    def get_count(self, **labels: str) -> int:
        """Get count of observations."""
        key = self._get_labels(**labels)
        return self._counts.get(key, 0)


class MetricsRegistry:
    """Registry for all metrics.

    Example:
        registry = MetricsRegistry()

        requests = registry.counter(
            "requests_total",
            "Total requests",
            ["endpoint", "status"],
        )
        active = registry.gauge("active_requests", "Active requests")
        duration = registry.histogram(
            "request_duration_seconds",
            "Request duration",
        )

        # Use metrics
        requests.inc(endpoint="/chat", status="success")
        active.inc()
        duration.observe(0.5)

        # Export
        exporter = PrometheusExporter(registry)
        print(exporter.export())
    """

    def __init__(self, prefix: str = "fastagentic") -> None:
        """Initialize registry.

        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        self._metrics: dict[str, Metric] = {}

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Counter:
        """Create or get a counter.

        Args:
            name: Metric name
            description: Metric description
            labels: Label names

        Returns:
            Counter metric
        """
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Counter(full_name, description, labels)
        return self._metrics[full_name]  # type: ignore

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Gauge:
        """Create or get a gauge.

        Args:
            name: Metric name
            description: Metric description
            labels: Label names

        Returns:
            Gauge metric
        """
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Gauge(full_name, description, labels)
        return self._metrics[full_name]  # type: ignore

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | list[float] | None = None,
    ) -> Histogram:
        """Create or get a histogram.

        Args:
            name: Metric name
            description: Metric description
            labels: Label names
            buckets: Histogram buckets

        Returns:
            Histogram metric
        """
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Histogram(full_name, description, labels, buckets)
        return self._metrics[full_name]  # type: ignore

    def get_all_metrics(self) -> list[Metric]:
        """Get all registered metrics."""
        return list(self._metrics.values())


class MetricExporter(ABC):
    """Base class for metric exporters."""

    def __init__(self, registry: MetricsRegistry) -> None:
        self.registry = registry

    @abstractmethod
    def export(self) -> str:
        """Export metrics as string."""
        ...


class PrometheusExporter(MetricExporter):
    """Export metrics in Prometheus format.

    Example:
        registry = MetricsRegistry()
        # ... register metrics ...
        exporter = PrometheusExporter(registry)
        prometheus_text = exporter.export()
    """

    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for metric in self.registry.get_all_metrics():
            # Add HELP and TYPE
            if metric.description:
                lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} {metric.get_type()}")

            # Add samples
            if isinstance(metric, Histogram):
                self._export_histogram(metric, lines)
            else:
                for labels, value in metric.get_samples():
                    label_str = labels.to_prometheus()
                    lines.append(f"{metric.name}{label_str} {value}")

            lines.append("")

        return "\n".join(lines)

    def _export_histogram(self, metric: Histogram, lines: list[str]) -> None:
        """Export histogram metric."""
        # Get unique label sets (without le)
        base_labels_set = set()
        for labels, _ in metric.get_samples():
            base = {k: v for k, v in labels.labels.items() if k != "le"}
            base_labels_set.add(MetricLabels(base))

        for base_labels in base_labels_set:
            # Bucket samples
            for labels, value in metric.get_samples():
                base = {k: v for k, v in labels.labels.items() if k != "le"}
                if MetricLabels(base) == base_labels:
                    label_str = labels.to_prometheus()
                    lines.append(f"{metric.name}_bucket{label_str} {value}")

            # Sum
            sum_labels = base_labels.to_prometheus()
            sum_value = metric.get_sum(**base_labels.labels)
            lines.append(f"{metric.name}_sum{sum_labels} {sum_value}")

            # Count
            count_value = metric.get_count(**base_labels.labels)
            lines.append(f"{metric.name}_count{sum_labels} {count_value}")


# Default registry
default_registry = MetricsRegistry()

# Pre-registered metrics
requests_total = default_registry.counter(
    "requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

request_duration = default_registry.histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
)

active_requests = default_registry.gauge(
    "active_requests",
    "Currently active requests",
)

tokens_total = default_registry.counter(
    "tokens_total",
    "Total tokens used",
    ["model", "type"],
)

cost_total = default_registry.counter(
    "cost_total",
    "Total cost in USD",
    ["model"],
)

errors_total = default_registry.counter(
    "errors_total",
    "Total errors",
    ["endpoint", "error_type"],
)

tool_calls_total = default_registry.counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool"],
)
