"""Production readiness checker for FastAgentic."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(str, Enum):
    """Status of a readiness check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class CheckCategory(str, Enum):
    """Category of readiness check."""

    SECURITY = "security"
    RELIABILITY = "reliability"
    OBSERVABILITY = "observability"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    CONFIGURATION = "configuration"


@dataclass
class CheckResult:
    """Result of a readiness check.

    Attributes:
        name: Check name
        status: Check status
        message: Human-readable message
        category: Check category
        details: Additional details
        recommendation: Fix recommendation
    """

    name: str
    status: CheckStatus
    message: str
    category: CheckCategory
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "category": self.category.value,
            "details": self.details,
            "recommendation": self.recommendation,
        }


@dataclass
class ReadinessCheck:
    """A single readiness check.

    Attributes:
        name: Check name
        description: Check description
        category: Check category
        check_fn: Function to run the check
        required: Whether this check is required to pass
    """

    name: str
    description: str
    category: CheckCategory
    check_fn: Callable[..., CheckResult | Awaitable[CheckResult]]
    required: bool = True


@dataclass
class ReadinessReport:
    """Report from readiness checks.

    Attributes:
        passed: Number of passed checks
        warned: Number of warnings
        failed: Number of failed checks
        skipped: Number of skipped checks
        is_ready: Whether all required checks passed
        results: Individual check results
        score: Readiness score (0-100)
    """

    passed: int = 0
    warned: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[CheckResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.warned + self.failed + self.skipped

    @property
    def is_ready(self) -> bool:
        """Check if system is ready for production."""
        # All required checks must pass (not fail)
        for result in self.results:
            if result.status == CheckStatus.FAIL:
                return False
        return True

    @property
    def score(self) -> int:
        """Calculate readiness score (0-100)."""
        if self.total == 0:
            return 0

        # Weights: pass=100, warn=50, fail=0, skip=0
        points = (self.passed * 100) + (self.warned * 50)
        max_points = self.total * 100

        return int((points / max_points) * 100)

    def get_by_category(
        self,
        category: CheckCategory,
    ) -> list[CheckResult]:
        """Get results for a category."""
        return [r for r in self.results if r.category == category]

    def get_failures(self) -> list[CheckResult]:
        """Get all failed checks."""
        return [r for r in self.results if r.status == CheckStatus.FAIL]

    def get_warnings(self) -> list[CheckResult]:
        """Get all warning checks."""
        return [r for r in self.results if r.status == CheckStatus.WARN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "score": self.score,
            "summary": {
                "passed": self.passed,
                "warned": self.warned,
                "failed": self.failed,
                "skipped": self.skipped,
                "total": self.total,
            },
            "results": [r.to_dict() for r in self.results],
        }


class ReadinessChecker:
    """Check production readiness.

    Example:
        checker = ReadinessChecker()

        # Run all checks
        report = await checker.run_checks(app_config)

        if report.is_ready:
            print(f"Ready for production! Score: {report.score}/100")
        else:
            print("Not ready. Issues found:")
            for failure in report.get_failures():
                print(f"  - {failure.name}: {failure.message}")
                print(f"    Fix: {failure.recommendation}")
    """

    def __init__(self) -> None:
        """Initialize readiness checker."""
        self._checks: list[ReadinessCheck] = []
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default readiness checks."""
        # Security checks
        self.add_check(
            ReadinessCheck(
                name="auth_configured",
                description="Authentication is configured",
                category=CheckCategory.SECURITY,
                check_fn=self._check_auth_configured,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="https_enabled",
                description="HTTPS/TLS is enabled",
                category=CheckCategory.SECURITY,
                check_fn=self._check_https,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="secrets_not_in_code",
                description="Secrets are not hardcoded",
                category=CheckCategory.SECURITY,
                check_fn=self._check_secrets,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="rate_limiting",
                description="Rate limiting is configured",
                category=CheckCategory.SECURITY,
                check_fn=self._check_rate_limiting,
                required=False,
            )
        )

        # Reliability checks
        self.add_check(
            ReadinessCheck(
                name="retry_policy",
                description="Retry policy is configured",
                category=CheckCategory.RELIABILITY,
                check_fn=self._check_retry_policy,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="timeout_configured",
                description="Timeouts are configured",
                category=CheckCategory.RELIABILITY,
                check_fn=self._check_timeouts,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="error_handling",
                description="Error handling is configured",
                category=CheckCategory.RELIABILITY,
                check_fn=self._check_error_handling,
            )
        )

        # Observability checks
        self.add_check(
            ReadinessCheck(
                name="logging_configured",
                description="Logging is configured",
                category=CheckCategory.OBSERVABILITY,
                check_fn=self._check_logging,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="metrics_enabled",
                description="Metrics collection is enabled",
                category=CheckCategory.OBSERVABILITY,
                check_fn=self._check_metrics,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="health_endpoint",
                description="Health endpoint is available",
                category=CheckCategory.OBSERVABILITY,
                check_fn=self._check_health_endpoint,
            )
        )

        # Performance checks
        self.add_check(
            ReadinessCheck(
                name="caching_configured",
                description="Caching is configured",
                category=CheckCategory.PERFORMANCE,
                check_fn=self._check_caching,
                required=False,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="connection_pooling",
                description="Connection pooling is enabled",
                category=CheckCategory.PERFORMANCE,
                check_fn=self._check_connection_pooling,
                required=False,
            )
        )

        # Compliance checks
        self.add_check(
            ReadinessCheck(
                name="pii_detection",
                description="PII detection is enabled",
                category=CheckCategory.COMPLIANCE,
                check_fn=self._check_pii_detection,
                required=False,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="audit_logging",
                description="Audit logging is enabled",
                category=CheckCategory.COMPLIANCE,
                check_fn=self._check_audit_logging,
                required=False,
            )
        )

        # Configuration checks
        self.add_check(
            ReadinessCheck(
                name="environment_configured",
                description="Environment is properly configured",
                category=CheckCategory.CONFIGURATION,
                check_fn=self._check_environment,
            )
        )

        self.add_check(
            ReadinessCheck(
                name="dependencies_available",
                description="Required dependencies are available",
                category=CheckCategory.CONFIGURATION,
                check_fn=self._check_dependencies,
            )
        )

    def add_check(self, check: ReadinessCheck) -> None:
        """Add a readiness check.

        Args:
            check: Check to add
        """
        self._checks.append(check)

    async def run_checks(
        self,
        config: dict[str, Any] | None = None,
        categories: list[CheckCategory] | None = None,
    ) -> ReadinessReport:
        """Run all readiness checks.

        Args:
            config: Application configuration
            categories: Only run checks in these categories

        Returns:
            ReadinessReport with results
        """
        config = config or {}
        report = ReadinessReport()

        for check in self._checks:
            # Filter by category
            if categories and check.category not in categories:
                continue

            try:
                result = check.check_fn(config)
                # Handle async check functions
                if hasattr(result, "__await__"):
                    result = await result
            except Exception as e:
                result = CheckResult(
                    name=check.name,
                    status=CheckStatus.FAIL,
                    message=f"Check failed with error: {e}",
                    category=check.category,
                )

            report.results.append(result)

            if result.status == CheckStatus.PASS:
                report.passed += 1
            elif result.status == CheckStatus.WARN:
                report.warned += 1
            elif result.status == CheckStatus.FAIL:
                report.failed += 1
            else:
                report.skipped += 1

        return report

    # Default check implementations

    def _check_auth_configured(self, config: dict[str, Any]) -> CheckResult:
        """Check if authentication is configured."""
        auth = config.get("auth", {})
        oidc = config.get("oidc_issuer")
        api_key = config.get("api_key")

        if oidc or api_key or auth.get("enabled"):
            return CheckResult(
                name="auth_configured",
                status=CheckStatus.PASS,
                message="Authentication is configured",
                category=CheckCategory.SECURITY,
            )

        return CheckResult(
            name="auth_configured",
            status=CheckStatus.FAIL,
            message="No authentication configured",
            category=CheckCategory.SECURITY,
            recommendation="Configure OIDC, API keys, or another auth method",
        )

    def _check_https(self, config: dict[str, Any]) -> CheckResult:
        """Check if HTTPS is enabled."""
        # Check environment or config
        ssl = config.get("ssl", {})
        env_ssl = os.environ.get("SSL_ENABLED", "").lower() == "true"
        is_prod = os.environ.get("ENVIRONMENT", "").lower() == "production"

        if ssl.get("enabled") or env_ssl:
            return CheckResult(
                name="https_enabled",
                status=CheckStatus.PASS,
                message="HTTPS/TLS is enabled",
                category=CheckCategory.SECURITY,
            )

        if not is_prod:
            return CheckResult(
                name="https_enabled",
                status=CheckStatus.WARN,
                message="HTTPS not enabled (acceptable for non-production)",
                category=CheckCategory.SECURITY,
                recommendation="Enable HTTPS before deploying to production",
            )

        return CheckResult(
            name="https_enabled",
            status=CheckStatus.FAIL,
            message="HTTPS not enabled in production",
            category=CheckCategory.SECURITY,
            recommendation="Configure SSL certificates and enable HTTPS",
        )

    def _check_secrets(self, config: dict[str, Any]) -> CheckResult:
        """Check that secrets are not hardcoded."""
        # Check for common secret patterns in config
        suspicious_keys = ["password", "secret", "api_key", "token"]
        hardcoded = []

        def check_dict(d: dict, path: str = "") -> None:
            for k, v in d.items():
                full_path = f"{path}.{k}" if path else k
                if isinstance(v, dict):
                    check_dict(v, full_path)
                elif isinstance(v, str) and len(v) > 8:
                    for sus in suspicious_keys:
                        if sus in k.lower() and not v.startswith("${"):
                            hardcoded.append(full_path)

        check_dict(config)

        if not hardcoded:
            return CheckResult(
                name="secrets_not_in_code",
                status=CheckStatus.PASS,
                message="No hardcoded secrets detected",
                category=CheckCategory.SECURITY,
            )

        return CheckResult(
            name="secrets_not_in_code",
            status=CheckStatus.WARN,
            message=f"Potential hardcoded secrets: {', '.join(hardcoded)}",
            category=CheckCategory.SECURITY,
            details={"suspicious_keys": hardcoded},
            recommendation="Use environment variables or a secrets manager",
        )

    def _check_rate_limiting(self, config: dict[str, Any]) -> CheckResult:
        """Check if rate limiting is configured."""
        rate_limit = config.get("rate_limit", {})

        if rate_limit.get("enabled") or rate_limit.get("rpm"):
            return CheckResult(
                name="rate_limiting",
                status=CheckStatus.PASS,
                message="Rate limiting is configured",
                category=CheckCategory.SECURITY,
                details=rate_limit,
            )

        return CheckResult(
            name="rate_limiting",
            status=CheckStatus.WARN,
            message="Rate limiting is not configured",
            category=CheckCategory.SECURITY,
            recommendation="Configure rate limits to prevent abuse",
        )

    def _check_retry_policy(self, config: dict[str, Any]) -> CheckResult:
        """Check if retry policy is configured."""
        retry = config.get("retry_policy", {})

        if retry.get("enabled") or retry.get("max_attempts"):
            return CheckResult(
                name="retry_policy",
                status=CheckStatus.PASS,
                message="Retry policy is configured",
                category=CheckCategory.RELIABILITY,
                details=retry,
            )

        return CheckResult(
            name="retry_policy",
            status=CheckStatus.WARN,
            message="No retry policy configured",
            category=CheckCategory.RELIABILITY,
            recommendation="Configure retry policy for transient failures",
        )

    def _check_timeouts(self, config: dict[str, Any]) -> CheckResult:
        """Check if timeouts are configured."""
        timeout = config.get("timeout")

        if timeout and timeout > 0:
            return CheckResult(
                name="timeout_configured",
                status=CheckStatus.PASS,
                message=f"Timeout configured: {timeout}s",
                category=CheckCategory.RELIABILITY,
            )

        return CheckResult(
            name="timeout_configured",
            status=CheckStatus.WARN,
            message="No timeout configured",
            category=CheckCategory.RELIABILITY,
            recommendation="Configure timeouts to prevent hanging requests",
        )

    def _check_error_handling(self, config: dict[str, Any]) -> CheckResult:
        """Check if error handling is configured."""
        # This is a soft check - assume OK if not explicitly disabled
        error_handling = config.get("error_handling", {})

        if error_handling.get("enabled", True):
            return CheckResult(
                name="error_handling",
                status=CheckStatus.PASS,
                message="Error handling is configured",
                category=CheckCategory.RELIABILITY,
            )

        return CheckResult(
            name="error_handling",
            status=CheckStatus.WARN,
            message="Error handling may not be configured",
            category=CheckCategory.RELIABILITY,
            recommendation="Ensure proper error handling and graceful degradation",
        )

    def _check_logging(self, config: dict[str, Any]) -> CheckResult:
        """Check if logging is configured."""
        logging_config = config.get("logging", {})
        log_level = os.environ.get("LOG_LEVEL", logging_config.get("level"))

        if log_level:
            return CheckResult(
                name="logging_configured",
                status=CheckStatus.PASS,
                message=f"Logging configured at level: {log_level}",
                category=CheckCategory.OBSERVABILITY,
            )

        return CheckResult(
            name="logging_configured",
            status=CheckStatus.WARN,
            message="Logging level not explicitly configured",
            category=CheckCategory.OBSERVABILITY,
            recommendation="Configure logging level and format",
        )

    def _check_metrics(self, config: dict[str, Any]) -> CheckResult:
        """Check if metrics are enabled."""
        telemetry = config.get("telemetry", False)
        metrics = config.get("metrics", {}).get("enabled", False)

        if telemetry or metrics:
            return CheckResult(
                name="metrics_enabled",
                status=CheckStatus.PASS,
                message="Metrics collection is enabled",
                category=CheckCategory.OBSERVABILITY,
            )

        return CheckResult(
            name="metrics_enabled",
            status=CheckStatus.WARN,
            message="Metrics collection not enabled",
            category=CheckCategory.OBSERVABILITY,
            recommendation="Enable telemetry for production monitoring",
        )

    def _check_health_endpoint(self, _config: dict[str, Any]) -> CheckResult:
        """Check if health endpoint is available."""
        # Health endpoint is always available in FastAgentic
        return CheckResult(
            name="health_endpoint",
            status=CheckStatus.PASS,
            message="Health endpoint available at /health",
            category=CheckCategory.OBSERVABILITY,
        )

    def _check_caching(self, config: dict[str, Any]) -> CheckResult:
        """Check if caching is configured."""
        cache = config.get("cache", {})

        if cache.get("enabled"):
            return CheckResult(
                name="caching_configured",
                status=CheckStatus.PASS,
                message="Caching is configured",
                category=CheckCategory.PERFORMANCE,
                details=cache,
            )

        return CheckResult(
            name="caching_configured",
            status=CheckStatus.SKIP,
            message="Caching not configured (optional)",
            category=CheckCategory.PERFORMANCE,
            recommendation="Consider enabling caching for improved performance",
        )

    def _check_connection_pooling(self, config: dict[str, Any]) -> CheckResult:
        """Check if connection pooling is enabled."""
        pool = config.get("connection_pool", {})

        if pool.get("enabled") or pool.get("size"):
            return CheckResult(
                name="connection_pooling",
                status=CheckStatus.PASS,
                message="Connection pooling is enabled",
                category=CheckCategory.PERFORMANCE,
            )

        return CheckResult(
            name="connection_pooling",
            status=CheckStatus.SKIP,
            message="Connection pooling not explicitly configured",
            category=CheckCategory.PERFORMANCE,
        )

    def _check_pii_detection(self, config: dict[str, Any]) -> CheckResult:
        """Check if PII detection is enabled."""
        pii = config.get("pii_detection", {})
        compliance = config.get("compliance", {})

        if pii.get("enabled") or compliance.get("pii_detection"):
            return CheckResult(
                name="pii_detection",
                status=CheckStatus.PASS,
                message="PII detection is enabled",
                category=CheckCategory.COMPLIANCE,
            )

        return CheckResult(
            name="pii_detection",
            status=CheckStatus.WARN,
            message="PII detection not enabled",
            category=CheckCategory.COMPLIANCE,
            recommendation="Enable PII detection for data privacy compliance",
        )

    def _check_audit_logging(self, config: dict[str, Any]) -> CheckResult:
        """Check if audit logging is enabled."""
        audit = config.get("audit", {})

        if audit.get("enabled"):
            return CheckResult(
                name="audit_logging",
                status=CheckStatus.PASS,
                message="Audit logging is enabled",
                category=CheckCategory.COMPLIANCE,
            )

        return CheckResult(
            name="audit_logging",
            status=CheckStatus.WARN,
            message="Audit logging not enabled",
            category=CheckCategory.COMPLIANCE,
            recommendation="Enable audit logging for compliance and debugging",
        )

    def _check_environment(self, config: dict[str, Any]) -> CheckResult:
        """Check environment configuration."""
        env = os.environ.get("ENVIRONMENT", config.get("environment", ""))

        if env.lower() in ("production", "prod", "staging", "stage"):
            return CheckResult(
                name="environment_configured",
                status=CheckStatus.PASS,
                message=f"Environment set to: {env}",
                category=CheckCategory.CONFIGURATION,
            )

        if env:
            return CheckResult(
                name="environment_configured",
                status=CheckStatus.WARN,
                message=f"Environment set to: {env} (non-production)",
                category=CheckCategory.CONFIGURATION,
            )

        return CheckResult(
            name="environment_configured",
            status=CheckStatus.WARN,
            message="ENVIRONMENT variable not set",
            category=CheckCategory.CONFIGURATION,
            recommendation="Set ENVIRONMENT to 'production' for production deployments",
        )

    def _check_dependencies(self, _config: dict[str, Any]) -> CheckResult:
        """Check that dependencies are available."""
        missing = []

        # Check optional dependencies
        try:
            import httpx  # noqa: F401
        except ImportError:
            missing.append("httpx (for SDK client)")

        if missing:
            return CheckResult(
                name="dependencies_available",
                status=CheckStatus.WARN,
                message=f"Optional dependencies missing: {', '.join(missing)}",
                category=CheckCategory.CONFIGURATION,
                details={"missing": missing},
            )

        return CheckResult(
            name="dependencies_available",
            status=CheckStatus.PASS,
            message="All dependencies available",
            category=CheckCategory.CONFIGURATION,
        )
