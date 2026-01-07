"""Tests for ops module."""

import pytest

from fastagentic.ops import (
    CheckResult,
    CheckStatus,
    ReadinessCheck,
    ReadinessChecker,
    ReadinessReport,
)
from fastagentic.ops.readiness import CheckCategory

# ============================================================================
# CheckResult Tests
# ============================================================================


class TestCheckResult:
    """Tests for CheckResult."""

    def test_create_pass_result(self):
        """Test creating a passing result."""
        result = CheckResult(
            name="test_check",
            status=CheckStatus.PASS,
            message="Check passed",
            category=CheckCategory.SECURITY,
        )
        assert result.status == CheckStatus.PASS
        assert result.name == "test_check"

    def test_create_fail_result(self):
        """Test creating a failing result."""
        result = CheckResult(
            name="test_check",
            status=CheckStatus.FAIL,
            message="Check failed",
            category=CheckCategory.SECURITY,
            recommendation="Fix this issue",
        )
        assert result.status == CheckStatus.FAIL
        assert result.recommendation == "Fix this issue"

    def test_to_dict(self):
        """Test serialization."""
        result = CheckResult(
            name="test",
            status=CheckStatus.WARN,
            message="Warning",
            category=CheckCategory.RELIABILITY,
        )
        data = result.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "warn"
        assert data["category"] == "reliability"


# ============================================================================
# ReadinessReport Tests
# ============================================================================


class TestReadinessReport:
    """Tests for ReadinessReport."""

    def test_empty_report(self):
        """Test empty report."""
        report = ReadinessReport()
        assert report.total == 0
        assert report.is_ready
        assert report.score == 0

    def test_all_passed(self):
        """Test all checks passed."""
        report = ReadinessReport(passed=5)
        report.results = [
            CheckResult("a", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("b", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("c", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("d", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("e", CheckStatus.PASS, "", CheckCategory.SECURITY),
        ]

        assert report.is_ready
        assert report.score == 100

    def test_with_failures(self):
        """Test report with failures."""
        report = ReadinessReport(passed=2, failed=1)
        report.results = [
            CheckResult("a", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("b", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("c", CheckStatus.FAIL, "", CheckCategory.SECURITY),
        ]

        assert not report.is_ready
        assert report.score < 100

    def test_with_warnings(self):
        """Test report with warnings."""
        report = ReadinessReport(passed=2, warned=1)
        report.results = [
            CheckResult("a", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("b", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("c", CheckStatus.WARN, "", CheckCategory.SECURITY),
        ]

        # Warnings don't block readiness
        assert report.is_ready
        # But affect score
        assert report.score < 100

    def test_get_failures(self):
        """Test getting failures."""
        report = ReadinessReport()
        report.results = [
            CheckResult("a", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("b", CheckStatus.FAIL, "", CheckCategory.SECURITY),
            CheckResult("c", CheckStatus.FAIL, "", CheckCategory.SECURITY),
        ]

        failures = report.get_failures()
        assert len(failures) == 2

    def test_get_by_category(self):
        """Test getting by category."""
        report = ReadinessReport()
        report.results = [
            CheckResult("a", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("b", CheckStatus.PASS, "", CheckCategory.RELIABILITY),
            CheckResult("c", CheckStatus.PASS, "", CheckCategory.SECURITY),
        ]

        security = report.get_by_category(CheckCategory.SECURITY)
        assert len(security) == 2

    def test_to_dict(self):
        """Test serialization."""
        report = ReadinessReport(passed=2, failed=1)
        report.results = [
            CheckResult("a", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("b", CheckStatus.PASS, "", CheckCategory.SECURITY),
            CheckResult("c", CheckStatus.FAIL, "", CheckCategory.SECURITY),
        ]

        data = report.to_dict()
        assert "is_ready" in data
        assert "score" in data
        assert "summary" in data
        assert "results" in data


# ============================================================================
# ReadinessChecker Tests
# ============================================================================


class TestReadinessChecker:
    """Tests for ReadinessChecker."""

    @pytest.mark.asyncio
    async def test_run_checks(self):
        """Test running all checks."""
        checker = ReadinessChecker()
        report = await checker.run_checks()

        assert report.total > 0
        assert len(report.results) > 0

    @pytest.mark.asyncio
    async def test_run_checks_with_config(self):
        """Test running checks with configuration."""
        checker = ReadinessChecker()
        config = {
            "auth": {"enabled": True},
            "retry_policy": {"max_attempts": 3},
            "timeout": 30,
            "telemetry": True,
        }
        report = await checker.run_checks(config)

        # Auth check should pass
        auth_results = [r for r in report.results if r.name == "auth_configured"]
        assert len(auth_results) == 1
        assert auth_results[0].status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_run_checks_by_category(self):
        """Test running checks for specific category."""
        checker = ReadinessChecker()
        report = await checker.run_checks(categories=[CheckCategory.SECURITY])

        # All results should be security category
        for result in report.results:
            assert result.category == CheckCategory.SECURITY

    @pytest.mark.asyncio
    async def test_auth_check_passes(self):
        """Test auth check passes with OIDC."""
        checker = ReadinessChecker()
        config = {"oidc_issuer": "https://auth.example.com"}
        report = await checker.run_checks(config)

        auth_results = [r for r in report.results if r.name == "auth_configured"]
        assert auth_results[0].status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_auth_check_fails(self):
        """Test auth check fails without auth."""
        checker = ReadinessChecker()
        config = {}  # No auth configured
        report = await checker.run_checks(config)

        auth_results = [r for r in report.results if r.name == "auth_configured"]
        assert auth_results[0].status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_timeout_check(self):
        """Test timeout check."""
        checker = ReadinessChecker()

        # With timeout
        config = {"timeout": 30}
        report = await checker.run_checks(config)
        timeout_results = [r for r in report.results if r.name == "timeout_configured"]
        assert timeout_results[0].status == CheckStatus.PASS

        # Without timeout
        config = {}
        report = await checker.run_checks(config)
        timeout_results = [r for r in report.results if r.name == "timeout_configured"]
        assert timeout_results[0].status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_add_custom_check(self):
        """Test adding custom check."""
        checker = ReadinessChecker()

        def custom_check(config):
            return CheckResult(
                name="custom",
                status=CheckStatus.PASS,
                message="Custom check passed",
                category=CheckCategory.CONFIGURATION,
            )

        checker.add_check(
            ReadinessCheck(
                name="custom",
                description="Custom check",
                category=CheckCategory.CONFIGURATION,
                check_fn=custom_check,
            )
        )

        report = await checker.run_checks()

        custom_results = [r for r in report.results if r.name == "custom"]
        assert len(custom_results) == 1
        assert custom_results[0].status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_check_exception_handling(self):
        """Test exception handling in checks."""
        checker = ReadinessChecker()

        def failing_check(config):
            raise ValueError("Check failed unexpectedly")

        checker.add_check(
            ReadinessCheck(
                name="failing",
                description="Failing check",
                category=CheckCategory.CONFIGURATION,
                check_fn=failing_check,
            )
        )

        # Should not raise, but mark as failed
        report = await checker.run_checks()

        failing_results = [r for r in report.results if r.name == "failing"]
        assert len(failing_results) == 1
        assert failing_results[0].status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_rate_limiting_check(self):
        """Test rate limiting check."""
        checker = ReadinessChecker()

        # Without rate limiting
        report = await checker.run_checks({})
        rate_results = [r for r in report.results if r.name == "rate_limiting"]
        assert rate_results[0].status == CheckStatus.WARN

        # With rate limiting
        config = {"rate_limit": {"rpm": 60}}
        report = await checker.run_checks(config)
        rate_results = [r for r in report.results if r.name == "rate_limiting"]
        assert rate_results[0].status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_metrics_check(self):
        """Test metrics check."""
        checker = ReadinessChecker()

        # Without metrics
        report = await checker.run_checks({})
        metrics_results = [r for r in report.results if r.name == "metrics_enabled"]
        assert metrics_results[0].status == CheckStatus.WARN

        # With metrics
        config = {"telemetry": True}
        report = await checker.run_checks(config)
        metrics_results = [r for r in report.results if r.name == "metrics_enabled"]
        assert metrics_results[0].status == CheckStatus.PASS


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIP.value == "skip"


class TestCheckCategory:
    """Tests for CheckCategory enum."""

    def test_category_values(self):
        """Test category values."""
        assert CheckCategory.SECURITY.value == "security"
        assert CheckCategory.RELIABILITY.value == "reliability"
        assert CheckCategory.OBSERVABILITY.value == "observability"
        assert CheckCategory.PERFORMANCE.value == "performance"
        assert CheckCategory.COMPLIANCE.value == "compliance"
        assert CheckCategory.CONFIGURATION.value == "configuration"
