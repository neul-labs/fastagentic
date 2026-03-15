# Production Readiness

FastAgentic includes a production readiness checker to validate your deployment configuration.

## Quick Start

```python
from fastagentic import ReadinessChecker

checker = ReadinessChecker()
report = await checker.run_checks()

print(f"Ready: {report.is_ready}")
print(f"Score: {report.score}/100")

for result in report.get_failures():
    print(f"FAIL: {result.name} - {result.recommendation}")
```

## Readiness Checker

### Basic Usage

```python
from fastagentic import ReadinessChecker

checker = ReadinessChecker()

# Run all checks
report = await checker.run_checks()

# Run with configuration
config = {
    "auth": {"enabled": True},
    "timeout": 30,
    "telemetry": True,
}
report = await checker.run_checks(config)
```

### Check Categories

```python
from fastagentic.ops.readiness import CheckCategory

# Available categories
CheckCategory.SECURITY        # Authentication, HTTPS, secrets
CheckCategory.RELIABILITY     # Timeouts, retries, rate limiting
CheckCategory.OBSERVABILITY   # Logging, metrics, health checks
CheckCategory.PERFORMANCE     # Resource usage, optimization
CheckCategory.COMPLIANCE      # PII detection, audit logging
CheckCategory.CONFIGURATION   # General configuration
```

### Run Specific Categories

```python
from fastagentic.ops.readiness import CheckCategory

report = await checker.run_checks(
    config,
    categories=[CheckCategory.SECURITY, CheckCategory.RELIABILITY],
)
```

## Built-in Checks

### Security Checks

| Check | Pass Condition |
|-------|----------------|
| `auth_configured` | Auth enabled or OIDC issuer set |
| `https_enforced` | HTTPS enabled in production |
| `secrets_secured` | No secrets in config |

### Reliability Checks

| Check | Pass Condition |
|-------|----------------|
| `timeout_configured` | Request timeout is set |
| `retry_policy` | Retry policy configured |
| `rate_limiting` | Rate limits defined |

### Observability Checks

| Check | Pass Condition |
|-------|----------------|
| `logging_configured` | Logging is enabled |
| `metrics_enabled` | Telemetry enabled |
| `health_endpoint` | Health check endpoint exists |

### Compliance Checks

| Check | Pass Condition |
|-------|----------------|
| `pii_detection` | PII detection enabled |
| `audit_logging` | Audit logging configured |

## Check Results

### CheckResult

```python
from fastagentic import CheckResult, CheckStatus

# Result properties
result.name           # Check name
result.status         # CheckStatus enum
result.message        # Description
result.category       # CheckCategory
result.recommendation # Fix suggestion (if failed)
```

### CheckStatus

```python
from fastagentic import CheckStatus

CheckStatus.PASS    # Check passed
CheckStatus.WARN    # Warning (doesn't block readiness)
CheckStatus.FAIL    # Check failed (blocks readiness)
CheckStatus.SKIP    # Check was skipped
```

## Readiness Report

### Report Properties

```python
report = await checker.run_checks()

# Summary
report.total        # Total checks run
report.passed       # Passed count
report.failed       # Failed count
report.warned       # Warning count
report.skipped      # Skipped count

# Status
report.is_ready     # True if no failures
report.score        # 0-100 readiness score

# Results
report.results      # All CheckResult objects
```

### Filtering Results

```python
# Get failures only
failures = report.get_failures()

# Get warnings only
warnings = report.get_warnings()

# Get by category
security_results = report.get_by_category(CheckCategory.SECURITY)
```

### Serialization

```python
data = report.to_dict()

# Returns:
# {
#     "is_ready": True,
#     "score": 85,
#     "summary": {"total": 10, "passed": 8, "warned": 2, ...},
#     "results": [...],
# }
```

## Custom Checks

### Adding Custom Checks

```python
from fastagentic import ReadinessChecker, ReadinessCheck, CheckResult, CheckStatus
from fastagentic.ops.readiness import CheckCategory

checker = ReadinessChecker()

def check_database(config):
    """Check database connection."""
    db_url = config.get("database_url")

    if not db_url:
        return CheckResult(
            name="database_configured",
            status=CheckStatus.FAIL,
            message="Database URL not configured",
            category=CheckCategory.CONFIGURATION,
            recommendation="Set DATABASE_URL environment variable",
        )

    return CheckResult(
        name="database_configured",
        status=CheckStatus.PASS,
        message="Database configured",
        category=CheckCategory.CONFIGURATION,
    )

# Add custom check
checker.add_check(ReadinessCheck(
    name="database_configured",
    description="Verify database is configured",
    category=CheckCategory.CONFIGURATION,
    check_fn=check_database,
))

report = await checker.run_checks({"database_url": "postgres://..."})
```

### Async Custom Checks

```python
async def check_external_service(config):
    """Check external service connectivity."""
    try:
        response = await http_client.get(config.get("service_url"))
        if response.status_code == 200:
            return CheckResult(
                name="external_service",
                status=CheckStatus.PASS,
                message="External service reachable",
                category=CheckCategory.RELIABILITY,
            )
    except Exception as e:
        pass

    return CheckResult(
        name="external_service",
        status=CheckStatus.WARN,
        message="External service unreachable",
        category=CheckCategory.RELIABILITY,
        recommendation="Verify service URL and network connectivity",
    )

checker.add_check(ReadinessCheck(
    name="external_service",
    description="Check external service connectivity",
    category=CheckCategory.RELIABILITY,
    check_fn=check_external_service,
    is_async=True,
))
```

## Exception Handling

Checks that raise exceptions are marked as failed:

```python
def risky_check(config):
    raise ValueError("Something went wrong")

checker.add_check(ReadinessCheck(
    name="risky",
    description="A risky check",
    category=CheckCategory.CONFIGURATION,
    check_fn=risky_check,
))

report = await checker.run_checks()
# The check will show as FAIL with the exception message
```

## Integration Examples

### CI/CD Pipeline

```python
import sys
from fastagentic import ReadinessChecker

async def check_deployment_readiness():
    checker = ReadinessChecker()
    config = load_config_from_env()

    report = await checker.run_checks(config)

    print(f"Readiness Score: {report.score}/100")

    if not report.is_ready:
        print("\nFailures:")
        for result in report.get_failures():
            print(f"  - {result.name}: {result.recommendation}")
        sys.exit(1)

    print("Deployment ready!")
    sys.exit(0)
```

### Health Endpoint

```python
from fastapi import FastAPI, HTTPException
from fastagentic import ReadinessChecker

app = FastAPI()
checker = ReadinessChecker()

@app.get("/ready")
async def readiness_check():
    config = get_current_config()
    report = await checker.run_checks(config)

    if not report.is_ready:
        raise HTTPException(
            status_code=503,
            detail=report.to_dict(),
        )

    return {"status": "ready", "score": report.score}
```

### Startup Validation

```python
from fastagentic import App, ReadinessChecker

app = App()

@app.on_event("startup")
async def validate_readiness():
    checker = ReadinessChecker()
    report = await checker.run_checks(app.config)

    if not report.is_ready:
        failures = [r.name for r in report.get_failures()]
        raise RuntimeError(
            f"App not ready for production: {failures}"
        )

    if report.warnings:
        for warning in report.get_warnings():
            logger.warning(f"Readiness warning: {warning.name}")
```

## Configuration Example

```python
# Comprehensive configuration for all checks
config = {
    # Security
    "auth": {"enabled": True},
    "oidc_issuer": "https://auth.example.com",
    "https": {"enabled": True},

    # Reliability
    "timeout": 30,
    "retry_policy": {"max_attempts": 3, "backoff": "exponential"},
    "rate_limit": {"rpm": 60, "rpd": 10000},

    # Observability
    "logging": {"level": "INFO", "format": "json"},
    "telemetry": True,
    "health_endpoint": "/health",

    # Compliance
    "pii_detection": {"enabled": True},
    "audit": {"enabled": True},
}

report = await checker.run_checks(config)
print(f"Score: {report.score}/100")  # Should be 100 with full config
```

## Best Practices

1. **Run checks in CI/CD**: Validate configuration before deployment
2. **Monitor score over time**: Track readiness score as a metric
3. **Don't ignore warnings**: Warnings indicate areas for improvement
4. **Add domain-specific checks**: Create checks for your specific requirements
5. **Use categories**: Focus on specific areas when debugging issues
