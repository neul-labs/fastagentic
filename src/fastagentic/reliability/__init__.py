"""Reliability patterns for FastAgentic.

Provides retry policies, circuit breakers, timeouts, and fallback chains
for building resilient agent applications.
"""

from fastagentic.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    with_circuit_breaker,
)
from fastagentic.reliability.fallback import FallbackChain, FallbackConfig, StrategyFallback
from fastagentic.reliability.rate_limit import RateLimit, RateLimitError
from fastagentic.reliability.retry import RetryError, RetryPolicy, with_retry
from fastagentic.reliability.timeout import Timeout, TimeoutError, with_timeout

__all__ = [
    # Retry
    "RetryPolicy",
    "RetryError",
    "with_retry",
    # Timeout
    "Timeout",
    "TimeoutError",
    "with_timeout",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "with_circuit_breaker",
    # Fallback
    "FallbackChain",
    "FallbackConfig",
    "StrategyFallback",
    # Rate Limit
    "RateLimit",
    "RateLimitError",
]
