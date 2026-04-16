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
from fastagentic.reliability.rate_limit import RateLimit, RateLimitExceeded
from fastagentic.reliability.retry import RetryExhausted, RetryPolicy, with_retry
from fastagentic.reliability.timeout import Timeout, TimeoutExceeded, with_timeout

__all__ = [
    # Retry
    "RetryPolicy",
    "RetryExhausted",
    "with_retry",
    # Timeout
    "Timeout",
    "TimeoutExceeded",
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
    "RateLimitExceeded",
]
