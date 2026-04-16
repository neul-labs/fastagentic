"""Circuit breaker implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from fastagentic.sdk.exceptions import FastAgenticError

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(FastAgenticError):
    """Raised when circuit is open and request is rejected."""

    def __init__(
        self,
        message: str = "Circuit is open",
        reset_time: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if reset_time is not None:
            details["reset_time"] = reset_time
        super().__init__(message, status_code=503, details=details)
        self.reset_time = reset_time


@dataclass
class CircuitBreaker:
    """Circuit breaker for failing dependencies.

    Prevents cascading failures by stopping requests to failing services
    and allowing them to recover.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, all requests rejected
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker(
            failure_threshold=5,
            failure_window_ms=60000,
            reset_timeout_ms=30000,
        )

        @agent_endpoint(path="/external", circuit_breaker=breaker)
        async def call_external(input: Input) -> Output:
            ...
    """

    failure_threshold: int = 5
    failure_window_ms: int = 60000  # 1 minute
    reset_timeout_ms: int = 30000  # 30 seconds
    half_open_requests: int = 2

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: list[float] = field(default_factory=list, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get the current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get the number of recent failures."""
        return len(self._failures)

    def _clean_old_failures(self) -> None:
        """Remove failures outside the window."""
        cutoff = time.time() - (self.failure_window_ms / 1000.0)
        self._failures = [t for t in self._failures if t > cutoff]
        # Bound the list to prevent unbounded growth even if window is large
        max_failures = self.failure_threshold * 10
        if len(self._failures) > max_failures:
            self._failures = self._failures[-max_failures:]

    def _should_open(self) -> bool:
        """Check if circuit should open."""
        self._clean_old_failures()
        return len(self._failures) >= self.failure_threshold

    def _should_close(self) -> bool:
        """Check if circuit should close (in half-open state)."""
        return self._half_open_successes >= self.half_open_requests

    def _can_attempt(self) -> bool:
        """Check if a request can be attempted."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.reset_timeout_ms / 1000.0:
                    return True
            return False

        # HALF_OPEN - allow limited requests
        return True

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: The async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The function's return value

        Raises:
            CircuitOpenError: If the circuit is open
        """
        async with self._lock:
            if not self._can_attempt():
                raise CircuitOpenError(
                    "Circuit is open",
                    reset_time=self._last_failure_time,
                )

            # Transition to half-open if coming from open
            if self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - handle state transitions
            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._should_close():
                        self._state = CircuitState.CLOSED
                        self._failures.clear()

            return result

        except Exception:
            # Failure - record and possibly open circuit
            async with self._lock:
                self._failures.append(time.time())
                self._last_failure_time = time.time()

                if self._state == CircuitState.HALF_OPEN:
                    # Any failure in half-open returns to open
                    self._state = CircuitState.OPEN
                elif self._should_open():
                    self._state = CircuitState.OPEN

            raise

    async def async_reset(self) -> None:
        """Reset the circuit breaker to closed state (async, thread-safe)."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._last_failure_time = None
            self._half_open_successes = 0

    async def async_trip(self) -> None:
        """Manually trip the circuit to open state (async, thread-safe)."""
        async with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()

    def reset(self) -> None:
        """Reset the circuit breaker to closed state.

        Note: This is a synchronous method meant to be called
        from outside the async context. For thread-safe reset
        during operation, use the state transition mechanism.
        """
        self._state = CircuitState.CLOSED
        self._failures.clear()
        self._last_failure_time = None
        self._half_open_successes = 0

    def trip(self) -> None:
        """Manually trip the circuit to open state.

        Note: This is a synchronous method meant to be called
        from outside the async context.
        """
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()


def with_circuit_breaker(
    failure_threshold: int = 5,
    failure_window_ms: int = 60000,
    reset_timeout_ms: int = 30000,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add circuit breaker to a function.

    Example:
        @with_circuit_breaker(failure_threshold=5)
        async def call_api():
            ...
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        failure_window_ms=failure_window_ms,
        reset_timeout_ms=reset_timeout_ms,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.execute(func, *args, **kwargs)

        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper  # type: ignore

    return decorator
