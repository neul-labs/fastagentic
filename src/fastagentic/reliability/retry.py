"""Retry policy implementation."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

    Supports exponential, linear, and fixed backoff strategies with
    optional jitter and configurable retry conditions.

    Example:
        policy = RetryPolicy(
            max_attempts=3,
            backoff="exponential",
            initial_delay_ms=1000,
            jitter=True,
            retry_on=["rate_limit", "timeout"],
        )

        @agent_endpoint(path="/analyze", retry=policy)
        async def analyze(input: Input) -> Output:
            ...
    """

    max_attempts: int = 3
    backoff: str = "exponential"  # "exponential", "linear", "fixed"
    initial_delay_ms: int = 1000
    max_delay_ms: int = 60000
    multiplier: float = 2.0
    jitter: bool = True
    retry_on: list[str] = field(default_factory=lambda: ["rate_limit", "timeout", "server_error"])

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: The attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        if self.backoff == "exponential":
            delay_ms = self.initial_delay_ms * (self.multiplier ** (attempt - 1))
        elif self.backoff == "linear":
            delay_ms = self.initial_delay_ms * attempt
        else:  # fixed
            delay_ms = self.initial_delay_ms

        # Cap at max delay
        delay_ms = min(delay_ms, self.max_delay_ms)

        # Add jitter (±25%)
        if self.jitter:
            jitter_factor = 0.75 + random.random() * 0.5
            delay_ms *= jitter_factor

        return delay_ms / 1000.0  # Convert to seconds

    def should_retry(self, error: Exception) -> bool:
        """Check if an error should trigger a retry.

        Args:
            error: The exception that occurred

        Returns:
            True if the error should be retried
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        for condition in self.retry_on:
            condition_lower = condition.lower()
            if condition_lower in error_str or condition_lower in error_type:
                return True

            # Check for common HTTP status codes
            if condition == "rate_limit" and ("429" in error_str or "rate" in error_str):
                return True
            if condition == "timeout" and ("timeout" in error_str or "timed out" in error_str):
                return True
            if condition == "server_error" and any(
                code in error_str for code in ["500", "502", "503", "504"]
            ):
                return True

        return False

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with retry logic.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The function's return value

        Raises:
            RetryError: If all attempts are exhausted
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_error = e

                if attempt >= self.max_attempts:
                    break

                if not self.should_retry(e):
                    raise

                delay = self.get_delay(attempt)
                await asyncio.sleep(delay)

        raise RetryError(
            f"All {self.max_attempts} retry attempts exhausted",
            attempts=self.max_attempts,
            last_error=last_error,
        )


def with_retry(
    max_attempts: int = 3,
    backoff: str = "exponential",
    initial_delay_ms: int = 1000,
    retry_on: list[str] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function.

    Example:
        @with_retry(max_attempts=3, backoff="exponential")
        async def call_api():
            ...
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff,
        initial_delay_ms=initial_delay_ms,
        retry_on=retry_on or ["rate_limit", "timeout", "server_error"],
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await policy.execute(func, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator
