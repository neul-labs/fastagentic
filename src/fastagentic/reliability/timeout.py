"""Timeout implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation times out."""

    def __init__(self, message: str, timeout_ms: int) -> None:
        super().__init__(message)
        self.timeout_ms = timeout_ms


@dataclass
class Timeout:
    """Configuration for timeout behavior.

    Supports multiple timeout levels for different operation types.

    Example:
        timeout = Timeout(
            total_ms=300000,      # 5 minutes total
            llm_call_ms=60000,    # 1 minute per LLM call
            tool_call_ms=30000,   # 30 seconds per tool
            checkpoint_ms=5000,   # 5 seconds for checkpoints
        )

        @agent_endpoint(path="/analyze", timeout=timeout)
        async def analyze(input: Input) -> Output:
            ...
    """

    total_ms: int = 300000  # 5 minutes
    llm_call_ms: int = 60000  # 1 minute
    tool_call_ms: int = 30000  # 30 seconds
    checkpoint_ms: int = 5000  # 5 seconds

    @property
    def total_seconds(self) -> float:
        """Total timeout in seconds."""
        return self.total_ms / 1000.0

    @property
    def llm_call_seconds(self) -> float:
        """LLM call timeout in seconds."""
        return self.llm_call_ms / 1000.0

    @property
    def tool_call_seconds(self) -> float:
        """Tool call timeout in seconds."""
        return self.tool_call_ms / 1000.0

    @property
    def checkpoint_seconds(self) -> float:
        """Checkpoint timeout in seconds."""
        return self.checkpoint_ms / 1000.0

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout_ms: int | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute a function with timeout.

        Args:
            func: The async function to execute
            *args: Positional arguments
            timeout_ms: Override timeout (uses total_ms if not specified)
            **kwargs: Keyword arguments

        Returns:
            The function's return value

        Raises:
            TimeoutError: If the operation times out
        """
        timeout = timeout_ms or self.total_ms
        timeout_seconds = timeout / 1000.0

        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            else:
                # For sync functions, run in executor with timeout
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=timeout_seconds,
                )

        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Operation timed out after {timeout}ms",
                timeout_ms=timeout,
            ) from e

    async def execute_llm_call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an LLM call with LLM-specific timeout."""
        return await self.execute(func, *args, timeout_ms=self.llm_call_ms, **kwargs)

    async def execute_tool_call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a tool call with tool-specific timeout."""
        return await self.execute(func, *args, timeout_ms=self.tool_call_ms, **kwargs)

    async def execute_checkpoint(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a checkpoint operation with checkpoint-specific timeout."""
        return await self.execute(func, *args, timeout_ms=self.checkpoint_ms, **kwargs)


def with_timeout(
    timeout_ms: int = 60000,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to a function.

    Example:
        @with_timeout(timeout_ms=30000)
        async def call_api():
            ...
    """
    timeout = Timeout(total_ms=timeout_ms)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await timeout.execute(func, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator
