"""Fallback chain implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class FallbackConfig:
    """Configuration for a fallback option."""

    model: str
    on: list[str] = field(default_factory=lambda: ["rate_limit", "timeout"])
    weight: float = 1.0


@dataclass
class FallbackChain:
    """Fallback chain for graceful degradation.

    Automatically falls back to alternative models or strategies
    when the primary option fails.

    Example:
        fallback = FallbackChain(
            primary="gpt-4o",
            fallbacks=[
                {"model": "gpt-4o-mini", "on": ["rate_limit", "timeout"]},
                {"model": "gpt-3.5-turbo", "on": ["rate_limit", "timeout", "server_error"]},
            ],
        )

        app = App(model_fallback=fallback)
    """

    primary: str
    fallbacks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Parse fallback configs."""
        self._fallback_configs = [
            FallbackConfig(
                model=f["model"],
                on=f.get("on", ["rate_limit", "timeout"]),
                weight=f.get("weight", 1.0),
            )
            for f in self.fallbacks
        ]

    def get_fallback_for_error(self, error: Exception) -> str | None:
        """Get the appropriate fallback model for an error.

        Args:
            error: The exception that occurred

        Returns:
            Fallback model name or None if no fallback applies
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        for config in self._fallback_configs:
            for condition in config.on:
                condition_lower = condition.lower()
                if condition_lower in error_str or condition_lower in error_type:
                    return config.model

                # Check for common patterns
                if condition == "rate_limit" and ("429" in error_str or "rate" in error_str):
                    return config.model
                if condition == "timeout" and "timeout" in error_str:
                    return config.model
                if condition == "server_error" and any(
                    code in error_str for code in ["500", "502", "503", "504"]
                ):
                    return config.model

        return None

    def get_all_models(self) -> list[str]:
        """Get all models in the chain (primary + fallbacks).

        Returns:
            List of model names in order
        """
        return [self.primary] + [f.model for f in self._fallback_configs]

    async def execute_with_fallback(
        self,
        func: Callable[..., T],
        *args: Any,
        model_kwarg: str = "model",
        **kwargs: Any,
    ) -> T:
        """Execute a function with automatic fallback.

        Tries the primary model first, then falls back to alternatives
        if appropriate errors occur.

        Args:
            func: The async function to execute
            *args: Positional arguments
            model_kwarg: Name of the model keyword argument
            **kwargs: Keyword arguments

        Returns:
            The function's return value

        Raises:
            Exception: If all models fail
        """
        models = self.get_all_models()
        last_error: Exception | None = None

        for model in models:
            try:
                kwargs[model_kwarg] = model
                import asyncio

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_error = e
                fallback = self.get_fallback_for_error(e)
                if fallback is None:
                    # Error doesn't trigger fallback
                    raise

                # Try next model
                continue

        # All models failed
        if last_error:
            raise last_error
        raise RuntimeError("No models available")


@dataclass
class StrategyFallback:
    """Fallback between different strategies (not just models).

    Example:
        fallback = StrategyFallback(
            strategies=[
                {"name": "primary", "func": primary_func},
                {"name": "fallback", "func": fallback_func, "on": ["error"]},
            ],
        )
    """

    strategies: list[dict[str, Any]] = field(default_factory=list)

    async def execute(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute strategies with fallback.

        Tries each strategy in order until one succeeds.

        Args:
            *args: Arguments to pass to strategy functions
            **kwargs: Keyword arguments

        Returns:
            Result from the first successful strategy

        Raises:
            Exception: If all strategies fail
        """
        last_error: Exception | None = None

        for strategy in self.strategies:
            func = strategy.get("func")
            if not func:
                continue

            try:
                import asyncio

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_error = e
                on_conditions = strategy.get("on", ["error"])
                error_str = str(e).lower()

                # Check if this error triggers fallback to next strategy
                should_continue = (
                    any(cond.lower() in error_str for cond in on_conditions)
                    or "error" in on_conditions
                )

                if not should_continue:
                    raise

        if last_error:
            raise last_error
        raise RuntimeError("No strategies configured")
