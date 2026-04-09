"""Rate limiting implementation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit: int,
        window_seconds: int,
        retry_after: float,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after


@dataclass
class RateLimit:
    """Simple rate limiter.

    Limits requests per minute (RPM) and/or tokens per minute (TPM)
    with optional per-user/tenant scoping.

    Example:
        rate_limit = RateLimit(
            rpm=60,
            tpm=100000,
            by="user",  # or "tenant", "ip", "global"
        )

        app = App(rate_limit=rate_limit)
    """

    rpm: int | None = None  # Requests per minute
    tpm: int | None = None  # Tokens per minute
    by: str = "global"  # "global", "user", "tenant", "ip"
    window_seconds: int = 60

    # Internal state
    _request_counts: dict[str, list[float]] = field(default_factory=dict, init=False)
    _token_counts: dict[str, list[tuple[float, int]]] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def _get_key(self, identifier: str | None = None) -> str:
        """Get the rate limit key based on scope."""
        if self.by == "global":
            return "global"
        return identifier or "unknown"

    def _clean_old_entries(
        self,
        entries: list[Any],
        window_seconds: int,
    ) -> list[Any]:
        """Remove entries outside the window."""
        cutoff = time.time() - window_seconds
        return [e for e in entries if (e[0] if isinstance(e, tuple) else e) > cutoff]

    async def check_request(
        self,
        identifier: str | None = None,
    ) -> bool:
        """Check if a request is allowed.

        Args:
            identifier: User/tenant/IP identifier (based on 'by' setting)

        Returns:
            True if request is allowed

        Raises:
            RateLimitError: If rate limit exceeded
        """
        if self.rpm is None:
            return True

        key = self._get_key(identifier)

        async with self._lock:
            if key not in self._request_counts:
                self._request_counts[key] = []

            # Clean old entries
            self._request_counts[key] = self._clean_old_entries(
                self._request_counts[key],
                self.window_seconds,
            )

            # Check limit
            if len(self._request_counts[key]) >= self.rpm:
                # Calculate retry after
                oldest = self._request_counts[key][0]
                retry_after = oldest + self.window_seconds - time.time()

                raise RateLimitError(
                    f"Rate limit exceeded: {self.rpm} requests per {self.window_seconds}s",
                    limit=self.rpm,
                    window_seconds=self.window_seconds,
                    retry_after=max(0, retry_after),
                )

            # Record request
            self._request_counts[key].append(time.time())
            return True

    async def check_tokens(
        self,
        token_count: int,
        identifier: str | None = None,
    ) -> bool:
        """Check if tokens are allowed.

        Args:
            token_count: Number of tokens to check
            identifier: User/tenant/IP identifier

        Returns:
            True if tokens are allowed

        Raises:
            RateLimitError: If token limit exceeded
        """
        if self.tpm is None:
            return True

        key = self._get_key(identifier)

        async with self._lock:
            if key not in self._token_counts:
                self._token_counts[key] = []

            # Clean old entries
            self._token_counts[key] = [
                (t, c) for t, c in self._token_counts[key] if t > time.time() - self.window_seconds
            ]

            # Sum current tokens
            current_tokens = sum(c for _, c in self._token_counts[key])

            if current_tokens + token_count > self.tpm:
                # Calculate retry after
                if self._token_counts[key]:
                    oldest = self._token_counts[key][0][0]
                    retry_after = oldest + self.window_seconds - time.time()
                else:
                    retry_after = self.window_seconds

                raise RateLimitError(
                    f"Token rate limit exceeded: {self.tpm} tokens per {self.window_seconds}s",
                    limit=self.tpm,
                    window_seconds=self.window_seconds,
                    retry_after=max(0, retry_after),
                )

            # Record tokens
            self._token_counts[key].append((time.time(), token_count))
            return True

    async def record_tokens(
        self,
        token_count: int,
        identifier: str | None = None,
    ) -> None:
        """Record token usage after a request.

        Use this when you don't know token count upfront.

        Args:
            token_count: Number of tokens used
            identifier: User/tenant/IP identifier
        """
        if self.tpm is None:
            return

        key = self._get_key(identifier)

        async with self._lock:
            if key not in self._token_counts:
                self._token_counts[key] = []
            self._token_counts[key].append((time.time(), token_count))

    def get_usage(
        self,
        identifier: str | None = None,
    ) -> dict[str, Any]:
        """Get current usage statistics.

        Args:
            identifier: User/tenant/IP identifier

        Returns:
            Dict with current usage stats
        """
        key = self._get_key(identifier)

        # Clean and count
        requests = self._clean_old_entries(
            self._request_counts.get(key, []),
            self.window_seconds,
        )
        tokens = sum(
            c for t, c in self._token_counts.get(key, []) if t > time.time() - self.window_seconds
        )

        return {
            "requests": len(requests),
            "requests_limit": self.rpm,
            "tokens": tokens,
            "tokens_limit": self.tpm,
            "window_seconds": self.window_seconds,
        }

    def reset(self, identifier: str | None = None) -> None:
        """Reset rate limit counters.

        Args:
            identifier: User/tenant/IP identifier (None resets all)
        """
        if identifier is None:
            self._request_counts.clear()
            self._token_counts.clear()
        else:
            key = self._get_key(identifier)
            self._request_counts.pop(key, None)
            self._token_counts.pop(key, None)
