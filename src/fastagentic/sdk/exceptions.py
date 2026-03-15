"""SDK exceptions."""

from __future__ import annotations

from typing import Any


class FastAgenticError(Exception):
    """Base exception for FastAgentic SDK errors.

    Attributes:
        message: Error message
        status_code: HTTP status code (if applicable)
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(FastAgenticError):
    """Authentication failed.

    Raised when:
    - API key is missing or invalid
    - Token has expired
    - Insufficient permissions
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(FastAgenticError):
    """Authorization failed.

    Raised when:
    - User lacks required permissions
    - Resource access denied
    - Scope requirements not met
    """

    def __init__(
        self,
        message: str = "Access denied",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=403, details=details)


class RateLimitError(FastAgenticError):
    """Rate limit exceeded.

    Attributes:
        retry_after: Seconds until rate limit resets
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after

    def __str__(self) -> str:
        base = super().__str__()
        if self.retry_after:
            return f"{base} (retry after {self.retry_after}s)"
        return base


class ValidationError(FastAgenticError):
    """Request validation failed.

    Raised when:
    - Required fields are missing
    - Field values are invalid
    - Input doesn't match schema
    """

    def __init__(
        self,
        message: str = "Validation error",
        errors: list[dict[str, Any]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if errors:
            details["errors"] = errors
        super().__init__(message, status_code=422, details=details)
        self.errors = errors or []


class NotFoundError(FastAgenticError):
    """Resource not found.

    Raised when:
    - Endpoint doesn't exist
    - Run ID is invalid
    - Resource has been deleted
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        super().__init__(message, status_code=404, details=details)


class TimeoutError(FastAgenticError):
    """Request timed out.

    Raised when:
    - Server didn't respond in time
    - Run exceeded timeout limit
    - Connection timed out
    """

    def __init__(
        self,
        message: str = "Request timed out",
        timeout: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if timeout:
            details["timeout"] = timeout
        super().__init__(message, status_code=408, details=details)
        self.timeout = timeout


class ServerError(FastAgenticError):
    """Server error.

    Raised when:
    - Internal server error
    - Service unavailable
    - Unexpected server response
    """

    def __init__(
        self,
        message: str = "Server error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, details=details)


class ConnectionError(FastAgenticError):
    """Connection error.

    Raised when:
    - Cannot connect to server
    - Network is unavailable
    - DNS resolution failed
    """

    def __init__(
        self,
        message: str = "Connection failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)


class StreamError(FastAgenticError):
    """Stream error.

    Raised when:
    - Stream connection lost
    - Invalid stream data
    - Stream parsing failed
    """

    def __init__(
        self,
        message: str = "Stream error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)


def raise_for_status(status_code: int, response_data: dict[str, Any]) -> None:
    """Raise appropriate exception based on status code.

    Args:
        status_code: HTTP status code
        response_data: Response data from server

    Raises:
        FastAgenticError: Appropriate exception for status code
    """
    message = response_data.get("message", response_data.get("error", ""))
    details = response_data.get("details", {})

    if status_code == 401:
        raise AuthenticationError(message or "Authentication failed", details)
    elif status_code == 403:
        raise AuthorizationError(message or "Access denied", details)
    elif status_code == 404:
        raise NotFoundError(message or "Not found", details=details)
    elif status_code == 408:
        raise TimeoutError(message or "Timeout", details=details)
    elif status_code == 422:
        raise ValidationError(
            message or "Validation error",
            errors=response_data.get("errors"),
            details=details,
        )
    elif status_code == 429:
        raise RateLimitError(
            message or "Rate limit exceeded",
            retry_after=response_data.get("retry_after"),
            details=details,
        )
    elif 500 <= status_code < 600:
        raise ServerError(
            message or "Server error",
            status_code=status_code,
            details=details,
        )
    elif status_code >= 400:
        raise FastAgenticError(
            message or f"Request failed with status {status_code}",
            status_code=status_code,
            details=details,
        )
