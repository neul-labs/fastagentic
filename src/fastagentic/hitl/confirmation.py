"""Confirmation dialogs for FastAgentic."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar


class ConfirmationType(str, Enum):
    """Types of confirmation dialogs."""

    SIMPLE = "simple"  # Yes/No
    DESTRUCTIVE = "destructive"  # Requires typing confirmation text
    MULTI_STEP = "multi_step"  # Multiple confirmations required
    TIMEOUT = "timeout"  # Auto-proceeds after timeout


@dataclass
class ConfirmationRequest:
    """A request for user confirmation.

    Attributes:
        id: Unique identifier
        message: Message to display to user
        confirmation_type: Type of confirmation required
        confirmation_text: Text user must type for DESTRUCTIVE type
        options: Available response options
        timeout_seconds: Timeout for TIMEOUT type
        metadata: Additional context
        created_at: When created
    """

    message: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confirmation_type: ConfirmationType = ConfirmationType.SIMPLE
    confirmation_text: str | None = None
    options: list[str] = field(default_factory=lambda: ["Yes", "No"])
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ConfirmationResponse:
    """Response to a confirmation request."""

    request_id: str
    confirmed: bool
    response: str = ""
    timestamp: float = field(default_factory=time.time)


# Type for confirmation handlers
ConfirmationHandler = Callable[[ConfirmationRequest], "asyncio.Future[ConfirmationResponse]"]

# Global confirmation handler
_confirmation_handler: ConfirmationHandler | None = None


def set_confirmation_handler(handler: ConfirmationHandler) -> None:
    """Set the global confirmation handler.

    This handler will be called when confirmation is needed.
    It should return a Future that resolves to a ConfirmationResponse.

    Args:
        handler: Async function to handle confirmation requests
    """
    global _confirmation_handler
    _confirmation_handler = handler


def get_confirmation_handler() -> ConfirmationHandler | None:
    """Get the current confirmation handler."""
    return _confirmation_handler


async def request_confirmation(
    message: str,
    *,
    confirmation_type: ConfirmationType = ConfirmationType.SIMPLE,
    confirmation_text: str | None = None,
    options: list[str] | None = None,
    timeout_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConfirmationResponse:
    """Request confirmation from a user.

    Args:
        message: Message to display
        confirmation_type: Type of confirmation
        confirmation_text: Required text for DESTRUCTIVE type
        options: Available options (default: Yes/No)
        timeout_seconds: Timeout for auto-proceed
        metadata: Additional context

    Returns:
        ConfirmationResponse

    Raises:
        RuntimeError: If no confirmation handler is set
        asyncio.TimeoutError: If timeout exceeded
    """
    handler = get_confirmation_handler()
    if handler is None:
        # Default behavior: auto-confirm in non-interactive mode
        return ConfirmationResponse(
            request_id=str(uuid.uuid4()),
            confirmed=True,
            response="auto-confirmed (no handler)",
        )

    request = ConfirmationRequest(
        message=message,
        confirmation_type=confirmation_type,
        confirmation_text=confirmation_text,
        options=options or ["Yes", "No"],
        timeout_seconds=timeout_seconds,
        metadata=metadata or {},
    )

    future = handler(request)

    if timeout_seconds:
        try:
            response = await asyncio.wait_for(future, timeout=timeout_seconds)
            return response
        except asyncio.TimeoutError:
            if confirmation_type == ConfirmationType.TIMEOUT:
                # Auto-confirm on timeout for TIMEOUT type
                return ConfirmationResponse(
                    request_id=request.id,
                    confirmed=True,
                    response="auto-confirmed (timeout)",
                )
            raise

    return await future


P = ParamSpec("P")
T = TypeVar("T")


def require_confirmation(
    message: str | Callable[..., str] = "Are you sure?",
    *,
    confirmation_type: ConfirmationType = ConfirmationType.SIMPLE,
    confirmation_text: str | None = None,
    on_cancel: Callable[[], Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to require confirmation before executing a function.

    Example:
        @require_confirmation("Delete all records?")
        async def delete_all():
            # This only runs if user confirms
            pass

        @require_confirmation(
            lambda resource: f"Delete {resource}?",
            confirmation_type=ConfirmationType.DESTRUCTIVE,
            confirmation_text="DELETE",
        )
        async def delete_resource(resource: str):
            pass

    Args:
        message: Confirmation message (string or callable)
        confirmation_type: Type of confirmation
        confirmation_text: Required text for DESTRUCTIVE type
        on_cancel: Function to call if cancelled

    Returns:
        Decorated function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate message
            if callable(message):
                msg = message(*args, **kwargs)
            else:
                msg = message

            # Request confirmation
            response = await request_confirmation(
                message=msg,
                confirmation_type=confirmation_type,
                confirmation_text=confirmation_text,
            )

            if not response.confirmed:
                if on_cancel:
                    return on_cancel()
                raise ConfirmationCancelledError(f"Action cancelled: {msg}")

            # For DESTRUCTIVE, verify the confirmation text
            if (
                confirmation_type == ConfirmationType.DESTRUCTIVE
                and response.response != confirmation_text
            ):
                raise ConfirmationCancelledError(
                    f"Invalid confirmation text. Expected '{confirmation_text}'"
                )

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class ConfirmationCancelledError(Exception):
    """Raised when a confirmation is cancelled."""

    pass


class InteractiveConfirmationHandler:
    """Handler for interactive confirmation in CLI/REPL environments.

    Example:
        handler = InteractiveConfirmationHandler()
        set_confirmation_handler(handler.handle)

        # Now confirmation requests will prompt in the terminal
    """

    def __init__(
        self,
        input_func: Callable[[str], str] | None = None,
        output_func: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize interactive handler.

        Args:
            input_func: Function to get input (default: input())
            output_func: Function to print output (default: print())
        """
        self._input = input_func or input
        self._output = output_func or print

    def handle(self, request: ConfirmationRequest) -> asyncio.Future[ConfirmationResponse]:
        """Handle a confirmation request interactively."""
        future: asyncio.Future[ConfirmationResponse] = asyncio.get_event_loop().create_future()

        async def _prompt() -> None:
            self._output(f"\n{request.message}")

            if request.confirmation_type == ConfirmationType.DESTRUCTIVE:
                self._output(f"Type '{request.confirmation_text}' to confirm:")
                response_text = self._input("> ")

                confirmed = response_text == request.confirmation_text
                future.set_result(
                    ConfirmationResponse(
                        request_id=request.id,
                        confirmed=confirmed,
                        response=response_text,
                    )
                )

            else:
                options_str = "/".join(request.options)
                self._output(f"[{options_str}]")
                response_text = self._input("> ")

                confirmed = response_text.lower() in ["yes", "y", "true", "1"]
                future.set_result(
                    ConfirmationResponse(
                        request_id=request.id,
                        confirmed=confirmed,
                        response=response_text,
                    )
                )

        asyncio.create_task(_prompt())
        return future


class QueuedConfirmationHandler:
    """Handler that queues confirmations for external processing.

    Useful for web/API-based confirmation flows.

    Example:
        handler = QueuedConfirmationHandler()
        set_confirmation_handler(handler.handle)

        # In your API:
        @app.get("/confirmations/pending")
        async def get_pending():
            return handler.get_pending()

        @app.post("/confirmations/{id}/respond")
        async def respond(id: str, confirmed: bool):
            handler.respond(id, confirmed)
    """

    def __init__(self) -> None:
        self._pending: dict[str, tuple[ConfirmationRequest, asyncio.Future]] = {}

    def handle(self, request: ConfirmationRequest) -> asyncio.Future[ConfirmationResponse]:
        """Handle a confirmation request by queueing it."""
        future: asyncio.Future[ConfirmationResponse] = asyncio.get_event_loop().create_future()
        self._pending[request.id] = (request, future)
        return future

    def get_pending(self) -> list[ConfirmationRequest]:
        """Get all pending confirmation requests."""
        return [req for req, _ in self._pending.values()]

    def respond(
        self,
        request_id: str,
        confirmed: bool,
        response_text: str = "",
    ) -> bool:
        """Respond to a pending confirmation.

        Args:
            request_id: Request to respond to
            confirmed: Whether confirmed
            response_text: Response text

        Returns:
            True if request was found and responded to
        """
        if request_id not in self._pending:
            return False

        _, future = self._pending.pop(request_id)

        response = ConfirmationResponse(
            request_id=request_id,
            confirmed=confirmed,
            response=response_text,
        )

        if not future.done():
            future.set_result(response)

        return True
