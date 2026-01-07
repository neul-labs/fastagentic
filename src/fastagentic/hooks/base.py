"""Base hook system for FastAgentic.

Hooks provide lifecycle interception points for observability,
guardrails, memory, and custom logic.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from fastagentic.app import App
    from fastagentic.context import AgentContext, UsageInfo, UserInfo

# Registry for hook functions
_hook_registry: dict[str, list[Callable[..., Any]]] = {}

T = TypeVar("T")


def reset_hooks() -> None:
    """Reset the hook registry.

    This is primarily useful for testing to ensure a clean state
    between test cases.
    """
    global _hook_registry
    _hook_registry = {}


class HookResultAction(str, Enum):
    """Action to take after a hook executes."""

    PROCEED = "proceed"  # Continue execution
    MODIFY = "modify"  # Continue with modified data
    SKIP = "skip"  # Skip the operation
    REJECT = "reject"  # Reject the request
    RETRY = "retry"  # Retry the operation


@dataclass
class HookResult:
    """Result from a hook execution.

    Hooks can control execution flow by returning different result types.
    """

    action: HookResultAction = HookResultAction.PROCEED
    data: Any = None
    message: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    @classmethod
    def proceed(cls) -> HookResult:
        """Continue execution normally."""
        return cls(action=HookResultAction.PROCEED)

    @classmethod
    def modify(cls, data: Any) -> HookResult:
        """Continue with modified data."""
        return cls(action=HookResultAction.MODIFY, data=data)

    @classmethod
    def skip(cls, message: str | None = None) -> HookResult:
        """Skip the current operation."""
        return cls(action=HookResultAction.SKIP, message=message)

    @classmethod
    def reject(cls, message: str) -> HookResult:
        """Reject the request."""
        return cls(action=HookResultAction.REJECT, message=message)

    @classmethod
    def retry(cls, max_retries: int = 3) -> HookResult:
        """Retry the operation."""
        return cls(action=HookResultAction.RETRY, max_retries=max_retries)


@dataclass
class HookContext:
    """Context passed to hook functions.

    Provides access to request data, run metadata, and utilities
    for modifying execution.
    """

    # Run information
    run_id: str
    endpoint: str

    # Request/Response data (depending on hook type)
    request: Any = None
    response: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    # User and usage
    user: UserInfo | None = None
    usage: UsageInfo | None = None

    # Tool/LLM specific
    tool_name: str | None = None
    tool_input: Any = None
    tool_output: Any = None
    model: str | None = None

    # Node specific (LangGraph)
    node_name: str | None = None
    node_input: Any = None
    node_output: Any = None

    # Error handling
    error: Exception | None = None
    retry_count: int = 0

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Memory access
    memory_content: str | None = None
    memory_query: str | None = None
    memory_results: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_agent_context(cls, ctx: AgentContext) -> HookContext:
        """Create a HookContext from an AgentContext."""
        return cls(
            run_id=ctx.run_id,
            endpoint=ctx.run.endpoint,
            user=ctx.user,
            usage=ctx.usage,
            metadata=ctx.metadata.copy(),
        )


class Hook(ABC):
    """Base class for hooks.

    Hooks intercept various lifecycle points in agent execution.
    Subclass this and implement the hook methods you need.

    Example:
        class MyHook(Hook):
            async def on_request(self, _ctx: HookContext) -> HookResult:
                # Log the request
                print(f"Request to {ctx.endpoint}")
                return HookResult.proceed()

            async def on_llm_end(self, ctx: HookContext) -> HookResult:
                # Track token usage
                print(f"Tokens: {ctx.usage.total_tokens}")
                return HookResult.proceed()
    """

    # Lifecycle hooks
    async def on_startup(self, app: App) -> None:
        """Called when the application starts."""
        pass

    async def on_shutdown(self, app: App) -> None:
        """Called when the application shuts down."""
        pass

    # Request hooks
    async def on_request(self, _ctx: HookContext) -> HookResult:
        """Called at the start of a request."""
        return HookResult.proceed()

    async def on_response(self, _ctx: HookContext) -> HookResult:
        """Called before sending a response."""
        return HookResult.proceed()

    # LLM hooks
    async def on_llm_start(self, _ctx: HookContext) -> HookResult:
        """Called before an LLM call."""
        return HookResult.proceed()

    async def on_llm_end(self, _ctx: HookContext) -> HookResult:
        """Called after an LLM call."""
        return HookResult.proceed()

    # Tool hooks
    async def on_tool_call(self, _ctx: HookContext) -> HookResult:
        """Called before a tool is invoked."""
        return HookResult.proceed()

    async def on_tool_result(self, _ctx: HookContext) -> HookResult:
        """Called after a tool returns."""
        return HookResult.proceed()

    # Node hooks (LangGraph)
    async def on_node_enter(self, _ctx: HookContext) -> HookResult:
        """Called when entering a graph node."""
        return HookResult.proceed()

    async def on_node_exit(self, _ctx: HookContext) -> HookResult:
        """Called when exiting a graph node."""
        return HookResult.proceed()

    # Durability hooks
    async def on_checkpoint(self, _ctx: HookContext) -> HookResult:
        """Called when a checkpoint is created."""
        return HookResult.proceed()

    async def on_resume(self, _ctx: HookContext) -> HookResult:
        """Called when resuming from a checkpoint."""
        return HookResult.proceed()

    # Error hooks
    async def on_error(self, _ctx: HookContext) -> HookResult:
        """Called when an error occurs."""
        return HookResult.proceed()

    async def on_retry(self, _ctx: HookContext) -> HookResult:
        """Called before a retry attempt."""
        return HookResult.proceed()

    # Memory hooks
    async def on_memory_add(self, _ctx: HookContext) -> HookResult:
        """Called before adding to memory."""
        return HookResult.proceed()

    async def on_memory_search(self, _ctx: HookContext) -> HookResult:
        """Called after memory search."""
        return HookResult.proceed()


def hook(
    event: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to register a function as a hook.

    Example:
        @hook("on_llm_start")
        async def check_guardrails(ctx: HookContext) -> HookResult:
            if contains_pii(ctx.messages):
                return HookResult.reject("PII detected")
            return HookResult.proceed()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if event not in _hook_registry:
            _hook_registry[event] = []
        _hook_registry[event].append(func)
        return func

    return decorator


def get_registered_hooks(event: str) -> list[Callable[..., Any]]:
    """Get all registered hooks for an event."""
    return _hook_registry.get(event, [])
