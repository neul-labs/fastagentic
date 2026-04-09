"""Base adapter interface for agent frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastagentic.types import StreamEvent

if TYPE_CHECKING:
    from fastagentic.context import AgentContext


@dataclass
class AdapterContext:
    """Context passed to adapter methods.

    Contains the agent context plus adapter-specific state.
    """

    agent_ctx: AgentContext
    config: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self.agent_ctx.run_id

    @property
    def metadata(self) -> dict[str, Any]:
        """Get run metadata."""
        return self.agent_ctx.metadata


class BaseAdapter(ABC):
    """Base class for all agent framework adapters.

    Adapters wrap agent frameworks (PydanticAI, LangGraph, CrewAI, LangChain)
    to provide a unified interface for FastAgentic.

    Subclasses must implement:
    - invoke(): Run the agent and return the result
    - stream(): Run the agent and yield streaming events

    Example:
        class MyAdapter(BaseAdapter):
            def __init__(self, agent):
                self.agent = agent

            async def invoke(self, input: Any, ctx: AdapterContext) -> Any:
                return await self.agent.run(input)

            async def stream(self, input: Any, ctx: AdapterContext) -> AsyncIterator[StreamEvent]:
                async for chunk in self.agent.stream(input):
                    yield StreamEvent(type=StreamEventType.TOKEN, data={"content": chunk})
    """

    @abstractmethod
    async def invoke(self, input: Any, ctx: AdapterContext | AgentContext) -> Any:
        """Run the agent and return the final result.

        Args:
            input: The input to the agent (typically a Pydantic model or dict)
            ctx: The adapter or agent context

        Returns:
            The agent's output
        """
        ...

    @abstractmethod
    async def stream(
        self, input: Any, ctx: AdapterContext | AgentContext
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent and yield streaming events.

        Args:
            input: The input to the agent
            ctx: The adapter or agent context

        Yields:
            StreamEvent objects representing tokens, tool calls, etc.
        """
        ...
        # Make this an async generator
        yield  # type: ignore[misc]

    def _ensure_adapter_context(self, ctx: AdapterContext | AgentContext) -> AdapterContext:
        """Convert AgentContext to AdapterContext if needed."""
        if isinstance(ctx, AdapterContext):
            return ctx

        # Import here to avoid circular imports
        from fastagentic.context import AgentContext

        if isinstance(ctx, AgentContext):
            return AdapterContext(agent_ctx=ctx)

        raise TypeError(f"Expected AdapterContext or AgentContext, got {type(ctx)}")

    async def on_checkpoint(self, state: dict[str, Any], ctx: AdapterContext) -> None:
        """Called when a checkpoint should be saved.

        Override this to customize checkpoint behavior.

        Args:
            state: The state to checkpoint
            ctx: The adapter context
        """
        ctx.agent_ctx.run.add_checkpoint(state)

    async def on_resume(self, ctx: AdapterContext) -> dict[str, Any] | None:
        """Called when resuming from a checkpoint.

        Override this to customize resume behavior.

        Args:
            ctx: The adapter context

        Returns:
            The checkpoint state to resume from, or None if no checkpoint
        """
        checkpoints = ctx.agent_ctx.run._checkpoints
        return checkpoints[-1] if checkpoints else None
