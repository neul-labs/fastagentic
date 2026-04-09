"""PydanticAI adapter for FastAgentic.

This adapter wraps PydanticAI Agents to expose them via FastAgentic endpoints
with full streaming support and Logfire integration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from pydantic_ai import Agent


class PydanticAIAdapter(BaseAdapter):
    """Adapter for PydanticAI Agents.

    Wraps a PydanticAI Agent to work with FastAgentic's endpoint system,
    providing streaming, checkpointing, and observability.

    Example:
        from pydantic_ai import Agent
        from fastagentic.adapters.pydanticai import PydanticAIAdapter

        agent = Agent("openai:gpt-4o", result_type=MyOutput)
        adapter = PydanticAIAdapter(agent)

        @agent_endpoint(path="/analyze", runnable=adapter, stream=True)
        async def analyze(input: AnalyzeInput) -> AnalyzeOutput:
            ...
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        deps: Any = None,
        model: str | None = None,
    ) -> None:
        """Initialize the PydanticAI adapter.

        Args:
            agent: A PydanticAI Agent instance
            deps: Optional dependencies to pass to the agent
            model: Optional model override
        """
        self.agent = agent
        self.deps = deps
        self.model = model

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the PydanticAI agent and return the result.

        Args:
            input: The input to the agent (string or dict with 'message' key)
            ctx: The adapter context

        Returns:
            The agent's typed output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Extract message from input
        message = self._extract_message(input)

        # Build run kwargs
        kwargs: dict[str, Any] = {}
        if self.deps is not None:
            kwargs["deps"] = self.deps
        if self.model:
            kwargs["model"] = self.model

        # Add metadata for observability
        kwargs["message_history"] = adapter_ctx.state.get("message_history")

        # Run the agent
        result = await self.agent.run(message, **kwargs)

        # Track usage
        if hasattr(result, "usage"):
            usage = result.usage()
            adapter_ctx.agent_ctx.usage.input_tokens += usage.request_tokens or 0
            adapter_ctx.agent_ctx.usage.output_tokens += usage.response_tokens or 0
            adapter_ctx.agent_ctx.usage.total_tokens += usage.total_tokens or 0

        return result.data

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from the PydanticAI agent.

        Args:
            input: The input to the agent
            ctx: The adapter context

        Yields:
            StreamEvent objects for tokens, tool calls, etc.
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Extract message from input
        message = self._extract_message(input)

        # Build run kwargs
        kwargs: dict[str, Any] = {}
        if self.deps is not None:
            kwargs["deps"] = self.deps
        if self.model:
            kwargs["model"] = self.model

        kwargs["message_history"] = adapter_ctx.state.get("message_history")

        try:
            async with self.agent.run_stream(message, **kwargs) as result:
                # Stream text chunks
                async for text in result.stream_text():
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": text},
                        run_id=adapter_ctx.run_id,
                    )

                # Get the final result
                final_result = await result.get_data()

                # Track usage
                if hasattr(result, "usage"):
                    usage = result.usage()
                    adapter_ctx.agent_ctx.usage.input_tokens += usage.request_tokens or 0
                    adapter_ctx.agent_ctx.usage.output_tokens += usage.response_tokens or 0
                    adapter_ctx.agent_ctx.usage.total_tokens += usage.total_tokens or 0

                # Yield final result
                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={
                        "result": (
                            final_result.model_dump()
                            if hasattr(final_result, "model_dump")
                            else final_result
                        )
                    },
                    run_id=adapter_ctx.run_id,
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    async def stream_with_tools(
        self, input: Any, ctx: AdapterContext | Any
    ) -> AsyncIterator[StreamEvent]:
        """Stream with detailed tool call events.

        This method provides more granular streaming including tool calls
        and their results.

        Args:
            input: The input to the agent
            ctx: The adapter context

        Yields:
            StreamEvent objects including tool calls
        """
        adapter_ctx = self._ensure_adapter_context(ctx)
        message = self._extract_message(input)

        kwargs: dict[str, Any] = {}
        if self.deps is not None:
            kwargs["deps"] = self.deps
        if self.model:
            kwargs["model"] = self.model

        try:
            async with self.agent.run_stream(message, **kwargs) as result:
                # Stream structured messages for detailed events
                async for message in result.stream_structured():
                    if hasattr(message, "role"):
                        if message.role == "tool-call":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL,
                                data={
                                    "name": getattr(message, "tool_name", "unknown"),
                                    "input": getattr(message, "args", {}),
                                },
                                run_id=adapter_ctx.run_id,
                            )
                        elif message.role == "tool-return":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_RESULT,
                                data={
                                    "name": getattr(message, "tool_name", "unknown"),
                                    "output": getattr(message, "content", None),
                                },
                                run_id=adapter_ctx.run_id,
                            )
                    else:
                        # Text content
                        content = getattr(message, "content", str(message))
                        if content:
                            yield StreamEvent(
                                type=StreamEventType.TOKEN,
                                data={"content": content},
                                run_id=adapter_ctx.run_id,
                            )

                final_result = await result.get_data()
                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={
                        "result": (
                            final_result.model_dump()
                            if hasattr(final_result, "model_dump")
                            else final_result
                        )
                    },
                    run_id=adapter_ctx.run_id,
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    def _extract_message(self, input: Any) -> str:
        """Extract the message string from various input formats."""
        if isinstance(input, str):
            return input

        if hasattr(input, "model_dump"):
            data = input.model_dump()
        elif isinstance(input, dict):
            data = input
        else:
            return str(input)

        # Look for common message field names
        for key in ("message", "query", "prompt", "input", "text", "content"):
            if key in data:
                return str(data[key])

        # Fallback to string representation
        return str(data)

    def with_deps(self, deps: Any) -> PydanticAIAdapter:
        """Create a new adapter with different dependencies.

        Args:
            deps: New dependencies to use

        Returns:
            A new PydanticAIAdapter with the updated deps
        """
        return PydanticAIAdapter(
            self.agent,
            deps=deps,
            model=self.model,
        )

    def with_model(self, model: str) -> PydanticAIAdapter:
        """Create a new adapter with a different model.

        Args:
            model: New model to use

        Returns:
            A new PydanticAIAdapter with the updated model
        """
        return PydanticAIAdapter(
            self.agent,
            deps=self.deps,
            model=model,
        )
