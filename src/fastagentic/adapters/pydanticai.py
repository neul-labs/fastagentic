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
        checkpoint_on_tool: bool = True,
    ) -> None:
        """Initialize the PydanticAI adapter.

        Args:
            agent: A PydanticAI Agent instance
            deps: Optional dependencies to pass to the agent
            model: Optional model override
            checkpoint_on_tool: Whether to create checkpoints after tool calls
        """
        self.agent = agent
        self.deps = deps
        self.model = model
        self.checkpoint_on_tool = checkpoint_on_tool

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the PydanticAI agent and return the result.

        Args:
            input: The input to the agent (string or dict with 'message' key)
            ctx: The adapter context

        Returns:
            The agent's typed output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            adapter_ctx.state["message_history"] = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True

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

        # Create checkpoint with final state
        messages = []
        if hasattr(result, "all_messages"):
            try:
                messages = [
                    {
                        "role": getattr(m, "role", "unknown"),
                        "content": getattr(m, "content", str(m)),
                    }
                    for m in result.all_messages()
                ]
            except Exception:
                pass

        await self.on_checkpoint(
            {
                "state": {"completed": True},
                "messages": messages,
                "context": {
                    "result": (
                        result.data.model_dump()
                        if hasattr(result.data, "model_dump")
                        else result.data
                    )
                },
            },
            adapter_ctx,
        )

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

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            adapter_ctx.state["message_history"] = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )

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

                # Create checkpoint with final state
                messages = []
                if hasattr(result, "all_messages"):
                    try:
                        messages = [
                            {
                                "role": getattr(m, "role", "unknown"),
                                "content": getattr(m, "content", str(m)),
                            }
                            for m in result.all_messages()
                        ]
                    except Exception:
                        pass

                await self.on_checkpoint(
                    {
                        "state": {"completed": True},
                        "messages": messages,
                        "context": {
                            "result": (
                                final_result.model_dump()
                                if hasattr(final_result, "model_dump")
                                else final_result
                            )
                        },
                    },
                    adapter_ctx,
                )

                yield StreamEvent(
                    type=StreamEventType.CHECKPOINT,
                    data={"step": "completed"},
                    run_id=adapter_ctx.run_id,
                )

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

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            adapter_ctx.state["message_history"] = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )

        message = self._extract_message(input)

        kwargs: dict[str, Any] = {}
        if self.deps is not None:
            kwargs["deps"] = self.deps
        if self.model:
            kwargs["model"] = self.model

        kwargs["message_history"] = adapter_ctx.state.get("message_history")

        # Track tool calls and messages for checkpointing
        accumulated_messages: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        try:
            async with self.agent.run_stream(message, **kwargs) as result:
                # Stream structured messages for detailed events
                async for msg in result.stream_structured():
                    if hasattr(msg, "role"):
                        if msg.role == "tool-call":
                            tool_name = getattr(msg, "tool_name", "unknown")
                            tool_input = getattr(msg, "args", {})
                            tool_calls.append({"name": tool_name, "input": tool_input})
                            accumulated_messages.append(
                                {
                                    "role": "tool-call",
                                    "tool_name": tool_name,
                                    "args": tool_input,
                                }
                            )

                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL,
                                data={
                                    "name": tool_name,
                                    "input": tool_input,
                                },
                                run_id=adapter_ctx.run_id,
                            )

                        elif msg.role == "tool-return":
                            tool_name = getattr(msg, "tool_name", "unknown")
                            tool_output = getattr(msg, "content", None)
                            accumulated_messages.append(
                                {
                                    "role": "tool-return",
                                    "tool_name": tool_name,
                                    "content": tool_output,
                                }
                            )

                            yield StreamEvent(
                                type=StreamEventType.TOOL_RESULT,
                                data={
                                    "name": tool_name,
                                    "output": tool_output,
                                },
                                run_id=adapter_ctx.run_id,
                            )

                            # Checkpoint after tool result if configured
                            if self.checkpoint_on_tool:
                                await self.on_checkpoint(
                                    {
                                        "state": {"step": "tool_complete", "tool_name": tool_name},
                                        "messages": accumulated_messages.copy(),
                                        "tool_calls": tool_calls.copy(),
                                    },
                                    adapter_ctx,
                                )
                                yield StreamEvent(
                                    type=StreamEventType.CHECKPOINT,
                                    data={"step": "tool_complete", "tool_name": tool_name},
                                    run_id=adapter_ctx.run_id,
                                )
                    else:
                        # Text content
                        content = getattr(msg, "content", str(msg))
                        if content:
                            accumulated_messages.append(
                                {
                                    "role": "assistant",
                                    "content": content,
                                }
                            )
                            yield StreamEvent(
                                type=StreamEventType.TOKEN,
                                data={"content": content},
                                run_id=adapter_ctx.run_id,
                            )

                final_result = await result.get_data()

                # Final checkpoint
                await self.on_checkpoint(
                    {
                        "state": {"completed": True},
                        "messages": accumulated_messages,
                        "tool_calls": tool_calls,
                        "context": {
                            "result": (
                                final_result.model_dump()
                                if hasattr(final_result, "model_dump")
                                else final_result
                            )
                        },
                    },
                    adapter_ctx,
                )

                yield StreamEvent(
                    type=StreamEventType.CHECKPOINT,
                    data={"step": "completed"},
                    run_id=adapter_ctx.run_id,
                )

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
            checkpoint_on_tool=self.checkpoint_on_tool,
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
            checkpoint_on_tool=self.checkpoint_on_tool,
        )

    def with_checkpoints(self, on_tool: bool = True) -> PydanticAIAdapter:
        """Create a new adapter with checkpoint configuration.

        Args:
            on_tool: Whether to checkpoint after each tool call

        Returns:
            A new PydanticAIAdapter with the specified checkpoint settings
        """
        return PydanticAIAdapter(
            self.agent,
            deps=self.deps,
            model=self.model,
            checkpoint_on_tool=on_tool,
        )
