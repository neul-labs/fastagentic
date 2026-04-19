"""LangChain adapter for FastAgentic.

This adapter wraps LangChain Runnables (chains, agents, LCEL pipelines)
to expose them via FastAgentic endpoints.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain Runnables.

    Wraps any LangChain Runnable (chains, agents, LCEL pipelines) to work
    with FastAgentic's endpoint system.

    Example:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from fastagentic.adapters.langchain import LangChainAdapter

        prompt = ChatPromptTemplate.from_messages([...])
        llm = ChatOpenAI(model="gpt-4o")
        chain = prompt | llm

        adapter = LangChainAdapter(chain)

        @agent_endpoint(path="/analyze", runnable=adapter)
        async def analyze(input: AnalyzeInput) -> AnalyzeOutput:
            ...
    """

    def __init__(
        self,
        runnable: Runnable[Any, Any],
        *,
        stream_mode: str = "values",
        checkpoint_events: list[str] | None = None,
    ) -> None:
        """Initialize the LangChain adapter.

        Args:
            runnable: A LangChain Runnable (chain, agent, LCEL pipeline)
            stream_mode: Streaming mode - "values" for final outputs,
                        "events" for all events
            checkpoint_events: Events that trigger checkpoints (e.g., ["on_chain_end", "on_tool_end"])
        """
        self.runnable = runnable
        self.stream_mode = stream_mode
        self.checkpoint_events = checkpoint_events or ["on_chain_end", "on_tool_end"]

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the LangChain runnable and return the result.

        Args:
            input: The input to the chain (dict or Pydantic model)
            ctx: The adapter context

        Returns:
            The chain's output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            checkpoint_state = checkpoint.get("state", {})
            adapter_ctx.agent_ctx.run._is_resumed = True
        else:
            checkpoint_state = {}

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # Merge checkpoint state with input
        if isinstance(input, dict) and checkpoint_state:
            input = {**checkpoint_state, **input}

        # Add run metadata to config
        config = {
            "metadata": {
                "run_id": adapter_ctx.run_id,
                **adapter_ctx.metadata,
            },
            "callbacks": [],
        }

        result = await self.runnable.ainvoke(input, config=config)

        # Create checkpoint with final state
        await self.on_checkpoint(
            {
                "state": {"completed": True},
                "context": {
                    "result": result if isinstance(result, dict) else {"output": str(result)}
                },
            },
            adapter_ctx,
        )

        return result

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from the LangChain runnable.

        Args:
            input: The input to the chain
            ctx: The adapter context

        Yields:
            StreamEvent objects for tokens, tool calls, etc.
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            checkpoint_state = checkpoint.get("state", {})
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )
        else:
            checkpoint_state = {}

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # Merge checkpoint state with input
        if isinstance(input, dict) and checkpoint_state:
            input = {**checkpoint_state, **input}

        config = {
            "metadata": {
                "run_id": adapter_ctx.run_id,
                **adapter_ctx.metadata,
            },
        }

        # Track accumulated state for checkpointing
        accumulated_state: dict[str, Any] = {}
        tool_calls: list[dict[str, Any]] = []

        try:
            if self.stream_mode == "events":
                # Use astream_events for detailed streaming
                async for event in self.runnable.astream_events(input, config=config, version="v2"):
                    event_type = event.get("event", "")
                    event_data = event.get("data", {})
                    event_name = event.get("name", "")

                    # Track state for checkpointing
                    if event_type == "on_chain_end":
                        output = event_data.get("output")
                        if output:
                            accumulated_state[event_name] = output
                    elif event_type == "on_tool_end":
                        tool_calls.append(
                            {
                                "name": event_name,
                                "output": event_data.get("output"),
                            }
                        )

                    # Convert and yield the event
                    stream_event = self._convert_langchain_event(event, adapter_ctx.run_id)
                    if stream_event:
                        yield stream_event

                    # Checkpoint on configured events
                    if event_type in self.checkpoint_events:
                        await self.on_checkpoint(
                            {
                                "state": accumulated_state.copy(),
                                "tool_calls": tool_calls.copy(),
                                "context": {"last_event": event_type, "event_name": event_name},
                            },
                            adapter_ctx,
                        )
                        yield StreamEvent(
                            type=StreamEventType.CHECKPOINT,
                            data={"event": event_type, "name": event_name},
                            run_id=adapter_ctx.run_id,
                        )
            else:
                # Use astream for simple value streaming
                full_content = ""
                async for chunk in self.runnable.astream(input, config=config):
                    content = self._extract_content(chunk)
                    full_content += content
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": content},
                        run_id=adapter_ctx.run_id,
                    )

                # Final checkpoint for values mode
                await self.on_checkpoint(
                    {
                        "state": {"completed": True},
                        "context": {"output": full_content},
                    },
                    adapter_ctx,
                )
                yield StreamEvent(
                    type=StreamEventType.CHECKPOINT,
                    data={"step": "completed"},
                    run_id=adapter_ctx.run_id,
                )

            # Signal completion
            yield StreamEvent(
                type=StreamEventType.DONE,
                data={},
                run_id=adapter_ctx.run_id,
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    def _convert_langchain_event(
        self, event: dict[str, Any], run_id: str | None = None
    ) -> StreamEvent | None:
        """Convert a LangChain event to a FastAgentic StreamEvent."""
        event_type = event.get("event", "")
        data = event.get("data", {})

        if event_type == "on_chat_model_stream":
            # Token from chat model
            chunk = data.get("chunk")
            if chunk:
                content = getattr(chunk, "content", "")
                if content:
                    return StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": content},
                        run_id=run_id,
                    )

        elif event_type == "on_tool_start":
            return StreamEvent(
                type=StreamEventType.TOOL_CALL,
                data={
                    "name": event.get("name", ""),
                    "input": data.get("input", {}),
                },
                run_id=run_id,
            )

        elif event_type == "on_tool_end":
            return StreamEvent(
                type=StreamEventType.TOOL_RESULT,
                data={
                    "name": event.get("name", ""),
                    "output": data.get("output"),
                },
                run_id=run_id,
            )

        elif event_type == "on_chain_start":
            return StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": event.get("name", "")},
                run_id=run_id,
            )

        elif event_type == "on_chain_end":
            return StreamEvent(
                type=StreamEventType.NODE_END,
                data={
                    "name": event.get("name", ""),
                    "output": data.get("output"),
                },
                run_id=run_id,
            )

        return None

    def _extract_content(self, chunk: Any) -> str:
        """Extract string content from various chunk types."""
        if isinstance(chunk, str):
            return chunk

        if hasattr(chunk, "content"):
            return str(chunk.content)

        if isinstance(chunk, dict):
            content = chunk.get("content")
            return str(content) if content is not None else str(chunk)

        return str(chunk)

    def with_checkpoints(self, events: list[str]) -> LangChainAdapter:
        """Create a new adapter with specific checkpoint events.

        Args:
            events: List of event types that trigger checkpoints
                   (e.g., ["on_chain_end", "on_tool_end"])

        Returns:
            A new LangChainAdapter with the specified checkpoint events
        """
        return LangChainAdapter(
            self.runnable,
            stream_mode=self.stream_mode,
            checkpoint_events=events,
        )

    def with_stream_mode(self, mode: str) -> LangChainAdapter:
        """Create a new adapter with a different stream mode.

        Args:
            mode: Stream mode ("values" or "events")

        Returns:
            A new LangChainAdapter with the specified stream mode
        """
        return LangChainAdapter(
            self.runnable,
            stream_mode=mode,
            checkpoint_events=self.checkpoint_events,
        )
