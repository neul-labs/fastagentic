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
    ) -> None:
        """Initialize the LangChain adapter.

        Args:
            runnable: A LangChain Runnable (chain, agent, LCEL pipeline)
            stream_mode: Streaming mode - "values" for final outputs,
                        "events" for all events
        """
        self.runnable = runnable
        self.stream_mode = stream_mode

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the LangChain runnable and return the result.

        Args:
            input: The input to the chain (dict or Pydantic model)
            ctx: The adapter context

        Returns:
            The chain's output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # Add run metadata to config
        config = {
            "metadata": {
                "run_id": adapter_ctx.run_id,
                **adapter_ctx.metadata,
            },
            "callbacks": [],
        }

        result = await self.runnable.ainvoke(input, config=config)

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

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        config = {
            "metadata": {
                "run_id": adapter_ctx.run_id,
                **adapter_ctx.metadata,
            },
        }

        try:
            if self.stream_mode == "events":
                # Use astream_events for detailed streaming
                async for event in self.runnable.astream_events(input, config=config, version="v2"):
                    stream_event = self._convert_langchain_event(event)
                    if stream_event:
                        yield stream_event
            else:
                # Use astream for simple value streaming
                async for chunk in self.runnable.astream(input, config=config):
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": self._extract_content(chunk)},
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

    def _convert_langchain_event(self, event: dict[str, Any]) -> StreamEvent | None:
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
                    )

        elif event_type == "on_tool_start":
            return StreamEvent(
                type=StreamEventType.TOOL_CALL,
                data={
                    "name": event.get("name", ""),
                    "input": data.get("input", {}),
                },
            )

        elif event_type == "on_tool_end":
            return StreamEvent(
                type=StreamEventType.TOOL_RESULT,
                data={
                    "name": event.get("name", ""),
                    "output": data.get("output"),
                },
            )

        elif event_type == "on_chain_start":
            return StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": event.get("name", "")},
            )

        elif event_type == "on_chain_end":
            return StreamEvent(
                type=StreamEventType.NODE_END,
                data={
                    "name": event.get("name", ""),
                    "output": data.get("output"),
                },
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
