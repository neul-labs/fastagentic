"""LangGraph adapter for FastAgentic.

This adapter wraps LangGraph StateGraphs to expose them via FastAgentic endpoints
with node-level checkpointing and streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph StateGraphs.

    Wraps a compiled LangGraph to work with FastAgentic's endpoint system,
    providing node-level streaming, checkpointing, and conditional routing.

    Example:
        from langgraph.graph import StateGraph
        from fastagentic.adapters.langgraph import LangGraphAdapter

        workflow = StateGraph(MyState)
        workflow.add_node("process", process_node)
        workflow.add_node("validate", validate_node)
        graph = workflow.compile()

        adapter = LangGraphAdapter(graph)

        @agent_endpoint(path="/workflow", runnable=adapter, durable=True)
        async def run_workflow(input: WorkflowInput) -> WorkflowOutput:
            ...
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        checkpoint_nodes: list[str] | None = None,
        stream_mode: str = "values",
    ) -> None:
        """Initialize the LangGraph adapter.

        Args:
            graph: A compiled LangGraph StateGraph
            checkpoint_nodes: Nodes after which to create checkpoints (None = all)
            stream_mode: Streaming mode - "values", "updates", or "debug"
        """
        self.graph = graph
        self.checkpoint_nodes = checkpoint_nodes
        self.stream_mode = stream_mode

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the LangGraph and return the final state.

        Args:
            input: The initial state or input dict
            ctx: The adapter context

        Returns:
            The final graph state
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            input = {**input, **checkpoint.get("state", {})}
            adapter_ctx.agent_ctx.run._is_resumed = True

        # Build config with run metadata
        config = {
            "configurable": {
                "thread_id": adapter_ctx.run_id,
            },
            "metadata": {
                "run_id": adapter_ctx.run_id,
                **adapter_ctx.metadata,
            },
        }

        # Run the graph
        result = await self.graph.ainvoke(input, config=config)

        return result

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from the LangGraph execution.

        Yields node-level events including state updates and checkpoints.

        Args:
            input: The initial state or input dict
            ctx: The adapter context

        Yields:
            StreamEvent objects for nodes, state updates, etc.
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            input = {**input, **checkpoint.get("state", {})}
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )

        config = {
            "configurable": {
                "thread_id": adapter_ctx.run_id,
            },
            "metadata": {
                "run_id": adapter_ctx.run_id,
                **adapter_ctx.metadata,
            },
        }

        try:
            current_node: str | None = None
            final_state: dict[str, Any] = {}

            async for event in self.graph.astream(
                input,
                config=config,
                stream_mode=self.stream_mode,
            ):
                if self.stream_mode == "values":
                    # Values mode: event is the current state
                    final_state = event if isinstance(event, dict) else {}
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"state": final_state},
                        run_id=adapter_ctx.run_id,
                    )

                elif self.stream_mode == "updates":
                    # Updates mode: event is {node_name: updates}
                    if isinstance(event, dict):
                        for node_name, updates in event.items():
                            # Node start
                            if node_name != current_node:
                                if current_node:
                                    yield StreamEvent(
                                        type=StreamEventType.NODE_END,
                                        data={"name": current_node},
                                        run_id=adapter_ctx.run_id,
                                    )
                                current_node = node_name
                                yield StreamEvent(
                                    type=StreamEventType.NODE_START,
                                    data={"name": node_name},
                                    run_id=adapter_ctx.run_id,
                                )

                            # State update
                            yield StreamEvent(
                                type=StreamEventType.TOKEN,
                                data={"node": node_name, "updates": updates},
                                run_id=adapter_ctx.run_id,
                            )

                            # Checkpoint if configured
                            if self._should_checkpoint(node_name):
                                await self.on_checkpoint(
                                    {"node": node_name, "state": updates},
                                    adapter_ctx,
                                )
                                yield StreamEvent(
                                    type=StreamEventType.CHECKPOINT,
                                    data={"node": node_name},
                                    run_id=adapter_ctx.run_id,
                                )

                            final_state = {**final_state, **updates}

                elif self.stream_mode == "debug":
                    # Debug mode: detailed execution events
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"debug": event},
                        run_id=adapter_ctx.run_id,
                    )

            # Close last node
            if current_node:
                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    data={"name": current_node},
                    run_id=adapter_ctx.run_id,
                )

            # Done
            yield StreamEvent(
                type=StreamEventType.DONE,
                data={"result": final_state},
                run_id=adapter_ctx.run_id,
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e), "node": current_node},
                run_id=adapter_ctx.run_id,
            )

    async def stream_events(
        self, input: Any, ctx: AdapterContext | Any
    ) -> AsyncIterator[StreamEvent]:
        """Stream detailed events including LLM tokens and tool calls.

        Uses LangGraph's astream_events for maximum detail.

        Args:
            input: The initial state
            ctx: The adapter context

        Yields:
            Detailed StreamEvent objects
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        if hasattr(input, "model_dump"):
            input = input.model_dump()

        config = {
            "configurable": {
                "thread_id": adapter_ctx.run_id,
            },
        }

        try:
            async for event in self.graph.astream_events(input, config=config, version="v2"):
                stream_event = self._convert_event(event, adapter_ctx.run_id)
                if stream_event:
                    yield stream_event

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

    def _convert_event(self, event: dict[str, Any], run_id: str) -> StreamEvent | None:
        """Convert a LangGraph event to a FastAgentic StreamEvent."""
        event_type = event.get("event", "")
        data = event.get("data", {})
        name = event.get("name", "")

        if event_type == "on_chain_start":
            return StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": name},
                run_id=run_id,
            )

        elif event_type == "on_chain_end":
            return StreamEvent(
                type=StreamEventType.NODE_END,
                data={"name": name, "output": data.get("output")},
                run_id=run_id,
            )

        elif event_type == "on_chat_model_stream":
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
                data={"name": name, "input": data.get("input")},
                run_id=run_id,
            )

        elif event_type == "on_tool_end":
            return StreamEvent(
                type=StreamEventType.TOOL_RESULT,
                data={"name": name, "output": data.get("output")},
                run_id=run_id,
            )

        return None

    def _should_checkpoint(self, node_name: str) -> bool:
        """Check if a checkpoint should be created after this node."""
        if self.checkpoint_nodes is None:
            return True  # Checkpoint all nodes
        return node_name in self.checkpoint_nodes

    async def resume_from(
        self,
        checkpoint_id: str,
        input: Any,
        ctx: AdapterContext | Any,
    ) -> AsyncIterator[StreamEvent]:
        """Resume execution from a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint to resume from
            input: Additional input to merge with checkpoint state
            ctx: The adapter context

        Yields:
            StreamEvent objects from resumed execution
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Load checkpoint from durable store
        # This is a simplified implementation - real implementation
        # would load from the configured durable store
        checkpoint = adapter_ctx.state.get("checkpoints", {}).get(checkpoint_id)
        if not checkpoint:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": f"Checkpoint {checkpoint_id} not found"},
                run_id=adapter_ctx.run_id,
            )
            return

        # Merge checkpoint state with new input
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        merged_state = {**checkpoint.get("state", {}), **input}

        # Continue streaming from merged state
        async for event in self.stream(merged_state, ctx):
            yield event

    def with_checkpoints(self, nodes: list[str]) -> LangGraphAdapter:
        """Create a new adapter with specific checkpoint nodes.

        Args:
            nodes: List of node names to checkpoint after

        Returns:
            A new LangGraphAdapter with the specified checkpoint nodes
        """
        return LangGraphAdapter(
            self.graph,
            checkpoint_nodes=nodes,
            stream_mode=self.stream_mode,
        )

    def with_stream_mode(self, mode: str) -> LangGraphAdapter:
        """Create a new adapter with a different stream mode.

        Args:
            mode: Stream mode ("values", "updates", or "debug")

        Returns:
            A new LangGraphAdapter with the specified stream mode
        """
        return LangGraphAdapter(
            self.graph,
            checkpoint_nodes=self.checkpoint_nodes,
            stream_mode=mode,
        )
