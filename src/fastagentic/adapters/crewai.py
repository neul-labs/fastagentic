"""CrewAI adapter for FastAgentic.

This adapter wraps CrewAI Crews to expose them via FastAgentic endpoints
with per-agent streaming and observability.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from crewai import Crew


class CrewAIAdapter(BaseAdapter):
    """Adapter for CrewAI Crews.

    Wraps a CrewAI Crew to work with FastAgentic's endpoint system,
    providing per-agent streaming and task-level observability.

    Example:
        from crewai import Agent, Task, Crew
        from fastagentic.adapters.crewai import CrewAIAdapter

        researcher = Agent(role="Researcher", ...)
        writer = Agent(role="Writer", ...)
        crew = Crew(agents=[researcher, writer], tasks=[...])

        adapter = CrewAIAdapter(crew)

        @agent_endpoint(path="/research", runnable=adapter)
        async def research(topic: str) -> Report:
            ...
    """

    def __init__(
        self,
        crew: Crew,
        *,
        stream_agent_output: bool = True,
        stream_task_output: bool = True,
        checkpoint_tasks: bool = True,
    ) -> None:
        """Initialize the CrewAI adapter.

        Args:
            crew: A CrewAI Crew instance
            stream_agent_output: Whether to stream per-agent output
            stream_task_output: Whether to stream per-task output
            checkpoint_tasks: Whether to create checkpoints after each task
        """
        self.crew = crew
        self.stream_agent_output = stream_agent_output
        self.stream_task_output = stream_task_output
        self.checkpoint_tasks = checkpoint_tasks

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the CrewAI crew and return the result.

        Args:
            input: The input to the crew (dict with task inputs)
            ctx: The adapter context

        Returns:
            The crew's output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        completed_tasks: int = 0
        task_outputs: list[Any] = []
        if checkpoint:
            completed_tasks = checkpoint.get("state", {}).get("completed_tasks", 0)
            task_outputs = checkpoint.get("task_outputs", [])
            adapter_ctx.agent_ctx.run._is_resumed = True

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # CrewAI's kickoff is synchronous, so we run it in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.crew.kickoff(inputs=input),
        )

        # Create checkpoint with final state
        if self.checkpoint_tasks:
            await self.on_checkpoint(
                {
                    "state": {
                        "completed": True,
                        "completed_tasks": len(self.crew.tasks),
                    },
                    "task_outputs": task_outputs,
                    "context": {"result": str(result) if result else None},
                },
                adapter_ctx,
            )

        return result

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from the CrewAI crew execution.

        Yields events for agent starts/ends and task completions.

        Args:
            input: The input to the crew
            ctx: The adapter context

        Yields:
            StreamEvent objects for agents, tasks, and output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        completed_tasks: int = 0
        task_outputs: list[Any] = []
        if checkpoint:
            completed_tasks = checkpoint.get("state", {}).get("completed_tasks", 0)
            task_outputs = checkpoint.get("task_outputs", [])
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        try:
            # Try native event bus streaming first (CrewAI 0.30+)
            import importlib.util

            has_event_bus = (
                importlib.util.find_spec("crewai.utilities.events") is not None
            )

            if has_event_bus:
                async for event in self._stream_with_event_bus(
                    input, adapter_ctx, completed_tasks, task_outputs
                ):
                    yield event
                return

            # Fallback to polling-based streaming
            # Yield crew start event
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={
                    "name": "crew",
                    "agents": [a.role for a in self.crew.agents],
                    "tasks": len(self.crew.tasks),
                },
                run_id=adapter_ctx.run_id,
            )

            # Run the crew in a thread with callbacks
            loop = asyncio.get_event_loop()

            # Emit agent/task events based on crew structure
            for i, task in enumerate(self.crew.tasks):
                agent = task.agent
                agent_role = agent.role if agent else f"Agent {i}"

                # Agent start
                yield StreamEvent(
                    type=StreamEventType.NODE_START,
                    data={
                        "name": f"agent:{agent_role}",
                        "task_index": i,
                        "task_description": task.description[:100] if task.description else "",
                    },
                    run_id=adapter_ctx.run_id,
                )

                # Task start
                if self.stream_task_output:
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={
                            "type": "task_start",
                            "task_index": i,
                            "agent": agent_role,
                        },
                        run_id=adapter_ctx.run_id,
                    )

            # Run the crew
            result = await loop.run_in_executor(
                None,
                lambda: self.crew.kickoff(inputs=input),
            )

            # Emit completion events for each task/agent with checkpoints
            for i, task in enumerate(self.crew.tasks):
                agent = task.agent
                agent_role = agent.role if agent else f"Agent {i}"

                # Extract tool calls from agent if available
                if agent and hasattr(agent, "tools") and agent.tools:
                    for tool in agent.tools:
                        tool_name = getattr(tool, "name", str(tool))
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            data={"name": tool_name, "agent": agent_role, "task_index": i},
                            run_id=adapter_ctx.run_id,
                        )

                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    data={
                        "name": f"agent:{agent_role}",
                        "task_index": i,
                    },
                    run_id=adapter_ctx.run_id,
                )

                # Checkpoint after each task
                if self.checkpoint_tasks:
                    task_outputs.append({"task_index": i, "agent": agent_role})
                    await self.on_checkpoint(
                        {
                            "state": {
                                "completed_tasks": i + 1,
                                "current_agent": agent_role,
                            },
                            "task_outputs": task_outputs.copy(),
                        },
                        adapter_ctx,
                    )
                    yield StreamEvent(
                        type=StreamEventType.CHECKPOINT,
                        data={"task_index": i, "agent": agent_role},
                        run_id=adapter_ctx.run_id,
                    )

            # Crew complete
            yield StreamEvent(
                type=StreamEventType.NODE_END,
                data={"name": "crew"},
                run_id=adapter_ctx.run_id,
            )

            # Final checkpoint
            if self.checkpoint_tasks:
                await self.on_checkpoint(
                    {
                        "state": {
                            "completed": True,
                            "completed_tasks": len(self.crew.tasks),
                        },
                        "task_outputs": task_outputs,
                        "context": {"result": str(result) if result else None},
                    },
                    adapter_ctx,
                )
                yield StreamEvent(
                    type=StreamEventType.CHECKPOINT,
                    data={"step": "completed"},
                    run_id=adapter_ctx.run_id,
                )

            # Final result
            yield StreamEvent(
                type=StreamEventType.DONE,
                data={
                    "result": str(result) if result else None,
                    "raw": result.raw if hasattr(result, "raw") else None,
                },
                run_id=adapter_ctx.run_id,
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    async def _stream_with_event_bus(
        self,
        input: Any,
        ctx: AdapterContext,
        completed_tasks: int,
        task_outputs: list[Any],
    ) -> AsyncIterator[StreamEvent]:
        """Stream using CrewAI's native event bus for real-time token streaming."""
        from queue import Empty, Queue

        from crewai.utilities.events import crewai_event_bus

        # Queue for events from the event bus
        event_queue: Queue[dict[str, Any]] = Queue()
        done_event = asyncio.Event()

        # Event handlers
        def on_llm_stream_chunk(event: Any) -> None:
            """Handle LLM stream chunk events."""
            chunk = getattr(event, "chunk", "")
            if chunk:
                event_queue.put({"type": "token", "content": chunk})

        def on_tool_use(event: Any) -> None:
            """Handle tool use events."""
            tool_name = getattr(event, "tool_name", "unknown")
            tool_input = getattr(event, "tool_input", {})
            event_queue.put({"type": "tool_call", "name": tool_name, "input": tool_input})

        def on_tool_result(event: Any) -> None:
            """Handle tool result events."""
            tool_name = getattr(event, "tool_name", "unknown")
            tool_output = getattr(event, "result", None)
            event_queue.put({"type": "tool_result", "name": tool_name, "output": tool_output})

        def on_task_complete(event: Any) -> None:
            """Handle task completion events."""
            task_index = getattr(event, "task_index", len(task_outputs))
            agent = getattr(event, "agent", "unknown")
            event_queue.put({"type": "task_complete", "task_index": task_index, "agent": agent})

        try:
            # Register event handlers using scoped handlers
            with crewai_event_bus.scoped_handlers() as bus:
                # Try to register for available event types
                try:
                    from crewai.utilities.events.llm_events import LLMStreamChunkEvent

                    bus.on(LLMStreamChunkEvent)(on_llm_stream_chunk)
                except (ImportError, AttributeError):
                    pass

                try:
                    from crewai.utilities.events.tool_events import ToolUseEvent

                    bus.on(ToolUseEvent)(on_tool_use)
                except (ImportError, AttributeError):
                    pass

                try:
                    from crewai.utilities.events.tool_events import ToolResultEvent

                    bus.on(ToolResultEvent)(on_tool_result)
                except (ImportError, AttributeError):
                    pass

                # Yield crew start event
                yield StreamEvent(
                    type=StreamEventType.NODE_START,
                    data={
                        "name": "crew",
                        "agents": [a.role for a in self.crew.agents],
                        "tasks": len(self.crew.tasks),
                    },
                    run_id=ctx.run_id,
                )

                # Run crew in executor
                loop = asyncio.get_event_loop()

                async def run_crew() -> Any:
                    try:
                        return await loop.run_in_executor(
                            None,
                            lambda: self.crew.kickoff(inputs=input),
                        )
                    finally:
                        done_event.set()

                crew_task = asyncio.create_task(run_crew())

                # Process events from queue while crew runs
                current_task_index = completed_tasks
                while not done_event.is_set() or not event_queue.empty():
                    try:
                        event = event_queue.get_nowait()

                        if event["type"] == "token":
                            yield StreamEvent(
                                type=StreamEventType.TOKEN,
                                data={"content": event["content"]},
                                run_id=ctx.run_id,
                            )
                        elif event["type"] == "tool_call":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL,
                                data={"name": event["name"], "input": event["input"]},
                                run_id=ctx.run_id,
                            )
                        elif event["type"] == "tool_result":
                            yield StreamEvent(
                                type=StreamEventType.TOOL_RESULT,
                                data={"name": event["name"], "output": event["output"]},
                                run_id=ctx.run_id,
                            )
                        elif event["type"] == "task_complete":
                            current_task_index = event["task_index"] + 1
                            task_outputs.append(
                                {
                                    "task_index": event["task_index"],
                                    "agent": event["agent"],
                                }
                            )

                            yield StreamEvent(
                                type=StreamEventType.NODE_END,
                                data={
                                    "name": f"agent:{event['agent']}",
                                    "task_index": event["task_index"],
                                },
                                run_id=ctx.run_id,
                            )

                            # Checkpoint after task
                            if self.checkpoint_tasks:
                                await self.on_checkpoint(
                                    {
                                        "state": {
                                            "completed_tasks": current_task_index,
                                            "current_agent": event["agent"],
                                        },
                                        "task_outputs": task_outputs.copy(),
                                    },
                                    ctx,
                                )
                                yield StreamEvent(
                                    type=StreamEventType.CHECKPOINT,
                                    data={
                                        "task_index": event["task_index"],
                                        "agent": event["agent"],
                                    },
                                    run_id=ctx.run_id,
                                )
                    except Empty:
                        await asyncio.sleep(0.05)

                # Get final result
                result = await crew_task

                # Crew complete
                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    data={"name": "crew"},
                    run_id=ctx.run_id,
                )

                # Final checkpoint
                if self.checkpoint_tasks:
                    await self.on_checkpoint(
                        {
                            "state": {
                                "completed": True,
                                "completed_tasks": len(self.crew.tasks),
                            },
                            "task_outputs": task_outputs,
                            "context": {"result": str(result) if result else None},
                        },
                        ctx,
                    )
                    yield StreamEvent(
                        type=StreamEventType.CHECKPOINT,
                        data={"step": "completed"},
                        run_id=ctx.run_id,
                    )

                # Final result
                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={
                        "result": str(result) if result else None,
                        "raw": result.raw if hasattr(result, "raw") else None,
                    },
                    run_id=ctx.run_id,
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=ctx.run_id,
            )

    async def stream_verbose(
        self, input: Any, ctx: AdapterContext | Any
    ) -> AsyncIterator[StreamEvent]:
        """Stream with verbose output including agent thoughts.

        This method enables CrewAI's verbose mode to capture agent
        reasoning and tool usage.

        Args:
            input: The input to the crew
            ctx: The adapter context

        Yields:
            Detailed StreamEvent objects including agent thoughts
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # Enable verbose mode temporarily
        original_verbose = self.crew.verbose
        self.crew.verbose = True

        try:

            async def capture_output() -> Any:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.crew.kickoff(inputs=input),
                )

            # Start the crew
            task = asyncio.create_task(capture_output())

            # Yield crew start
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "crew", "verbose": True},
                run_id=adapter_ctx.run_id,
            )

            # Wait for completion
            result = await task

            yield StreamEvent(
                type=StreamEventType.DONE,
                data={"result": str(result) if result else None},
                run_id=adapter_ctx.run_id,
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )
        finally:
            self.crew.verbose = original_verbose

    def get_agent_info(self) -> list[dict[str, Any]]:
        """Get information about agents in the crew.

        Returns:
            List of agent info dicts with role, goal, etc.
        """
        return [
            {
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory[:200] if agent.backstory else None,
                "tools": [t.name for t in (agent.tools or []) if hasattr(t, "name")],
            }
            for agent in self.crew.agents
        ]

    def get_task_info(self) -> list[dict[str, Any]]:
        """Get information about tasks in the crew.

        Returns:
            List of task info dicts with description, agent, etc.
        """
        return [
            {
                "description": task.description[:200] if task.description else None,
                "agent": task.agent.role if task.agent else None,
                "expected_output": task.expected_output[:100] if task.expected_output else None,
            }
            for task in self.crew.tasks
        ]

    def with_verbose(self, verbose: bool = True) -> CrewAIAdapter:
        """Create a new adapter with verbose mode.

        Args:
            verbose: Whether to enable verbose mode

        Returns:
            A new CrewAIAdapter with verbose setting
        """
        new_crew = self.crew
        new_crew.verbose = verbose
        return CrewAIAdapter(
            new_crew,
            stream_agent_output=self.stream_agent_output,
            stream_task_output=self.stream_task_output,
            checkpoint_tasks=self.checkpoint_tasks,
        )

    def with_checkpoints(self, enabled: bool = True) -> CrewAIAdapter:
        """Create a new adapter with checkpoint configuration.

        Args:
            enabled: Whether to checkpoint after each task

        Returns:
            A new CrewAIAdapter with the specified checkpoint settings
        """
        return CrewAIAdapter(
            self.crew,
            stream_agent_output=self.stream_agent_output,
            stream_task_output=self.stream_task_output,
            checkpoint_tasks=enabled,
        )

    def with_streaming(self, agent_output: bool = True, task_output: bool = True) -> CrewAIAdapter:
        """Create a new adapter with different streaming settings.

        Args:
            agent_output: Whether to stream per-agent output
            task_output: Whether to stream per-task output

        Returns:
            A new CrewAIAdapter with the specified streaming settings
        """
        return CrewAIAdapter(
            self.crew,
            stream_agent_output=agent_output,
            stream_task_output=task_output,
            checkpoint_tasks=self.checkpoint_tasks,
        )
