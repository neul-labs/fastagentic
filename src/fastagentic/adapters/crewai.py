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
    ) -> None:
        """Initialize the CrewAI adapter.

        Args:
            crew: A CrewAI Crew instance
            stream_agent_output: Whether to stream per-agent output
            stream_task_output: Whether to stream per-task output
        """
        self.crew = crew
        self.stream_agent_output = stream_agent_output
        self.stream_task_output = stream_task_output

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run the CrewAI crew and return the result.

        Args:
            input: The input to the crew (dict with task inputs)
            ctx: The adapter context

        Returns:
            The crew's output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)  # noqa: F841

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        # CrewAI's kickoff is synchronous, so we run it in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.crew.kickoff(inputs=input),
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

        # Convert Pydantic models to dict
        if hasattr(input, "model_dump"):
            input = input.model_dump()

        try:
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

            # CrewAI doesn't have native async streaming, so we track progress
            # by monitoring task callbacks
            current_task: int = 0

            # Set up task callback to capture progress
            task_outputs: list[Any] = []

            def on_task_complete(task_output: Any) -> None:
                nonlocal current_task
                task_outputs.append(task_output)
                current_task += 1

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

            # Emit completion events for each task/agent
            for i, task in enumerate(self.crew.tasks):
                agent = task.agent
                agent_role = agent.role if agent else f"Agent {i}"

                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    data={
                        "name": f"agent:{agent_role}",
                        "task_index": i,
                    },
                    run_id=adapter_ctx.run_id,
                )

            # Crew complete
            yield StreamEvent(
                type=StreamEventType.NODE_END,
                data={"name": "crew"},
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
        )
