"""Semantic Kernel adapter for FastAgentic.

This adapter wraps Microsoft Semantic Kernel to expose SK agents
and functions via FastAgentic endpoints with streaming support.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    pass  # Semantic Kernel types would be imported here


class SemanticKernelAdapter(BaseAdapter):
    """Adapter for Microsoft Semantic Kernel.

    Wraps Semantic Kernel agents and functions to work with FastAgentic's
    endpoint system, providing streaming, checkpointing, and observability.

    Example:
        import semantic_kernel as sk
        from fastagentic.adapters.semantic_kernel import SemanticKernelAdapter

        kernel = sk.Kernel()
        # Add plugins, services, etc.
        adapter = SemanticKernelAdapter(kernel, function_name="chat")

        @agent_endpoint(path="/chat", runnable=adapter, stream=True)
        async def chat(input: ChatInput) -> ChatOutput:
            ...
    """

    def __init__(
        self,
        kernel: Any,
        *,
        function_name: str | None = None,
        plugin_name: str | None = None,
        agent: Any | None = None,
        settings: Any | None = None,
        checkpoint_functions: bool = True,
    ) -> None:
        """Initialize the Semantic Kernel adapter.

        Args:
            kernel: A Semantic Kernel Kernel instance
            function_name: Name of the function to invoke (if using functions)
            plugin_name: Name of the plugin containing the function
            agent: Optional SK Agent instance for agent-based workflows
            settings: Optional prompt execution settings
            checkpoint_functions: Whether to create checkpoints after function calls
        """
        self.kernel = kernel
        self.function_name = function_name
        self.plugin_name = plugin_name
        self.agent = agent
        self.settings = settings
        self.checkpoint_functions = checkpoint_functions

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run Semantic Kernel and return the result.

        Args:
            input: The input to the kernel function or agent
            ctx: The adapter context

        Returns:
            The kernel function or agent output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            adapter_ctx.state["chat_history"] = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True

        # Extract input parameters
        arguments = self._build_arguments(input)

        # Track function calls for tool events
        function_calls: list[dict[str, Any]] = []

        try:
            if self.agent is not None:
                # Agent-based invocation
                result = await self._invoke_agent(arguments, adapter_ctx)
            elif self.function_name:
                # Function-based invocation
                result = await self._invoke_function(arguments, adapter_ctx, function_calls)
            else:
                # Direct prompt invocation
                result = await self._invoke_prompt(arguments, adapter_ctx)

            extracted_result = self._extract_result(result)

            # Create checkpoint with final state
            if self.checkpoint_functions:
                await self.on_checkpoint(
                    {
                        "state": {"completed": True},
                        "messages": adapter_ctx.state.get("chat_history", []),
                        "function_calls": function_calls,
                        "context": {
                            "result": (
                                extracted_result
                                if isinstance(extracted_result, (dict, str, int, float, bool))
                                else str(extracted_result)
                            )
                        },
                    },
                    adapter_ctx,
                )

            return extracted_result

        except Exception as e:
            raise RuntimeError(f"Semantic Kernel invocation failed: {e}") from e

    async def _invoke_agent(self, arguments: dict[str, Any], ctx: AdapterContext) -> Any:
        """Invoke SK agent."""
        assert self.agent is not None
        # Get message from arguments
        message = arguments.get("message", arguments.get("input", ""))

        # Create chat history if needed
        chat_history = ctx.state.get("chat_history", [])

        # Invoke agent
        response = await self.agent.invoke(
            messages=chat_history + [{"role": "user", "content": message}]
        )

        # Track in state
        ctx.state["chat_history"] = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": str(response)},
        ]

        return response

    async def _invoke_function(
        self,
        arguments: dict[str, Any],
        _ctx: AdapterContext,
        function_calls: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Invoke SK function."""
        # Get function from kernel
        if self.plugin_name:
            function = self.kernel.get_function(self.plugin_name, self.function_name)
        else:
            function = self.kernel.get_function_from_prompt(self.function_name)

        # Create kernel arguments
        from semantic_kernel.functions import KernelArguments

        kernel_args = KernelArguments(**arguments)

        if self.settings:
            kernel_args.execution_settings = self.settings

        # Track function call if list provided
        if function_calls is not None:
            function_calls.append(
                {
                    "name": f"{self.plugin_name}.{self.function_name}"
                    if self.plugin_name
                    else self.function_name,
                    "input": arguments,
                }
            )

        # Invoke function
        result = await self.kernel.invoke(function, kernel_args)

        # Track function result
        if function_calls is not None:
            function_calls[-1]["output"] = self._extract_result(result)

        return result

    async def _invoke_prompt(self, arguments: dict[str, Any], _ctx: AdapterContext) -> Any:
        """Invoke SK with a prompt template."""
        prompt = arguments.get("message", arguments.get("prompt", ""))

        result = await self.kernel.invoke_prompt(
            prompt,
            settings=self.settings,
        )

        return result

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from Semantic Kernel.

        Args:
            input: The input to the kernel
            ctx: The adapter context

        Yields:
            StreamEvent objects for tokens, function calls, etc.
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            adapter_ctx.state["chat_history"] = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )

        arguments = self._build_arguments(input)

        try:
            if self.agent is not None:
                async for event in self._stream_agent(arguments, adapter_ctx):
                    yield event
            elif self.function_name:
                async for event in self._stream_function(arguments, adapter_ctx):
                    yield event
            else:
                async for event in self._stream_prompt(arguments, adapter_ctx):
                    yield event

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    async def _stream_agent(
        self, arguments: dict[str, Any], ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream from SK agent."""
        assert self.agent is not None
        message = arguments.get("message", arguments.get("input", ""))
        chat_history = ctx.state.get("chat_history", [])

        full_response = ""

        async for chunk in self.agent.invoke_stream(
            messages=chat_history + [{"role": "user", "content": message}]
        ):
            content = str(chunk)
            full_response += content

            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": content},
                run_id=ctx.run_id,
            )

        # Update chat history
        ctx.state["chat_history"] = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response},
        ]

        # Create checkpoint with final state
        if self.checkpoint_functions:
            await self.on_checkpoint(
                {
                    "state": {"completed": True},
                    "messages": ctx.state["chat_history"],
                    "context": {"result": full_response},
                },
                ctx,
            )
            yield StreamEvent(
                type=StreamEventType.CHECKPOINT,
                data={"step": "completed"},
                run_id=ctx.run_id,
            )

        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": full_response},
            run_id=ctx.run_id,
        )

    async def _stream_function(
        self, arguments: dict[str, Any], ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream from SK function."""
        if self.plugin_name:
            function = self.kernel.get_function(self.plugin_name, self.function_name)
        else:
            function = self.kernel.get_function_from_prompt(self.function_name)

        from semantic_kernel.functions import KernelArguments

        kernel_args = KernelArguments(**arguments)

        if self.settings:
            kernel_args.execution_settings = self.settings

        # Emit tool call event
        function_name = (
            f"{self.plugin_name}.{self.function_name}" if self.plugin_name else self.function_name
        )
        yield StreamEvent(
            type=StreamEventType.TOOL_CALL,
            data={"name": function_name, "input": arguments},
            run_id=ctx.run_id,
        )

        full_response = ""

        async for chunk in self.kernel.invoke_stream(function, kernel_args):
            content = str(chunk)
            full_response += content

            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": content},
                run_id=ctx.run_id,
            )

        # Emit tool result event
        yield StreamEvent(
            type=StreamEventType.TOOL_RESULT,
            data={"name": function_name, "output": full_response},
            run_id=ctx.run_id,
        )

        # Create checkpoint with final state
        if self.checkpoint_functions:
            await self.on_checkpoint(
                {
                    "state": {"completed": True, "function": function_name},
                    "function_calls": [
                        {"name": function_name, "input": arguments, "output": full_response}
                    ],
                    "context": {"result": full_response},
                },
                ctx,
            )
            yield StreamEvent(
                type=StreamEventType.CHECKPOINT,
                data={"step": "completed", "function": function_name},
                run_id=ctx.run_id,
            )

        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": full_response},
            run_id=ctx.run_id,
        )

    async def _stream_prompt(
        self, arguments: dict[str, Any], ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream from SK prompt."""
        prompt = arguments.get("message", arguments.get("prompt", ""))

        full_response = ""

        async for chunk in self.kernel.invoke_prompt_stream(
            prompt,
            settings=self.settings,
        ):
            content = str(chunk)
            full_response += content

            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": content},
                run_id=ctx.run_id,
            )

        # Create checkpoint with final state
        if self.checkpoint_functions:
            await self.on_checkpoint(
                {
                    "state": {"completed": True},
                    "context": {"result": full_response, "prompt": prompt},
                },
                ctx,
            )
            yield StreamEvent(
                type=StreamEventType.CHECKPOINT,
                data={"step": "completed"},
                run_id=ctx.run_id,
            )

        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": full_response},
            run_id=ctx.run_id,
        )

    def _build_arguments(self, input: Any) -> dict[str, Any]:
        """Build kernel arguments from input."""
        if isinstance(input, dict):
            return input

        if hasattr(input, "model_dump"):
            result: dict[str, Any] = input.model_dump()
            return result

        if isinstance(input, str):
            return {"message": input}

        return {"input": input}

    def _extract_result(self, result: Any) -> Any:
        """Extract result value from SK response."""
        if hasattr(result, "value"):
            return result.value

        if hasattr(result, "result"):
            return result.result

        return result

    def with_function(
        self, function_name: str, plugin_name: str | None = None
    ) -> SemanticKernelAdapter:
        """Create a new adapter targeting a different function.

        Args:
            function_name: The function name to invoke
            plugin_name: Optional plugin name

        Returns:
            A new SemanticKernelAdapter with the updated function
        """
        return SemanticKernelAdapter(
            self.kernel,
            function_name=function_name,
            plugin_name=plugin_name or self.plugin_name,
            agent=self.agent,
            settings=self.settings,
            checkpoint_functions=self.checkpoint_functions,
        )

    def with_settings(self, settings: Any) -> SemanticKernelAdapter:
        """Create a new adapter with different settings.

        Args:
            settings: New prompt execution settings

        Returns:
            A new SemanticKernelAdapter with the updated settings
        """
        return SemanticKernelAdapter(
            self.kernel,
            function_name=self.function_name,
            plugin_name=self.plugin_name,
            agent=self.agent,
            settings=settings,
            checkpoint_functions=self.checkpoint_functions,
        )

    def with_checkpoints(self, enabled: bool = True) -> SemanticKernelAdapter:
        """Create a new adapter with checkpoint configuration.

        Args:
            enabled: Whether to checkpoint after function calls

        Returns:
            A new SemanticKernelAdapter with the specified checkpoint settings
        """
        return SemanticKernelAdapter(
            self.kernel,
            function_name=self.function_name,
            plugin_name=self.plugin_name,
            agent=self.agent,
            settings=self.settings,
            checkpoint_functions=enabled,
        )

    def with_agent(self, agent: Any) -> SemanticKernelAdapter:
        """Create a new adapter with a different agent.

        Args:
            agent: New SK Agent instance

        Returns:
            A new SemanticKernelAdapter with the updated agent
        """
        return SemanticKernelAdapter(
            self.kernel,
            function_name=self.function_name,
            plugin_name=self.plugin_name,
            agent=agent,
            settings=self.settings,
            checkpoint_functions=self.checkpoint_functions,
        )
