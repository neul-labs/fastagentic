"""DSPy adapter for FastAgentic.

This adapter wraps DSPy modules and programs to expose them
via FastAgentic endpoints with streaming support.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    pass  # DSPy types would be imported here


class DSPyAdapter(BaseAdapter):
    """Adapter for DSPy.

    Wraps DSPy modules, signatures, and programs to work with
    FastAgentic's endpoint system with optimized prompt support.

    Example:
        import dspy
        from fastagentic.adapters.dspy import DSPyAdapter

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        qa_module = dspy.ChainOfThought(QA)
        adapter = DSPyAdapter(qa_module)

        @agent_endpoint(path="/qa", runnable=adapter)
        async def qa(input: QAInput) -> QAOutput:
            ...
    """

    def __init__(
        self,
        module: Any,
        *,
        lm: Any | None = None,
        trace: bool = False,
        stream_fields: list[str] | None = None,
    ) -> None:
        """Initialize the DSPy adapter.

        Args:
            module: A DSPy module or program
            lm: Optional language model to use (overrides dspy.settings)
            trace: Whether to include trace information in responses
            stream_fields: Output fields to stream (None = auto-detect)
        """
        self.module = module
        self.lm = lm
        self.trace = trace
        self.stream_fields = stream_fields

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run DSPy module and return the result.

        Args:
            input: The input to the module
            ctx: The adapter context

        Returns:
            The module output
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            # DSPy doesn't have native resume; return cached result if completed
            cached_result = checkpoint.get("context", {}).get("result")
            if cached_result and checkpoint.get("state", {}).get("completed"):
                adapter_ctx.agent_ctx.run._is_resumed = True
                return cached_result

        kwargs = self._build_kwargs(input)

        try:
            # Set LM if provided
            if self.lm is not None:
                import dspy

                with dspy.context(lm=self.lm):
                    result = await self._run_module(kwargs, adapter_ctx)
            else:
                result = await self._run_module(kwargs, adapter_ctx)

            formatted_result = self._format_result(result, adapter_ctx)

            # Create checkpoint with result
            checkpoint_data: dict[str, Any] = {
                "state": {"completed": True},
                "context": {"result": formatted_result},
            }
            if self.trace:
                trace_data = self._extract_trace(result)
                if trace_data:
                    checkpoint_data["context"]["trace"] = trace_data

            await self.on_checkpoint(checkpoint_data, adapter_ctx)

            return formatted_result

        except Exception as e:
            raise RuntimeError(f"DSPy invocation failed: {e}") from e

    async def _run_module(self, kwargs: dict[str, Any], _ctx: AdapterContext) -> Any:
        """Run the DSPy module."""
        import asyncio

        # DSPy modules are typically synchronous
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def run_sync() -> Any:
            return self.module(**kwargs)

        result = await loop.run_in_executor(None, run_sync)

        return result

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from DSPy module.

        Supports native streaming via dspy.streamify() if available,
        otherwise falls back to simulated streaming.

        Args:
            input: The input to the module
            ctx: The adapter context

        Yields:
            StreamEvent objects
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            cached_result = checkpoint.get("context", {}).get("result")
            if cached_result and checkpoint.get("state", {}).get("completed"):
                adapter_ctx.agent_ctx.run._is_resumed = True
                yield StreamEvent(
                    type=StreamEventType.NODE_START,
                    data={"name": "__resume__", "cached": True},
                    run_id=adapter_ctx.run_id,
                )
                # Stream cached result
                output_text = (
                    str(cached_result.get(self._get_output_fields()[0], ""))
                    if self._get_output_fields()
                    else str(cached_result)
                )
                for i in range(0, len(output_text), 20):
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": output_text[i : i + 20]},
                        run_id=adapter_ctx.run_id,
                    )
                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={"result": cached_result},
                    run_id=adapter_ctx.run_id,
                )
                return

        kwargs = self._build_kwargs(input)

        try:
            # Check if native streaming is available
            if self._supports_native_streaming():
                async for event in self._stream_native(kwargs, adapter_ctx):
                    yield event
            else:
                async for event in self._stream_simulated(kwargs, adapter_ctx):
                    yield event

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    def _supports_native_streaming(self) -> bool:
        """Check if DSPy native streaming is available."""
        try:
            import dspy

            return hasattr(dspy, "streamify")
        except ImportError:
            return False

    async def _stream_native(
        self, kwargs: dict[str, Any], ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Use DSPy's native streamify for true token streaming."""
        import dspy

        # Determine which fields to stream
        fields_to_stream = self.stream_fields or self._get_output_fields()
        if not fields_to_stream:
            fields_to_stream = ["answer"]  # Default fallback

        yield StreamEvent(
            type=StreamEventType.NODE_START,
            data={"name": self.module.__class__.__name__, "fields": fields_to_stream},
            run_id=ctx.run_id,
        )

        try:
            # Import streaming components
            from dspy.streaming import StreamListener, StreamResponse

            # Create stream listeners for each field
            listeners = [
                StreamListener(signature_field_name=field, allow_reuse=True)
                for field in fields_to_stream
                if isinstance(field, str)
            ]

            # Wrap module with streamify
            stream_module = dspy.streamify(self.module, stream_listeners=listeners)

            current_field: str | None = None
            final_prediction: Any = None

            # Execute with optional LM context
            if self.lm is not None:
                with dspy.context(lm=self.lm):
                    output_generator = stream_module(**kwargs)
            else:
                output_generator = stream_module(**kwargs)

            # Consume the generator
            for chunk in output_generator:
                if isinstance(chunk, StreamResponse):
                    # Track field changes
                    if chunk.signature_field_name != current_field:
                        if current_field:
                            yield StreamEvent(
                                type=StreamEventType.NODE_END,
                                data={"name": f"field:{current_field}"},
                                run_id=ctx.run_id,
                            )
                        current_field = chunk.signature_field_name
                        yield StreamEvent(
                            type=StreamEventType.NODE_START,
                            data={"name": f"field:{current_field}"},
                            run_id=ctx.run_id,
                        )

                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": chunk.chunk, "field": chunk.signature_field_name},
                        run_id=ctx.run_id,
                    )
                else:
                    # Final prediction object
                    final_prediction = chunk

            # Close last field
            if current_field:
                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    data={"name": f"field:{current_field}"},
                    run_id=ctx.run_id,
                )

            if final_prediction:
                # Extract and yield tool events from ReAct trace
                tool_events = self._extract_tool_events(final_prediction)
                for tool_event in tool_events:
                    if tool_event["type"] == "tool_call":
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            data={"name": tool_event["name"], "input": tool_event["input"]},
                            run_id=ctx.run_id,
                        )
                    elif tool_event["type"] == "tool_result":
                        yield StreamEvent(
                            type=StreamEventType.TOOL_RESULT,
                            data={"name": tool_event["name"], "output": tool_event["output"]},
                            run_id=ctx.run_id,
                        )

                # Yield trace if enabled
                if self.trace:
                    trace_data = self._extract_trace(final_prediction)
                    if trace_data:
                        yield StreamEvent(
                            type=StreamEventType.TRACE,
                            data=trace_data,
                            run_id=ctx.run_id,
                        )

                formatted_result = self._format_result(final_prediction, ctx)

                # Checkpoint
                await self.on_checkpoint(
                    {
                        "state": {"completed": True},
                        "context": {"result": formatted_result},
                    },
                    ctx,
                )
                yield StreamEvent(
                    type=StreamEventType.CHECKPOINT,
                    data={"step": "completed"},
                    run_id=ctx.run_id,
                )

                yield StreamEvent(
                    type=StreamEventType.NODE_END,
                    data={"name": self.module.__class__.__name__},
                    run_id=ctx.run_id,
                )

                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={"result": formatted_result},
                    run_id=ctx.run_id,
                )

        except ImportError:
            # Fall back to simulated if streaming imports fail
            async for event in self._stream_simulated(kwargs, ctx):
                yield event

    async def _stream_simulated(
        self, kwargs: dict[str, Any], ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Simulated streaming by chunking output."""
        # Yield start event
        yield StreamEvent(
            type=StreamEventType.MESSAGE,
            data={"status": "processing"},
            run_id=ctx.run_id,
        )

        # Run the module
        if self.lm is not None:
            import dspy

            with dspy.context(lm=self.lm):
                result = await self._run_module(kwargs, ctx)
        else:
            result = await self._run_module(kwargs, ctx)

        # Extract and yield tool events from ReAct trace
        tool_events = self._extract_tool_events(result)
        for tool_event in tool_events:
            if tool_event["type"] == "tool_call":
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    data={"name": tool_event["name"], "input": tool_event["input"]},
                    run_id=ctx.run_id,
                )
            elif tool_event["type"] == "tool_result":
                yield StreamEvent(
                    type=StreamEventType.TOOL_RESULT,
                    data={"name": tool_event["name"], "output": tool_event["output"]},
                    run_id=ctx.run_id,
                )

        # Extract output text
        output_text = self._extract_output_text(result)

        # Simulate streaming by chunking the output
        chunk_size = 20
        for i in range(0, len(output_text), chunk_size):
            chunk = output_text[i : i + chunk_size]
            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": chunk},
                run_id=ctx.run_id,
            )

        # Yield trace if enabled
        if self.trace:
            trace_data = self._extract_trace(result)
            if trace_data:
                yield StreamEvent(
                    type=StreamEventType.TRACE,
                    data=trace_data,
                    run_id=ctx.run_id,
                )

        formatted_result = self._format_result(result, ctx)

        # Checkpoint
        await self.on_checkpoint(
            {
                "state": {"completed": True},
                "context": {"result": formatted_result},
            },
            ctx,
        )
        yield StreamEvent(
            type=StreamEventType.CHECKPOINT,
            data={"step": "completed"},
            run_id=ctx.run_id,
        )

        # Yield done event
        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": formatted_result},
            run_id=ctx.run_id,
        )

    def _extract_tool_events(self, result: Any) -> list[dict[str, Any]]:
        """Extract tool call/result events from DSPy ReAct trace."""
        tool_events: list[dict[str, Any]] = []

        # Check for ReAct-style action history
        if hasattr(result, "actions") or hasattr(result, "action_history"):
            actions = getattr(result, "actions", None) or getattr(result, "action_history", [])
            for action in actions:
                if hasattr(action, "tool"):
                    tool_events.append(
                        {
                            "type": "tool_call",
                            "name": action.tool,
                            "input": getattr(action, "tool_input", {}),
                        }
                    )
                    if hasattr(action, "observation"):
                        tool_events.append(
                            {
                                "type": "tool_result",
                                "name": action.tool,
                                "output": action.observation,
                            }
                        )

        # Check for tool calls in completions trace
        if hasattr(result, "completions"):
            for completion in result.completions:
                if hasattr(completion, "tool_calls"):
                    for tc in completion.tool_calls:
                        tool_events.append(
                            {
                                "type": "tool_call",
                                "name": tc.get("name", "unknown"),
                                "input": tc.get("arguments", {}),
                            }
                        )

        return tool_events

    def _build_kwargs(self, input: Any) -> dict[str, Any]:
        """Build kwargs for module invocation."""
        if isinstance(input, dict):
            return input

        if hasattr(input, "model_dump"):
            result: dict[str, Any] = input.model_dump()
            return result

        if isinstance(input, str):
            # Try to infer the input field name from the signature
            input_fields = self._get_input_fields()
            if input_fields:
                return {input_fields[0]: input}
            return {"question": input}  # Common default

        return {"input": input}

    def _get_input_fields(self) -> list[str]:
        """Get input field names from DSPy signature."""
        if hasattr(self.module, "signature"):
            sig = self.module.signature
            if hasattr(sig, "input_fields"):
                return list(sig.input_fields.keys())

        return []

    def _get_output_fields(self) -> list[str]:
        """Get output field names from DSPy signature."""
        if hasattr(self.module, "signature"):
            sig = self.module.signature
            if hasattr(sig, "output_fields"):
                return list(sig.output_fields.keys())

        return []

    def _extract_output_text(self, result: Any) -> str:
        """Extract main output text from result."""
        output_fields = self._get_output_fields()

        if output_fields:
            # Get the first output field
            primary_output = output_fields[0]
            if hasattr(result, primary_output):
                return str(getattr(result, primary_output))

        # Fallback to common field names
        for field in ("answer", "output", "response", "result", "text"):
            if hasattr(result, field):
                return str(getattr(result, field))

        return str(result)

    def _format_result(self, result: Any, _ctx: AdapterContext) -> dict[str, Any]:
        """Format DSPy result."""
        output: dict[str, Any] = {}

        # Extract all output fields
        output_fields = self._get_output_fields()
        for field in output_fields:
            if hasattr(result, field):
                output[field] = getattr(result, field)

        # If no specific fields, try to extract everything
        if not output:
            if hasattr(result, "__dict__"):
                for key, value in result.__dict__.items():
                    if not key.startswith("_"):
                        output[key] = value
            else:
                output["result"] = str(result)

        # Add trace if enabled
        if self.trace:
            trace_data = self._extract_trace(result)
            if trace_data:
                output["_trace"] = trace_data

        return output

    def _extract_trace(self, result: Any) -> dict[str, Any] | None:
        """Extract trace information from DSPy result."""
        trace: dict[str, Any] = {}

        # Extract reasoning/rationale if available (ChainOfThought)
        if hasattr(result, "rationale"):
            trace["rationale"] = result.rationale

        if hasattr(result, "reasoning"):
            trace["reasoning"] = result.reasoning

        # Extract completions/history
        if hasattr(result, "completions"):
            trace["completions"] = [
                {
                    "prompt": getattr(c, "prompt", None),
                    "response": getattr(c, "response", None),
                }
                for c in result.completions
            ]

        return trace if trace else None

    def with_lm(self, lm: Any) -> DSPyAdapter:
        """Create a new adapter with a different language model.

        Args:
            lm: The new language model

        Returns:
            A new DSPyAdapter
        """
        return DSPyAdapter(
            self.module,
            lm=lm,
            trace=self.trace,
            stream_fields=self.stream_fields,
        )

    def with_trace(self, trace: bool = True) -> DSPyAdapter:
        """Create a new adapter with trace enabled/disabled.

        Args:
            trace: Whether to include trace information

        Returns:
            A new DSPyAdapter
        """
        return DSPyAdapter(
            self.module,
            lm=self.lm,
            trace=trace,
            stream_fields=self.stream_fields,
        )

    def with_stream_fields(self, fields: list[str]) -> DSPyAdapter:
        """Create a new adapter with specific fields to stream.

        Args:
            fields: List of output field names to stream

        Returns:
            A new DSPyAdapter with the specified stream fields
        """
        return DSPyAdapter(
            self.module,
            lm=self.lm,
            trace=self.trace,
            stream_fields=fields,
        )


class DSPyProgramAdapter(DSPyAdapter):
    """Adapter for DSPy compiled programs.

    This adapter is specifically designed for DSPy programs that have
    been compiled/optimized with a teleprompter.

    Example:
        import dspy
        from dspy.teleprompt import BootstrapFewShot
        from fastagentic.adapters.dspy import DSPyProgramAdapter

        # Define and compile program
        class RAG(dspy.Module):
            def __init__(self):
                self.retrieve = dspy.Retrieve(k=3)
                self.generate = dspy.ChainOfThought("context, question -> answer")

            def forward(self, question):
                context = self.retrieve(question).passages
                return self.generate(context=context, question=question)

        teleprompter = BootstrapFewShot(metric=my_metric)
        compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

        adapter = DSPyProgramAdapter(compiled_rag)
    """

    def __init__(
        self,
        program: Any,
        *,
        lm: Any | None = None,
        trace: bool = False,
        include_retrieval: bool = True,
    ) -> None:
        """Initialize the DSPy program adapter.

        Args:
            program: A compiled DSPy program
            lm: Optional language model override
            trace: Whether to include trace information
            include_retrieval: Whether to include retrieval results in output
        """
        super().__init__(program, lm=lm, trace=trace)
        self.include_retrieval = include_retrieval

    async def _run_module(self, kwargs: dict[str, Any], ctx: AdapterContext) -> Any:
        """Run the compiled DSPy program."""
        import asyncio

        loop = asyncio.get_event_loop()

        def run_sync() -> Any:
            # Use forward() for Module subclasses
            if hasattr(self.module, "forward"):
                return self.module.forward(**kwargs)
            return self.module(**kwargs)

        result = await loop.run_in_executor(None, run_sync)

        # Store retrieval results in context if available
        if self.include_retrieval and hasattr(result, "passages"):
            ctx.state["retrieved_passages"] = result.passages

        return result

    def _format_result(self, result: Any, ctx: AdapterContext) -> dict[str, Any]:
        """Format compiled program result with retrieval info."""
        output = super()._format_result(result, ctx)

        # Add retrieved passages if enabled
        if self.include_retrieval:
            passages = ctx.state.get("retrieved_passages")
            if passages:
                output["sources"] = passages

        return output
