"""AutoGen adapter for FastAgentic.

This adapter wraps Microsoft AutoGen to expose multi-agent conversations
via FastAgentic endpoints with streaming support.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    pass  # AutoGen types would be imported here


class AutoGenAdapter(BaseAdapter):
    """Adapter for Microsoft AutoGen.

    Wraps AutoGen agents and group chats to work with FastAgentic's
    endpoint system, providing streaming and multi-agent conversation support.

    Example:
        from autogen import AssistantAgent, UserProxyAgent
        from fastagentic.adapters.autogen import AutoGenAdapter

        assistant = AssistantAgent("assistant", llm_config=config)
        user_proxy = UserProxyAgent("user", code_execution_config={"use_docker": False})

        adapter = AutoGenAdapter(
            initiator=user_proxy,
            recipient=assistant,
        )

        @agent_endpoint(path="/chat", runnable=adapter, stream=True)
        async def chat(input: ChatInput) -> ChatOutput:
            ...
    """

    def __init__(
        self,
        initiator: Any,
        recipient: Any,
        *,
        group_chat: Any | None = None,
        max_turns: int | None = None,
        clear_history: bool = False,
        silent: bool = True,
        checkpoint_turns: bool = True,
    ) -> None:
        """Initialize the AutoGen adapter.

        Args:
            initiator: The agent that initiates the conversation
            recipient: The agent that receives messages (or GroupChat)
            group_chat: Optional GroupChat for multi-agent scenarios
            max_turns: Maximum number of conversation turns
            clear_history: Whether to clear history between runs
            silent: Whether to suppress console output
            checkpoint_turns: Whether to create checkpoints after each turn
        """
        self.initiator = initiator
        self.recipient = recipient
        self.group_chat = group_chat
        self.max_turns = max_turns
        self.clear_history = clear_history
        self.silent = silent
        self.checkpoint_turns = checkpoint_turns
        self._message_buffer: list[dict[str, Any]] = []

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run AutoGen conversation and return the result.

        Args:
            input: The input message to start the conversation
            ctx: The adapter context

        Returns:
            The final conversation result
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            self._message_buffer = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True
        else:
            # Clear buffer
            self._message_buffer = []

        message = self._extract_message(input)

        try:
            if self.group_chat is not None:
                result = await self._invoke_group_chat(message, adapter_ctx)
            else:
                result = await self._invoke_two_agent(message, adapter_ctx)

            # Create checkpoint with final state
            if self.checkpoint_turns:
                await self.on_checkpoint(
                    {
                        "state": {"completed": True, "turn": len(self._message_buffer)},
                        "messages": self._message_buffer.copy(),
                        "context": {"result": result},
                    },
                    adapter_ctx,
                )

            return result

        except Exception as e:
            raise RuntimeError(f"AutoGen invocation failed: {e}") from e

    async def _invoke_two_agent(self, message: str, _ctx: AdapterContext) -> dict[str, Any]:
        """Invoke two-agent conversation."""
        # Register message callback
        self._register_message_callback()

        # Initiate chat
        chat_result = await self.initiator.a_initiate_chat(
            self.recipient,
            message=message,
            max_turns=self.max_turns,
            clear_history=self.clear_history,
            silent=self.silent,
        )

        # Extract result
        return self._extract_chat_result(chat_result)

    async def _invoke_group_chat(self, message: str, _ctx: AdapterContext) -> dict[str, Any]:
        """Invoke group chat conversation."""
        from autogen import GroupChatManager

        # Create group chat manager
        manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.initiator.llm_config,
        )

        # Register message callback
        self._register_message_callback()

        # Initiate chat with manager
        chat_result = await self.initiator.a_initiate_chat(
            manager,
            message=message,
            max_turns=self.max_turns,
            clear_history=self.clear_history,
            silent=self.silent,
        )

        return self._extract_chat_result(chat_result)

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from AutoGen conversation.

        Args:
            input: The input message
            ctx: The adapter context

        Yields:
            StreamEvent objects for messages, tool calls, etc.
        """
        adapter_ctx = self._ensure_adapter_context(ctx)

        # Check for resume from checkpoint
        checkpoint = await self.on_resume(adapter_ctx)
        if checkpoint:
            self._message_buffer = checkpoint.get("messages", [])
            adapter_ctx.agent_ctx.run._is_resumed = True
            yield StreamEvent(
                type=StreamEventType.NODE_START,
                data={"name": "__resume__", "checkpoint": checkpoint},
                run_id=adapter_ctx.run_id,
            )
        else:
            # Clear buffer
            self._message_buffer = []

        message = self._extract_message(input)

        try:
            # Register streaming callback
            self._register_streaming_callback(adapter_ctx)

            if self.group_chat is not None:
                async for event in self._stream_group_chat(message, adapter_ctx):
                    yield event
            else:
                async for event in self._stream_two_agent(message, adapter_ctx):
                    yield event

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    async def _stream_two_agent(
        self, message: str, ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream two-agent conversation."""
        import asyncio

        # Try native streaming first (AutoGen v0.4+)
        if hasattr(self.recipient, "on_messages_stream"):
            async for event in self._stream_native(message, ctx):
                yield event
            return

        # Fallback to polling-based streaming
        # Start conversation in background
        task = asyncio.create_task(
            self.initiator.a_initiate_chat(
                self.recipient,
                message=message,
                max_turns=self.max_turns,
                clear_history=self.clear_history,
                silent=self.silent,
            )
        )

        # Yield messages as they come in
        last_index = 0
        turn_count = 0
        while not task.done():
            await asyncio.sleep(0.1)

            # Check for new messages
            while last_index < len(self._message_buffer):
                msg = self._message_buffer[last_index]
                yield self._message_to_event(msg, ctx)
                last_index += 1
                turn_count += 1

                # Checkpoint after each turn
                if self.checkpoint_turns:
                    await self.on_checkpoint(
                        {
                            "state": {"turn": turn_count},
                            "messages": self._message_buffer[:last_index].copy(),
                        },
                        ctx,
                    )
                    yield StreamEvent(
                        type=StreamEventType.CHECKPOINT,
                        data={"turn": turn_count},
                        run_id=ctx.run_id,
                    )

        # Get final result
        chat_result = await task

        # Yield any remaining messages
        while last_index < len(self._message_buffer):
            msg = self._message_buffer[last_index]
            yield self._message_to_event(msg, ctx)
            last_index += 1
            turn_count += 1

        # Final checkpoint
        if self.checkpoint_turns:
            result = self._extract_chat_result(chat_result)
            await self.on_checkpoint(
                {
                    "state": {"completed": True, "turn": turn_count},
                    "messages": self._message_buffer.copy(),
                    "context": {"result": result},
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
            data={"result": self._extract_chat_result(chat_result)},
            run_id=ctx.run_id,
        )

    async def _stream_native(self, message: str, ctx: AdapterContext) -> AsyncIterator[StreamEvent]:
        """Stream using native AutoGen v0.4+ streaming API."""
        turn_count = 0

        # Build initial message
        messages = [{"role": "user", "content": message}]

        try:
            async for response in self.recipient.on_messages_stream(messages):
                # Check for streaming chunk events
                if hasattr(response, "content"):
                    content = response.content
                    if content:
                        yield StreamEvent(
                            type=StreamEventType.TOKEN,
                            data={"content": content},
                            run_id=ctx.run_id,
                        )
                        self._message_buffer.append(
                            {
                                "sender": getattr(self.recipient, "name", "agent"),
                                "content": content,
                                "role": "assistant",
                            }
                        )

                # Check for complete message
                if hasattr(response, "chat_message") and response.chat_message:
                    turn_count += 1
                    if self.checkpoint_turns:
                        await self.on_checkpoint(
                            {
                                "state": {"turn": turn_count},
                                "messages": self._message_buffer.copy(),
                            },
                            ctx,
                        )
                        yield StreamEvent(
                            type=StreamEventType.CHECKPOINT,
                            data={"turn": turn_count},
                            run_id=ctx.run_id,
                        )

            # Final checkpoint
            if self.checkpoint_turns:
                await self.on_checkpoint(
                    {
                        "state": {"completed": True, "turn": turn_count},
                        "messages": self._message_buffer.copy(),
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
                data={"result": {"messages": self._message_buffer}},
                run_id=ctx.run_id,
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=ctx.run_id,
            )

    async def _stream_group_chat(
        self, message: str, ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream group chat conversation."""
        import asyncio

        from autogen import GroupChatManager

        manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.initiator.llm_config,
        )

        # Try native streaming first (AutoGen v0.4+)
        if hasattr(manager, "run_stream"):
            async for event in self._stream_group_native(manager, message, ctx):
                yield event
            return

        # Fallback to polling-based streaming
        # Start conversation in background
        task = asyncio.create_task(
            self.initiator.a_initiate_chat(
                manager,
                message=message,
                max_turns=self.max_turns,
                clear_history=self.clear_history,
                silent=self.silent,
            )
        )

        # Yield messages as they come in
        last_index = 0
        turn_count = 0
        while not task.done():
            await asyncio.sleep(0.1)

            while last_index < len(self._message_buffer):
                msg = self._message_buffer[last_index]
                yield self._message_to_event(msg, ctx)
                last_index += 1
                turn_count += 1

                # Checkpoint after each turn
                if self.checkpoint_turns:
                    await self.on_checkpoint(
                        {
                            "state": {"turn": turn_count},
                            "messages": self._message_buffer[:last_index].copy(),
                        },
                        ctx,
                    )
                    yield StreamEvent(
                        type=StreamEventType.CHECKPOINT,
                        data={"turn": turn_count},
                        run_id=ctx.run_id,
                    )

        chat_result = await task

        while last_index < len(self._message_buffer):
            msg = self._message_buffer[last_index]
            yield self._message_to_event(msg, ctx)
            last_index += 1
            turn_count += 1

        # Final checkpoint
        if self.checkpoint_turns:
            result = self._extract_chat_result(chat_result)
            await self.on_checkpoint(
                {
                    "state": {"completed": True, "turn": turn_count},
                    "messages": self._message_buffer.copy(),
                    "context": {"result": result},
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
            data={"result": self._extract_chat_result(chat_result)},
            run_id=ctx.run_id,
        )

    async def _stream_group_native(
        self, manager: Any, message: str, ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream using native AutoGen v0.4+ group chat streaming API."""
        turn_count = 0

        try:
            async for item in manager.run_stream(task=message):
                # Check for streaming chunk events
                if hasattr(item, "content"):
                    content = item.content
                    if content:
                        yield StreamEvent(
                            type=StreamEventType.TOKEN,
                            data={"content": content},
                            run_id=ctx.run_id,
                        )
                        self._message_buffer.append(
                            {
                                "sender": getattr(item, "source", "agent"),
                                "content": content,
                                "role": "assistant",
                            }
                        )

                # Check for agent messages
                if hasattr(item, "source") and hasattr(item, "message"):
                    turn_count += 1
                    yield StreamEvent(
                        type=StreamEventType.MESSAGE,
                        data={
                            "content": str(item.message),
                            "agent": item.source,
                        },
                        run_id=ctx.run_id,
                    )

                    if self.checkpoint_turns:
                        await self.on_checkpoint(
                            {
                                "state": {"turn": turn_count, "agent": item.source},
                                "messages": self._message_buffer.copy(),
                            },
                            ctx,
                        )
                        yield StreamEvent(
                            type=StreamEventType.CHECKPOINT,
                            data={"turn": turn_count, "agent": item.source},
                            run_id=ctx.run_id,
                        )

            # Final checkpoint
            if self.checkpoint_turns:
                await self.on_checkpoint(
                    {
                        "state": {"completed": True, "turn": turn_count},
                        "messages": self._message_buffer.copy(),
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
                data={"result": {"messages": self._message_buffer}},
                run_id=ctx.run_id,
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=ctx.run_id,
            )

    def _register_message_callback(self) -> None:
        """Register callback to capture messages."""

        def on_message(
            sender: Any,
            message: str | dict[str, Any],
            recipient: Any,
            _silent: bool,
        ) -> None:
            self._message_buffer.append(
                {
                    "sender": getattr(sender, "name", str(sender)),
                    "recipient": getattr(recipient, "name", str(recipient)),
                    "content": message if isinstance(message, str) else message.get("content", ""),
                    "role": message.get("role", "assistant")
                    if isinstance(message, dict)
                    else "assistant",
                }
            )

        # Register with both agents
        if hasattr(self.initiator, "register_reply"):
            self.initiator.register_reply(
                [type(self.recipient)],
                lambda *_args, **_kwargs: (False, None),
                position=0,
            )

    def _register_streaming_callback(self, _ctx: AdapterContext) -> None:
        """Register streaming callback."""
        self._register_message_callback()

    def _message_to_event(self, msg: dict[str, Any], ctx: AdapterContext) -> StreamEvent:
        """Convert message to stream event."""
        content = msg.get("content", "")
        sender = msg.get("sender", "agent")

        # Check for tool calls
        if isinstance(content, dict) and "function_call" in content:
            return StreamEvent(
                type=StreamEventType.TOOL_CALL,
                data={
                    "name": content["function_call"].get("name", "unknown"),
                    "input": content["function_call"].get("arguments", {}),
                    "agent": sender,
                },
                run_id=ctx.run_id,
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE,
            data={
                "content": content,
                "agent": sender,
                "role": msg.get("role", "assistant"),
            },
            run_id=ctx.run_id,
        )

    def _extract_message(self, input: Any) -> str:
        """Extract message string from input."""
        if isinstance(input, str):
            return input

        if hasattr(input, "model_dump"):
            data = input.model_dump()
        elif isinstance(input, dict):
            data = input
        else:
            return str(input)

        for key in ("message", "query", "prompt", "input", "text", "content"):
            if key in data:
                return str(data[key])

        return str(data)

    def _extract_chat_result(self, chat_result: Any) -> dict[str, Any]:
        """Extract result from chat completion."""
        if hasattr(chat_result, "chat_history"):
            history = chat_result.chat_history
            last_message = history[-1] if history else {}

            return {
                "response": last_message.get("content", ""),
                "history": history,
                "cost": getattr(chat_result, "cost", None),
                "summary": getattr(chat_result, "summary", None),
            }

        return {"response": str(chat_result)}

    def with_max_turns(self, max_turns: int) -> AutoGenAdapter:
        """Create a new adapter with different max turns.

        Args:
            max_turns: Maximum conversation turns

        Returns:
            A new AutoGenAdapter
        """
        return AutoGenAdapter(
            self.initiator,
            self.recipient,
            group_chat=self.group_chat,
            max_turns=max_turns,
            clear_history=self.clear_history,
            silent=self.silent,
            checkpoint_turns=self.checkpoint_turns,
        )

    def with_group_chat(self, group_chat: Any) -> AutoGenAdapter:
        """Create a new adapter with a group chat.

        Args:
            group_chat: The GroupChat instance

        Returns:
            A new AutoGenAdapter
        """
        return AutoGenAdapter(
            self.initiator,
            self.recipient,
            group_chat=group_chat,
            max_turns=self.max_turns,
            clear_history=self.clear_history,
            silent=self.silent,
            checkpoint_turns=self.checkpoint_turns,
        )

    def with_checkpoints(self, enabled: bool = True) -> AutoGenAdapter:
        """Create a new adapter with checkpoint configuration.

        Args:
            enabled: Whether to checkpoint after each turn

        Returns:
            A new AutoGenAdapter with the specified checkpoint settings
        """
        return AutoGenAdapter(
            self.initiator,
            self.recipient,
            group_chat=self.group_chat,
            max_turns=self.max_turns,
            clear_history=self.clear_history,
            silent=self.silent,
            checkpoint_turns=enabled,
        )
