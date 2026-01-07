"""LlamaIndex adapter for FastAgentic.

This adapter wraps LlamaIndex agents, query engines, and chat engines
to expose them via FastAgentic endpoints with streaming support.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

if TYPE_CHECKING:
    pass  # LlamaIndex types would be imported here


class LlamaIndexAdapter(BaseAdapter):
    """Adapter for LlamaIndex.

    Wraps LlamaIndex agents, query engines, and chat engines to work with
    FastAgentic's endpoint system, providing streaming and RAG support.

    Example:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.agent import ReActAgent
        from fastagentic.adapters.llamaindex import LlamaIndexAdapter

        # Query engine example
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        adapter = LlamaIndexAdapter(query_engine=query_engine)

        # Agent example
        agent = ReActAgent.from_tools(tools, llm=llm)
        adapter = LlamaIndexAdapter(agent=agent)

        @agent_endpoint(path="/query", runnable=adapter, stream=True)
        async def query(input: QueryInput) -> QueryOutput:
            ...
    """

    def __init__(
        self,
        agent: Any | None = None,
        query_engine: Any | None = None,
        chat_engine: Any | None = None,
        *,
        streaming: bool = True,
        similarity_top_k: int | None = None,
    ) -> None:
        """Initialize the LlamaIndex adapter.

        Args:
            agent: A LlamaIndex agent (ReActAgent, OpenAIAgent, etc.)
            query_engine: A LlamaIndex query engine
            chat_engine: A LlamaIndex chat engine
            streaming: Whether to use streaming by default
            similarity_top_k: Number of similar documents to retrieve
        """
        if not any([agent, query_engine, chat_engine]):
            raise ValueError("Must provide at least one of: agent, query_engine, chat_engine")

        self.agent = agent
        self.query_engine = query_engine
        self.chat_engine = chat_engine
        self.streaming = streaming
        self.similarity_top_k = similarity_top_k

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Run LlamaIndex and return the result.

        Args:
            input: The input query or message
            ctx: The adapter context

        Returns:
            The query or agent response
        """
        adapter_ctx = self._ensure_adapter_context(ctx)
        query = self._extract_query(input)

        try:
            if self.agent is not None:
                result = await self._invoke_agent(query, adapter_ctx)
            elif self.chat_engine is not None:
                result = await self._invoke_chat_engine(query, adapter_ctx)
            else:
                result = await self._invoke_query_engine(query, adapter_ctx)

            return self._format_response(result)

        except Exception as e:
            raise RuntimeError(f"LlamaIndex invocation failed: {e}") from e

    async def _invoke_agent(self, query: str, ctx: AdapterContext) -> Any:
        """Invoke LlamaIndex agent."""
        assert self.agent is not None
        # Get chat history from context
        chat_history = ctx.state.get("chat_history", [])

        # Run agent
        if hasattr(self.agent, "achat"):
            response = await self.agent.achat(query, chat_history=chat_history)
        else:
            response = self.agent.chat(query, chat_history=chat_history)

        # Update chat history
        ctx.state["chat_history"] = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(response)},
        ]

        return response

    async def _invoke_chat_engine(self, query: str, ctx: AdapterContext) -> Any:
        """Invoke LlamaIndex chat engine."""
        assert self.chat_engine is not None
        chat_history = ctx.state.get("chat_history", [])

        if hasattr(self.chat_engine, "achat"):
            response = await self.chat_engine.achat(query, chat_history=chat_history)
        else:
            response = self.chat_engine.chat(query, chat_history=chat_history)

        ctx.state["chat_history"] = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": str(response)},
        ]

        return response

    async def _invoke_query_engine(self, query: str, _ctx: AdapterContext) -> Any:
        """Invoke LlamaIndex query engine."""
        assert self.query_engine is not None
        if hasattr(self.query_engine, "aquery"):
            response = await self.query_engine.aquery(query)
        else:
            response = self.query_engine.query(query)

        return response

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream events from LlamaIndex.

        Args:
            input: The input query or message
            ctx: The adapter context

        Yields:
            StreamEvent objects for tokens, sources, etc.
        """
        adapter_ctx = self._ensure_adapter_context(ctx)
        query = self._extract_query(input)

        try:
            if self.agent is not None:
                async for event in self._stream_agent(query, adapter_ctx):
                    yield event
            elif self.chat_engine is not None:
                async for event in self._stream_chat_engine(query, adapter_ctx):
                    yield event
            else:
                async for event in self._stream_query_engine(query, adapter_ctx):
                    yield event

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                run_id=adapter_ctx.run_id,
            )

    async def _stream_agent(self, query: str, ctx: AdapterContext) -> AsyncIterator[StreamEvent]:
        """Stream from LlamaIndex agent."""
        assert self.agent is not None
        chat_history = ctx.state.get("chat_history", [])
        full_response = ""

        if hasattr(self.agent, "astream_chat"):
            response_gen = await self.agent.astream_chat(query, chat_history=chat_history)

            async for token in response_gen.async_response_gen():
                full_response += token
                yield StreamEvent(
                    type=StreamEventType.TOKEN,
                    data={"content": token},
                    run_id=ctx.run_id,
                )

            # Yield source nodes if available
            if hasattr(response_gen, "source_nodes"):
                for node in response_gen.source_nodes:
                    yield StreamEvent(
                        type=StreamEventType.SOURCE,
                        data={
                            "text": node.text[:500],
                            "score": getattr(node, "score", None),
                            "metadata": getattr(node, "metadata", {}),
                        },
                        run_id=ctx.run_id,
                    )

        else:
            # Fallback to non-streaming
            response = await self._invoke_agent(query, ctx)
            full_response = str(response)

            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": full_response},
                run_id=ctx.run_id,
            )

        ctx.state["chat_history"] = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": full_response},
        ]

        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": full_response},
            run_id=ctx.run_id,
        )

    async def _stream_chat_engine(
        self, query: str, ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream from LlamaIndex chat engine."""
        assert self.chat_engine is not None
        chat_history = ctx.state.get("chat_history", [])
        full_response = ""

        if hasattr(self.chat_engine, "astream_chat"):
            response_gen = await self.chat_engine.astream_chat(query, chat_history=chat_history)

            async for token in response_gen.async_response_gen():
                full_response += token
                yield StreamEvent(
                    type=StreamEventType.TOKEN,
                    data={"content": token},
                    run_id=ctx.run_id,
                )

            if hasattr(response_gen, "source_nodes"):
                for node in response_gen.source_nodes:
                    yield StreamEvent(
                        type=StreamEventType.SOURCE,
                        data={
                            "text": node.text[:500],
                            "score": getattr(node, "score", None),
                        },
                        run_id=ctx.run_id,
                    )

        else:
            response = await self._invoke_chat_engine(query, ctx)
            full_response = str(response)

            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": full_response},
                run_id=ctx.run_id,
            )

        ctx.state["chat_history"] = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": full_response},
        ]

        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": full_response},
            run_id=ctx.run_id,
        )

    async def _stream_query_engine(
        self, query: str, ctx: AdapterContext
    ) -> AsyncIterator[StreamEvent]:
        """Stream from LlamaIndex query engine."""
        assert self.query_engine is not None
        full_response = ""

        if hasattr(self.query_engine, "aquery") and self.streaming:
            # Try streaming query
            try:
                response = await self.query_engine.aquery(query)

                if hasattr(response, "response_gen"):
                    async for token in response.response_gen:
                        full_response += token
                        yield StreamEvent(
                            type=StreamEventType.TOKEN,
                            data={"content": token},
                            run_id=ctx.run_id,
                        )
                else:
                    full_response = str(response)
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": full_response},
                        run_id=ctx.run_id,
                    )

                # Yield source nodes
                if hasattr(response, "source_nodes"):
                    for node in response.source_nodes:
                        yield StreamEvent(
                            type=StreamEventType.SOURCE,
                            data={
                                "text": getattr(node, "text", "")[:500],
                                "score": getattr(node, "score", None),
                                "metadata": getattr(node.node, "metadata", {}),
                            },
                            run_id=ctx.run_id,
                        )

            except Exception:
                # Fallback to non-streaming
                response = await self._invoke_query_engine(query, ctx)
                full_response = str(response)

                yield StreamEvent(
                    type=StreamEventType.TOKEN,
                    data={"content": full_response},
                    run_id=ctx.run_id,
                )
        else:
            response = await self._invoke_query_engine(query, ctx)
            full_response = str(response)

            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": full_response},
                run_id=ctx.run_id,
            )

        yield StreamEvent(
            type=StreamEventType.DONE,
            data={"result": full_response},
            run_id=ctx.run_id,
        )

    def _extract_query(self, input: Any) -> str:
        """Extract query string from input."""
        if isinstance(input, str):
            return input

        if hasattr(input, "model_dump"):
            data = input.model_dump()
        elif isinstance(input, dict):
            data = input
        else:
            return str(input)

        for key in ("query", "message", "question", "prompt", "input", "text"):
            if key in data:
                return str(data[key])

        return str(data)

    def _format_response(self, response: Any) -> dict[str, Any]:
        """Format LlamaIndex response."""
        result: dict[str, Any] = {}

        # Extract response text
        if hasattr(response, "response"):
            result["response"] = response.response
        elif hasattr(response, "content"):
            result["response"] = response.content
        else:
            result["response"] = str(response)

        # Extract source nodes
        if hasattr(response, "source_nodes"):
            result["sources"] = [
                {
                    "text": getattr(node, "text", "")[:500],
                    "score": getattr(node, "score", None),
                    "metadata": getattr(node.node, "metadata", {}) if hasattr(node, "node") else {},
                }
                for node in response.source_nodes
            ]

        # Extract metadata
        if hasattr(response, "metadata"):
            result["metadata"] = response.metadata

        return result

    def with_query_engine(self, query_engine: Any) -> LlamaIndexAdapter:
        """Create a new adapter with a different query engine.

        Args:
            query_engine: The new query engine

        Returns:
            A new LlamaIndexAdapter
        """
        return LlamaIndexAdapter(
            agent=self.agent,
            query_engine=query_engine,
            chat_engine=self.chat_engine,
            streaming=self.streaming,
            similarity_top_k=self.similarity_top_k,
        )

    def with_agent(self, agent: Any) -> LlamaIndexAdapter:
        """Create a new adapter with a different agent.

        Args:
            agent: The new agent

        Returns:
            A new LlamaIndexAdapter
        """
        return LlamaIndexAdapter(
            agent=agent,
            query_engine=self.query_engine,
            chat_engine=self.chat_engine,
            streaming=self.streaming,
            similarity_top_k=self.similarity_top_k,
        )

    def with_chat_engine(self, chat_engine: Any) -> LlamaIndexAdapter:
        """Create a new adapter with a different chat engine.

        Args:
            chat_engine: The new chat engine

        Returns:
            A new LlamaIndexAdapter
        """
        return LlamaIndexAdapter(
            agent=self.agent,
            query_engine=self.query_engine,
            chat_engine=chat_engine,
            streaming=self.streaming,
            similarity_top_k=self.similarity_top_k,
        )
