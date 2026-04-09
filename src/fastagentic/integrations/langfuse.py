"""Langfuse observability integration for FastAgentic.

Langfuse provides LLM observability with tracing, metrics, and analytics.
https://langfuse.com

Example:
    from fastagentic import App
    from fastagentic.integrations import LangfuseIntegration

    app = App(
        title="My Agent",
        integrations=[
            LangfuseIntegration(
                public_key="pk-...",
                secret_key="sk-...",
            )
        ]
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastagentic.hooks.base import Hook, HookContext, HookResult
from fastagentic.integrations.base import Integration, IntegrationConfig

if TYPE_CHECKING:
    from fastagentic.app import App

try:
    from langfuse import Langfuse
    from langfuse.decorators import langfuse_context

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    langfuse_context = None


@dataclass
class LangfuseConfig(IntegrationConfig):
    """Configuration for Langfuse integration."""

    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    release: str | None = None
    debug: bool = False
    flush_at: int = 15
    flush_interval: float = 0.5


class LangfuseHook(Hook):
    """Hook for Langfuse observability.

    Automatically creates traces, spans, and generations for:
    - Agent requests/responses
    - LLM calls with token usage
    - Tool invocations
    - Errors and retries
    """

    def __init__(self, client: Any, config: LangfuseConfig) -> None:
        self.client = client
        self.config = config
        self._traces: dict[str, Any] = {}
        self._spans: dict[str, Any] = {}

    async def on_request(self, ctx: HookContext) -> HookResult:
        """Start a trace for the request."""
        if not self.client:
            return HookResult.proceed()

        trace = self.client.trace(
            name=ctx.endpoint,
            user_id=ctx.user.user_id if ctx.user else None,
            session_id=ctx.user.session_id if ctx.user else None,
            metadata={
                "run_id": ctx.run_id,
                "endpoint": ctx.endpoint,
                **ctx.metadata,
            },
            release=self.config.release,
        )
        self._traces[ctx.run_id] = trace
        ctx.metadata["langfuse_trace_id"] = trace.id

        return HookResult.proceed()

    async def on_response(self, ctx: HookContext) -> HookResult:
        """End the trace with response data."""
        if not self.client:
            return HookResult.proceed()

        trace = self._traces.pop(ctx.run_id, None)
        if trace:
            trace.update(
                output=ctx.response,
                metadata={
                    "status": "success" if not ctx.error else "error",
                    "tokens": ctx.usage.total_tokens if ctx.usage else None,
                },
            )

        return HookResult.proceed()

    async def on_llm_start(self, ctx: HookContext) -> HookResult:
        """Start a generation span for LLM call."""
        if not self.client:
            return HookResult.proceed()

        trace = self._traces.get(ctx.run_id)
        if not trace:
            return HookResult.proceed()

        span_key = f"{ctx.run_id}:llm:{time.time()}"
        generation = trace.generation(
            name=f"llm-{ctx.model or 'unknown'}",
            model=ctx.model,
            input=ctx.messages,
            metadata={"model": ctx.model},
        )
        self._spans[span_key] = generation
        ctx.metadata["langfuse_generation_key"] = span_key

        return HookResult.proceed()

    async def on_llm_end(self, ctx: HookContext) -> HookResult:
        """End the generation span with response and usage."""
        if not self.client:
            return HookResult.proceed()

        span_key = ctx.metadata.get("langfuse_generation_key")
        if not span_key:
            return HookResult.proceed()

        generation = self._spans.pop(span_key, None)
        if generation:
            usage = None
            if ctx.usage:
                usage = {
                    "input": ctx.usage.input_tokens,
                    "output": ctx.usage.output_tokens,
                    "total": ctx.usage.total_tokens,
                }

            generation.end(
                output=ctx.response,
                usage=usage,
                metadata={"model": ctx.model},
            )

        return HookResult.proceed()

    async def on_tool_call(self, ctx: HookContext) -> HookResult:
        """Start a span for tool invocation."""
        if not self.client:
            return HookResult.proceed()

        trace = self._traces.get(ctx.run_id)
        if not trace:
            return HookResult.proceed()

        span_key = f"{ctx.run_id}:tool:{ctx.tool_name}:{time.time()}"
        span = trace.span(
            name=f"tool-{ctx.tool_name}",
            input=ctx.tool_input,
            metadata={"tool_name": ctx.tool_name},
        )
        self._spans[span_key] = span
        ctx.metadata["langfuse_tool_span_key"] = span_key

        return HookResult.proceed()

    async def on_tool_result(self, ctx: HookContext) -> HookResult:
        """End the tool span with result."""
        if not self.client:
            return HookResult.proceed()

        span_key = ctx.metadata.get("langfuse_tool_span_key")
        if not span_key:
            return HookResult.proceed()

        span = self._spans.pop(span_key, None)
        if span:
            span.end(
                output=ctx.tool_output,
                metadata={
                    "tool_name": ctx.tool_name,
                    "success": ctx.error is None,
                },
            )

        return HookResult.proceed()

    async def on_error(self, ctx: HookContext) -> HookResult:
        """Log error to trace."""
        if not self.client:
            return HookResult.proceed()

        trace = self._traces.get(ctx.run_id)
        if trace:
            trace.update(
                metadata={
                    "error": str(ctx.error),
                    "error_type": type(ctx.error).__name__ if ctx.error else None,
                },
                level="ERROR",
            )

        return HookResult.proceed()

    async def on_node_enter(self, ctx: HookContext) -> HookResult:
        """Start a span for graph node."""
        if not self.client:
            return HookResult.proceed()

        trace = self._traces.get(ctx.run_id)
        if not trace:
            return HookResult.proceed()

        span_key = f"{ctx.run_id}:node:{ctx.node_name}:{time.time()}"
        span = trace.span(
            name=f"node-{ctx.node_name}",
            input=ctx.node_input,
            metadata={"node_name": ctx.node_name},
        )
        self._spans[span_key] = span
        ctx.metadata["langfuse_node_span_key"] = span_key

        return HookResult.proceed()

    async def on_node_exit(self, ctx: HookContext) -> HookResult:
        """End the node span."""
        if not self.client:
            return HookResult.proceed()

        span_key = ctx.metadata.get("langfuse_node_span_key")
        if not span_key:
            return HookResult.proceed()

        span = self._spans.pop(span_key, None)
        if span:
            span.end(output=ctx.node_output)

        return HookResult.proceed()


class LangfuseIntegration(Integration):
    """Langfuse observability integration.

    Provides automatic tracing for all agent operations including
    LLM calls, tool invocations, and graph node executions.

    Example:
        app = App(
            integrations=[
                LangfuseIntegration(
                    public_key="pk-...",
                    secret_key="sk-...",
                )
            ]
        )

    Environment variables:
        LANGFUSE_PUBLIC_KEY: Public API key
        LANGFUSE_SECRET_KEY: Secret API key
        LANGFUSE_HOST: API host (default: https://cloud.langfuse.com)
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str = "https://cloud.langfuse.com",
        release: str | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        config = LangfuseConfig(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release=release,
            debug=debug,
            extra=kwargs,
        )
        super().__init__(config)
        self._config = config
        self._client: Any = None
        self._hook: LangfuseHook | None = None

    @property
    def name(self) -> str:
        return "langfuse"

    def is_available(self) -> bool:
        return LANGFUSE_AVAILABLE

    def validate_config(self) -> list[str]:
        errors = super().validate_config()

        if not self.is_available():
            errors.append("langfuse package not installed. Run: pip install langfuse")
            return errors

        # Keys can come from env vars
        import os

        public_key = self._config.public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = self._config.secret_key or os.getenv("LANGFUSE_SECRET_KEY")

        if not public_key:
            errors.append("Langfuse public_key is required")
        if not secret_key:
            errors.append("Langfuse secret_key is required")

        return errors

    def get_hooks(self) -> list[Hook]:
        if not self._hook:
            self._hook = LangfuseHook(self._client, self._config)
        return [self._hook]

    def setup(self, _app: App) -> None:
        """Initialize the Langfuse client."""
        if not self.is_available():
            return

        import os

        public_key = self._config.public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = self._config.secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = self._config.host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=self._config.debug,
                flush_at=self._config.flush_at,
                flush_interval=self._config.flush_interval,
            )
            # Update hook with client
            if self._hook:
                self._hook.client = self._client

    async def on_startup(self) -> None:
        """Verify Langfuse connection."""
        await super().on_startup()
        # Client is already initialized in setup()

    async def on_shutdown(self) -> None:
        """Flush any pending events."""
        await super().on_shutdown()
        if self._client:
            self._client.flush()

    def teardown(self) -> None:
        """Shutdown the Langfuse client."""
        if self._client:
            self._client.shutdown()
            self._client = None

    def get_client(self) -> Any:
        """Get the underlying Langfuse client for advanced usage."""
        return self._client
