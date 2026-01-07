"""Portkey AI Gateway integration for FastAgentic.

Portkey provides unified API access to 200+ LLMs with:
- Automatic fallbacks and load balancing
- Semantic caching
- Observability and analytics
- Guardrails and budget controls

https://portkey.ai

Example:
    from fastagentic import App
    from fastagentic.integrations import PortkeyIntegration

    app = App(
        title="My Agent",
        integrations=[
            PortkeyIntegration(
                api_key="pk-...",
                virtual_key="vk-...",
                config={
                    "retry": {"attempts": 3},
                    "cache": {"mode": "semantic"},
                }
            )
        ]
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastagentic.hooks.base import Hook, HookContext, HookResult
from fastagentic.integrations.base import Integration, IntegrationConfig

if TYPE_CHECKING:
    from fastagentic.app import App

try:
    from portkey_ai import Portkey, createHeaders

    PORTKEY_AVAILABLE = True
except ImportError:
    PORTKEY_AVAILABLE = False
    Portkey = None
    createHeaders = None


@dataclass
class PortkeyConfig(IntegrationConfig):
    """Configuration for Portkey integration."""

    virtual_key: str | None = None
    config_id: str | None = None
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id_prefix: str = "fastagentic"
    cache_mode: str | None = None  # "simple" or "semantic"
    retry_attempts: int = 3
    timeout: int = 60000  # ms


class PortkeyHook(Hook):
    """Hook for Portkey AI gateway.

    Adds Portkey headers to LLM requests for routing,
    caching, and observability.
    """

    def __init__(self, config: PortkeyConfig) -> None:
        self.config = config

    async def on_llm_start(self, ctx: HookContext) -> HookResult:
        """Add Portkey headers to LLM request."""
        if not createHeaders:
            return HookResult.proceed()

        headers = createHeaders(
            api_key=self.config.api_key,
            virtual_key=self.config.virtual_key,
            config=self.config.config_id,
            trace_id=f"{self.config.trace_id_prefix}-{ctx.run_id}",
            metadata={
                "run_id": ctx.run_id,
                "endpoint": ctx.endpoint,
                "user_id": ctx.user.user_id if ctx.user else None,
                **self.config.metadata,
            },
        )

        # Store headers in context for the adapter to use
        ctx.metadata["portkey_headers"] = headers

        return HookResult.proceed()


class PortkeyIntegration(Integration):
    """Portkey AI Gateway integration.

    Provides unified access to 200+ LLMs with automatic fallbacks,
    load balancing, caching, and observability.

    Features:
    - **Fallbacks**: Automatic failover between providers
    - **Load Balancing**: Distribute requests across models
    - **Caching**: Semantic or simple caching
    - **Guardrails**: Input/output validation
    - **Budget Controls**: Spend limits and alerts
    - **Analytics**: Usage tracking and insights

    Example:
        # Basic usage with API key
        app = App(
            integrations=[
                PortkeyIntegration(api_key="pk-...")
            ]
        )

        # With config for fallbacks
        app = App(
            integrations=[
                PortkeyIntegration(
                    api_key="pk-...",
                    config={
                        "strategy": {"mode": "fallback"},
                        "targets": [
                            {"virtual_key": "openai-key"},
                            {"virtual_key": "anthropic-key"},
                        ],
                    }
                )
            ]
        )

    Environment variables:
        PORTKEY_API_KEY: Portkey API key
        PORTKEY_VIRTUAL_KEY: Virtual key for provider
    """

    def __init__(
        self,
        api_key: str | None = None,
        virtual_key: str | None = None,
        config_id: str | None = None,
        config: dict[str, Any] | None = None,
        cache_mode: str | None = None,
        retry_attempts: int = 3,
        timeout: int = 60000,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pk_config = PortkeyConfig(
            api_key=api_key,
            virtual_key=virtual_key,
            config_id=config_id,
            config=config,
            cache_mode=cache_mode,
            retry_attempts=retry_attempts,
            timeout=timeout,
            metadata=metadata or {},
            extra=kwargs,
        )
        super().__init__(pk_config)
        self._config = pk_config
        self._client: Any = None
        self._hook: PortkeyHook | None = None

    @property
    def name(self) -> str:
        return "portkey"

    def is_available(self) -> bool:
        return PORTKEY_AVAILABLE

    def validate_config(self) -> list[str]:
        errors = super().validate_config()

        if not self.is_available():
            errors.append("portkey-ai package not installed. Run: pip install portkey-ai")
            return errors

        import os

        api_key = self._config.api_key or os.getenv("PORTKEY_API_KEY")
        if not api_key:
            errors.append("Portkey api_key is required")

        return errors

    def get_hooks(self) -> list[Hook]:
        if not self._hook:
            self._hook = PortkeyHook(self._config)
        return [self._hook]

    def setup(self, _app: App) -> None:
        """Initialize the Portkey client."""
        if not self.is_available():
            return

        import os

        api_key = self._config.api_key or os.getenv("PORTKEY_API_KEY")
        virtual_key = self._config.virtual_key or os.getenv("PORTKEY_VIRTUAL_KEY")

        if api_key:
            client_kwargs: dict[str, Any] = {"api_key": api_key}

            if virtual_key:
                client_kwargs["virtual_key"] = virtual_key

            if self._config.config:
                client_kwargs["config"] = self._config.config

            self._client = Portkey(**client_kwargs)

    def get_client(self) -> Any:
        """Get the Portkey client for direct LLM calls.

        Returns a Portkey client that can be used like OpenAI:

            client = portkey.get_client()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
        """
        return self._client

    def create_openai_client(self, **kwargs: Any) -> Any:
        """Create an OpenAI-compatible client routed through Portkey.

        Useful for integrating with frameworks that expect OpenAI client.

        Example:
            openai_client = portkey.create_openai_client()
            # Use with PydanticAI, LangChain, etc.
        """
        if not self.is_available():
            raise RuntimeError("portkey-ai not installed")

        import os

        api_key = self._config.api_key or os.getenv("PORTKEY_API_KEY")
        virtual_key = self._config.virtual_key or os.getenv("PORTKEY_VIRTUAL_KEY")

        return Portkey(
            api_key=api_key,
            virtual_key=virtual_key,
            **kwargs,
        )

    def get_headers(self, trace_id: str | None = None, **metadata: Any) -> dict[str, str]:
        """Get Portkey headers for manual integration.

        Useful when integrating with existing LLM clients:

            headers = portkey.get_headers(trace_id="my-trace")
            # Add to your OpenAI client extra_headers
        """
        if not createHeaders:
            return {}

        return createHeaders(
            api_key=self._config.api_key,
            virtual_key=self._config.virtual_key,
            config=self._config.config_id,
            trace_id=trace_id,
            metadata={**self._config.metadata, **metadata},
        )

    async def on_shutdown(self) -> None:
        """Cleanup Portkey client."""
        await super().on_shutdown()
        self._client = None
