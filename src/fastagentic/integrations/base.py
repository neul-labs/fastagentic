"""Base integration classes for FastAgentic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastagentic.app import App
    from fastagentic.hooks.base import Hook


@dataclass
class IntegrationConfig:
    """Base configuration for integrations."""

    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Integration(ABC):
    """Base class for FastAgentic integrations.

    Integrations connect FastAgentic to external services for
    observability, security, memory, and other capabilities.

    Integrations can:
    - Provide hooks for lifecycle events
    - Wrap LLM clients for observability/routing
    - Add middleware for request/response processing
    - Provide memory/storage backends

    Example:
        class MyIntegration(Integration):
            def __init__(self, api_key: str):
                super().__init__(IntegrationConfig(api_key=api_key))

            def get_hooks(self) -> list[Hook]:
                return [MyObservabilityHook(self.config.api_key)]

            def setup(self, app: App) -> None:
                # Configure the app
                pass
    """

    def __init__(self, config: IntegrationConfig | None = None) -> None:
        self.config = config or IntegrationConfig()
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The integration name."""
        ...

    def get_hooks(self) -> list[Hook]:
        """Return hooks provided by this integration.

        Override this to add lifecycle hooks for observability,
        security checks, etc.

        Returns:
            List of Hook instances
        """
        return []

    def setup(self, app: App) -> None:
        """Set up the integration with the app.

        Called when the integration is added to the app.
        Override to configure routes, middleware, etc.

        Args:
            app: The FastAgentic app instance
        """
        pass

    def teardown(self) -> None:
        """Clean up resources.

        Called when the app shuts down.
        Override to close connections, flush buffers, etc.
        """
        pass

    async def on_startup(self) -> None:
        """Async startup hook.

        Called when the app starts up.
        Override for async initialization.
        """
        self._initialized = True

    async def on_shutdown(self) -> None:
        """Async shutdown hook.

        Called when the app shuts down.
        Override for async cleanup.
        """
        self._initialized = False

    def is_available(self) -> bool:
        """Check if the integration's dependencies are available.

        Returns:
            True if the integration can be used
        """
        return True

    def validate_config(self) -> list[str]:
        """Validate the integration configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.config.enabled:
            return errors
        return errors
