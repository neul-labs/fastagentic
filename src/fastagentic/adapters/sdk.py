"""Community Adapter SDK for FastAgentic.

This module provides utilities for building custom adapters that integrate
seamlessly with FastAgentic's deployment features.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent, StreamEventType

T = TypeVar("T")


@dataclass
class AdapterMetadata:
    """Metadata for a community adapter.

    Used for registration, discovery, and documentation.
    """

    name: str
    version: str
    description: str
    author: str
    framework: str
    framework_version: str | None = None
    homepage: str | None = None
    repository: str | None = None
    license: str = "MIT"
    tags: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "tags": self.tags,
            "capabilities": self.capabilities,
        }


class CommunityAdapter(BaseAdapter):
    """Base class for community adapters.

    Extends BaseAdapter with additional features for community plugins:
    - Metadata registration
    - Capability declarations
    - Health checks
    - Configuration validation

    Example:
        from fastagentic.adapters.sdk import CommunityAdapter, AdapterMetadata

        class MyFrameworkAdapter(CommunityAdapter):
            metadata = AdapterMetadata(
                name="myframework-adapter",
                version="1.0.0",
                description="Adapter for MyFramework",
                author="Your Name",
                framework="myframework",
                capabilities=["streaming", "tools"],
            )

            def __init__(self, agent):
                self.agent = agent

            async def invoke(self, input, ctx):
                return await self.agent.run(input)

            async def stream(self, input, ctx):
                async for chunk in self.agent.stream(input):
                    yield StreamEvent(type=StreamEventType.TOKEN, data={"content": chunk})
    """

    metadata: AdapterMetadata

    @classmethod
    def get_metadata(cls) -> AdapterMetadata:
        """Get adapter metadata."""
        if not hasattr(cls, "metadata"):
            raise NotImplementedError(f"{cls.__name__} must define a 'metadata' class attribute")
        return cls.metadata

    def validate_config(self, _config: dict[str, Any]) -> list[str]:
        """Validate adapter configuration.

        Override this to add custom configuration validation.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    async def health_check(self) -> dict[str, Any]:
        """Check adapter health.

        Override this to add custom health checks.

        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "adapter": self.get_metadata().name,
            "version": self.get_metadata().version,
        }

    def get_capabilities(self) -> list[str]:
        """Get adapter capabilities.

        Returns:
            List of capability strings
        """
        return self.get_metadata().capabilities


class SimpleAdapter(CommunityAdapter):
    """Simplified adapter for common use cases.

    Wraps a callable that takes input and returns output, with optional
    streaming support via a generator function.

    Example:
        def my_agent(input: str) -> str:
            return f"Response to: {input}"

        adapter = SimpleAdapter(
            invoke_fn=my_agent,
            metadata=AdapterMetadata(
                name="simple-agent",
                version="1.0.0",
                description="A simple agent adapter",
                author="You",
                framework="custom",
            ),
        )
    """

    def __init__(
        self,
        invoke_fn: Callable[..., Any],
        *,
        stream_fn: Callable[..., AsyncIterator[str]] | None = None,
        metadata: AdapterMetadata | None = None,
    ) -> None:
        """Initialize the simple adapter.

        Args:
            invoke_fn: Function to invoke (sync or async)
            stream_fn: Optional async generator for streaming
            metadata: Optional adapter metadata
        """
        self.invoke_fn = invoke_fn
        self.stream_fn = stream_fn

        if metadata:
            self.metadata = metadata
        else:
            self.metadata = AdapterMetadata(
                name="simple-adapter",
                version="1.0.0",
                description="Simple function adapter",
                author="unknown",
                framework="custom",
            )

    async def invoke(self, input: Any, ctx: AdapterContext | Any) -> Any:
        """Invoke the wrapped function."""
        self._ensure_adapter_context(ctx)
        extracted_input = self._extract_input(input)

        import asyncio

        if asyncio.iscoroutinefunction(self.invoke_fn):
            return await self.invoke_fn(extracted_input)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.invoke_fn, extracted_input)

    async def stream(self, input: Any, ctx: AdapterContext | Any) -> AsyncIterator[StreamEvent]:
        """Stream from the wrapped function."""
        adapter_ctx = self._ensure_adapter_context(ctx)
        extracted_input = self._extract_input(input)

        if self.stream_fn is not None:
            try:
                full_response = ""
                async for chunk in self.stream_fn(extracted_input):
                    full_response += chunk
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"content": chunk},
                        run_id=adapter_ctx.run_id,
                    )

                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={"result": full_response},
                    run_id=adapter_ctx.run_id,
                )
            except Exception as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": str(e)},
                    run_id=adapter_ctx.run_id,
                )
        else:
            # Fallback to invoke
            try:
                result = await self.invoke(input, ctx)
                result_str = str(result)

                yield StreamEvent(
                    type=StreamEventType.TOKEN,
                    data={"content": result_str},
                    run_id=adapter_ctx.run_id,
                )

                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={"result": result_str},
                    run_id=adapter_ctx.run_id,
                )
            except Exception as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": str(e)},
                    run_id=adapter_ctx.run_id,
                )

    def _extract_input(self, input: Any) -> Any:
        """Extract input value."""
        if hasattr(input, "model_dump"):
            data = input.model_dump()
            for key in ("message", "input", "query", "prompt", "text"):
                if key in data:
                    return data[key]
            return data

        return input


class AdapterRegistry:
    """Registry for community adapters.

    Provides discovery and management of community adapters.

    Example:
        registry = AdapterRegistry()
        registry.register(MyAdapter)

        # Get all adapters
        adapters = registry.list_adapters()

        # Get adapter by name
        adapter_cls = registry.get("myframework-adapter")
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._adapters: dict[str, type[CommunityAdapter]] = {}

    def register(self, adapter_cls: type[CommunityAdapter]) -> None:
        """Register an adapter.

        Args:
            adapter_cls: The adapter class to register
        """
        metadata = adapter_cls.get_metadata()
        self._adapters[metadata.name] = adapter_cls

    def unregister(self, name: str) -> bool:
        """Unregister an adapter.

        Args:
            name: The adapter name

        Returns:
            True if adapter was unregistered
        """
        if name in self._adapters:
            del self._adapters[name]
            return True
        return False

    def get(self, name: str) -> type[CommunityAdapter] | None:
        """Get an adapter by name.

        Args:
            name: The adapter name

        Returns:
            The adapter class or None
        """
        return self._adapters.get(name)

    def list_adapters(self) -> list[AdapterMetadata]:
        """List all registered adapters.

        Returns:
            List of adapter metadata
        """
        return [cls.get_metadata() for cls in self._adapters.values()]

    def find_by_framework(self, framework: str) -> list[type[CommunityAdapter]]:
        """Find adapters by framework.

        Args:
            framework: The framework name

        Returns:
            List of matching adapter classes
        """
        return [cls for cls in self._adapters.values() if cls.get_metadata().framework == framework]

    def find_by_capability(self, capability: str) -> list[type[CommunityAdapter]]:
        """Find adapters by capability.

        Args:
            capability: The capability string

        Returns:
            List of matching adapter classes
        """
        return [
            cls for cls in self._adapters.values() if capability in cls.get_metadata().capabilities
        ]


# Global registry instance
_global_registry = AdapterRegistry()


def register_adapter(adapter_cls: type[CommunityAdapter]) -> type[CommunityAdapter]:
    """Decorator to register an adapter.

    Example:
        @register_adapter
        class MyAdapter(CommunityAdapter):
            metadata = AdapterMetadata(...)
    """
    _global_registry.register(adapter_cls)
    return adapter_cls


def get_registry() -> AdapterRegistry:
    """Get the global adapter registry."""
    return _global_registry


def reset_registry() -> None:
    """Reset the global adapter registry.

    This is primarily useful for testing to ensure a clean state
    between test cases.
    """
    global _global_registry
    _global_registry = AdapterRegistry()
