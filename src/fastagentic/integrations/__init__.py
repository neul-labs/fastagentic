"""First-class integrations for FastAgentic.

FastAgentic integrates best-of-breed tools for specialized concerns:
- Observability: Langfuse, Braintrust
- Security: Lakera, Prompt Armor
- AI Gateway: Portkey
- Memory: Mem0

These integrations are optional and lazy-loaded to avoid requiring
all dependencies.
"""

from fastagentic.integrations.base import Integration, IntegrationConfig

__all__ = [
    "Integration",
    "IntegrationConfig",
    # Lazy-loaded integrations
    "LangfuseIntegration",
    "PortkeyIntegration",
    "LakeraIntegration",
    "Mem0Integration",
]


def __getattr__(name: str) -> type:
    """Lazy-load optional integrations."""
    if name == "LangfuseIntegration":
        from fastagentic.integrations.langfuse import LangfuseIntegration

        return LangfuseIntegration
    elif name == "PortkeyIntegration":
        from fastagentic.integrations.portkey import PortkeyIntegration

        return PortkeyIntegration
    elif name == "LakeraIntegration":
        from fastagentic.integrations.lakera import LakeraIntegration

        return LakeraIntegration
    elif name == "Mem0Integration":
        from fastagentic.integrations.mem0 import Mem0Integration

        return Mem0Integration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
