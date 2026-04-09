"""Template ecosystem for FastAgentic.

Provides template management, marketplace integration, versioning,
and template composition features.
"""

from fastagentic.templates.base import (
    Template,
    TemplateFile,
    TemplateMetadata,
    TemplateVariable,
    TemplateVersion,
)
from fastagentic.templates.composer import ComposedTemplate, CompositionConfig, TemplateComposer
from fastagentic.templates.marketplace import (
    Marketplace,
    MarketplaceConfig,
    TemplateRating,
    TemplateReview,
)
from fastagentic.templates.registry import (
    EnterpriseRegistry,
    LocalRegistry,
    RemoteRegistry,
    TemplateRegistry,
)

__all__ = [
    # Base
    "Template",
    "TemplateMetadata",
    "TemplateFile",
    "TemplateVariable",
    "TemplateVersion",
    # Registry
    "TemplateRegistry",
    "LocalRegistry",
    "RemoteRegistry",
    "EnterpriseRegistry",
    # Marketplace
    "Marketplace",
    "MarketplaceConfig",
    "TemplateRating",
    "TemplateReview",
    # Composer
    "TemplateComposer",
    "CompositionConfig",
    "ComposedTemplate",
]
