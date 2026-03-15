"""Template ecosystem for FastAgentic.

Provides template management, marketplace integration, versioning,
and template composition features.
"""

from fastagentic.templates.base import (
    Template,
    TemplateMetadata,
    TemplateFile,
    TemplateVariable,
    TemplateVersion,
)
from fastagentic.templates.registry import (
    TemplateRegistry,
    LocalRegistry,
    RemoteRegistry,
    EnterpriseRegistry,
)
from fastagentic.templates.marketplace import (
    Marketplace,
    MarketplaceConfig,
    TemplateRating,
    TemplateReview,
)
from fastagentic.templates.composer import (
    TemplateComposer,
    CompositionConfig,
    ComposedTemplate,
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
