"""Advanced prompt management for FastAgentic.

Provides prompt templates, versioning, A/B testing,
and a centralized prompt registry.
"""

from fastagentic.prompts.template import (
    PromptTemplate,
    PromptVariable,
    render_template,
)
from fastagentic.prompts.registry import (
    PromptRegistry,
    PromptVersion,
    PromptMetadata,
)
from fastagentic.prompts.testing import (
    PromptVariant,
    ABTest,
    ABTestResult,
)

__all__ = [
    # Templates
    "PromptTemplate",
    "PromptVariable",
    "render_template",
    # Registry
    "PromptRegistry",
    "PromptVersion",
    "PromptMetadata",
    # A/B Testing
    "PromptVariant",
    "ABTest",
    "ABTestResult",
]
