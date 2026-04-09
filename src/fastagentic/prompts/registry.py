"""Prompt registry and versioning for FastAgentic."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from fastagentic.prompts.template import PromptTemplate


@dataclass
class PromptMetadata:
    """Metadata for a prompt.

    Attributes:
        author: Who created/modified the prompt
        created_at: Creation timestamp
        updated_at: Last update timestamp
        usage_count: How many times the prompt has been used
        avg_tokens: Average token count when rendered
        tags: Categorization tags
        notes: Additional notes
    """

    author: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0
    avg_tokens: float = 0.0
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def created_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.created_at)

    @property
    def updated_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.updated_at)


@dataclass
class PromptVersion:
    """A specific version of a prompt.

    Attributes:
        version: Semantic version string (e.g., "1.0.0")
        template: The prompt template
        metadata: Version metadata
        content_hash: Hash of the template content
        is_active: Whether this version is currently active
        changelog: Description of changes in this version
    """

    version: str
    template: PromptTemplate
    metadata: PromptMetadata = field(default_factory=PromptMetadata)
    content_hash: str = ""
    is_active: bool = False
    changelog: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for the template."""
        content = self.template.content + str(self.template.variables)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class PromptStore(Protocol):
    """Protocol for prompt storage backends."""

    async def save(self, name: str, version: PromptVersion) -> None:
        """Save a prompt version."""
        ...

    async def load(self, name: str, version: str | None = None) -> PromptVersion | None:
        """Load a prompt version (latest if version is None)."""
        ...

    async def list_versions(self, name: str) -> list[str]:
        """List all versions of a prompt."""
        ...

    async def list_prompts(self) -> list[str]:
        """List all prompt names."""
        ...

    async def delete(self, name: str, version: str | None = None) -> None:
        """Delete a prompt version or all versions."""
        ...


class InMemoryPromptStore:
    """In-memory prompt store for development/testing."""

    def __init__(self) -> None:
        self._prompts: dict[str, dict[str, PromptVersion]] = {}

    async def save(self, name: str, version: PromptVersion) -> None:
        if name not in self._prompts:
            self._prompts[name] = {}
        self._prompts[name][version.version] = version

    async def load(self, name: str, version: str | None = None) -> PromptVersion | None:
        if name not in self._prompts:
            return None

        versions = self._prompts[name]
        if not versions:
            return None

        if version:
            return versions.get(version)

        # Return active version or latest
        for v in sorted(versions.keys(), reverse=True):
            if versions[v].is_active:
                return versions[v]

        # Return latest by version string
        latest = max(versions.keys())
        return versions[latest]

    async def list_versions(self, name: str) -> list[str]:
        if name not in self._prompts:
            return []
        return sorted(self._prompts[name].keys())

    async def list_prompts(self) -> list[str]:
        return list(self._prompts.keys())

    async def delete(self, name: str, version: str | None = None) -> None:
        if name not in self._prompts:
            return

        if version:
            self._prompts[name].pop(version, None)
        else:
            del self._prompts[name]


class PromptRegistry:
    """Central registry for managing prompts.

    Example:
        registry = PromptRegistry()

        # Register a prompt
        template = PromptTemplate(
            name="greeting",
            content="Hello, {{name}}! Welcome to {{company}}.",
            variables=[
                PromptVariable(name="name"),
                PromptVariable(name="company"),
            ]
        )
        await registry.register(template, version="1.0.0")

        # Get and use a prompt
        prompt = await registry.get("greeting")
        rendered = prompt.render(name="Alice", company="Acme")

        # Create a new version
        template_v2 = PromptTemplate(
            name="greeting",
            content="Hi {{name}}! Welcome to {{company}}. How can I help?",
            variables=template.variables,
        )
        await registry.register(template_v2, version="1.1.0", changelog="Added question")

        # Activate a version
        await registry.activate("greeting", "1.1.0")
    """

    def __init__(
        self,
        store: PromptStore | None = None,
    ) -> None:
        """Initialize prompt registry.

        Args:
            store: Storage backend for prompts
        """
        self._store = store or InMemoryPromptStore()
        self._cache: dict[str, PromptVersion] = {}

    async def register(
        self,
        template: PromptTemplate,
        version: str = "1.0.0",
        *,
        author: str | None = None,
        changelog: str = "",
        activate: bool = True,
        tags: list[str] | None = None,
    ) -> PromptVersion:
        """Register a new prompt or version.

        Args:
            template: The prompt template
            version: Version string (semver recommended)
            author: Who created this version
            changelog: Description of changes
            activate: Whether to make this the active version
            tags: Categorization tags

        Returns:
            The created PromptVersion
        """
        metadata = PromptMetadata(
            author=author,
            tags=tags or template.tags,
        )

        prompt_version = PromptVersion(
            version=version,
            template=template,
            metadata=metadata,
            is_active=activate,
            changelog=changelog,
        )

        # If activating, deactivate other versions
        if activate:
            existing_versions = await self._store.list_versions(template.name)
            for v in existing_versions:
                existing = await self._store.load(template.name, v)
                if existing and existing.is_active:
                    existing.is_active = False
                    await self._store.save(template.name, existing)

        await self._store.save(template.name, prompt_version)

        # Update cache
        if activate:
            self._cache[template.name] = prompt_version

        return prompt_version

    async def get(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptTemplate | None:
        """Get a prompt template.

        Args:
            name: Prompt name
            version: Specific version (latest/active if None)

        Returns:
            The prompt template or None
        """
        prompt_version = await self.get_version(name, version)
        return prompt_version.template if prompt_version else None

    async def get_version(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptVersion | None:
        """Get a prompt version with metadata.

        Args:
            name: Prompt name
            version: Specific version (latest/active if None)

        Returns:
            The prompt version or None
        """
        # Check cache for active version
        if version is None and name in self._cache:
            return self._cache[name]

        prompt_version = await self._store.load(name, version)

        # Update cache if this is the active version
        if prompt_version and prompt_version.is_active:
            self._cache[name] = prompt_version

        return prompt_version

    async def render(
        self,
        name: str,
        version: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Render a prompt with variables.

        Args:
            name: Prompt name
            version: Specific version (latest/active if None)
            **kwargs: Template variables

        Returns:
            Rendered prompt string or None if not found
        """
        template = await self.get(name, version)
        if template is None:
            return None

        # Track usage
        prompt_version = await self.get_version(name, version)
        if prompt_version:
            prompt_version.metadata.usage_count += 1

        return template.render(**kwargs)

    async def activate(self, name: str, version: str) -> bool:
        """Activate a specific version.

        Args:
            name: Prompt name
            version: Version to activate

        Returns:
            True if successful
        """
        target = await self._store.load(name, version)
        if target is None:
            return False

        # Deactivate all other versions
        existing_versions = await self._store.list_versions(name)
        for v in existing_versions:
            existing = await self._store.load(name, v)
            if existing:
                existing.is_active = v == version
                await self._store.save(name, existing)

        # Update cache
        self._cache[name] = target
        target.is_active = True

        return True

    async def list_prompts(self) -> list[str]:
        """List all registered prompt names."""
        return await self._store.list_prompts()

    async def list_versions(self, name: str) -> list[str]:
        """List all versions of a prompt."""
        return await self._store.list_versions(name)

    async def delete(
        self,
        name: str,
        version: str | None = None,
    ) -> None:
        """Delete a prompt or specific version.

        Args:
            name: Prompt name
            version: Version to delete (all if None)
        """
        await self._store.delete(name, version)

        # Clear cache
        if name in self._cache and (version is None or self._cache[name].version == version):
            del self._cache[name]

    async def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
    ) -> list[PromptVersion]:
        """Search for prompts.

        Args:
            query: Search in name/description
            tags: Filter by tags

        Returns:
            List of matching prompt versions
        """
        results: list[PromptVersion] = []

        for name in await self._store.list_prompts():
            prompt_version = await self._store.load(name)
            if prompt_version is None:
                continue

            # Check query match
            if query:
                query_lower = query.lower()
                name_match = query_lower in name.lower()
                desc_match = query_lower in prompt_version.template.description.lower()
                if not (name_match or desc_match):
                    continue

            # Check tag match
            if tags:
                prompt_tags = set(prompt_version.metadata.tags)
                if not prompt_tags.intersection(tags):
                    continue

            results.append(prompt_version)

        return results
