"""Template registry for FastAgentic.

Provides local, remote, and enterprise template registries.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastagentic.templates.base import Template, TemplateCategory, TemplateMetadata


class TemplateRegistry(ABC):
    """Abstract base class for template registries."""

    @abstractmethod
    def list_templates(
        self,
        category: TemplateCategory | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
    ) -> list[TemplateMetadata]:
        """List available templates.

        Args:
            category: Filter by category
            framework: Filter by framework
            tags: Filter by tags

        Returns:
            List of template metadata
        """
        ...

    @abstractmethod
    def get_template(self, name: str, version: str | None = None) -> Template | None:
        """Get a template by name.

        Args:
            name: Template name
            version: Optional specific version

        Returns:
            Template or None if not found
        """
        ...

    @abstractmethod
    def search(self, query: str) -> list[TemplateMetadata]:
        """Search for templates.

        Args:
            query: Search query

        Returns:
            List of matching template metadata
        """
        ...

    def template_exists(self, name: str) -> bool:
        """Check if a template exists.

        Args:
            name: Template name

        Returns:
            True if template exists
        """
        return self.get_template(name) is not None


class LocalRegistry(TemplateRegistry):
    """Local file-based template registry.

    Example:
        registry = LocalRegistry("/path/to/templates")
        templates = registry.list_templates()
    """

    def __init__(self, base_path: str | Path) -> None:
        """Initialize the local registry.

        Args:
            base_path: Base directory for templates
        """
        self.base_path = Path(base_path)
        self._cache: dict[str, Template] = {}

    def list_templates(
        self,
        category: TemplateCategory | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
    ) -> list[TemplateMetadata]:
        """List available templates."""
        templates: list[TemplateMetadata] = []

        if not self.base_path.exists():
            return templates

        for path in self.base_path.iterdir():
            if path.is_dir():
                template = self._load_template(path)
                if template:
                    meta = template.metadata

                    # Apply filters
                    if category and meta.category != category:
                        continue
                    if framework and meta.framework != framework:
                        continue
                    if tags and not all(t in meta.tags for t in tags):
                        continue

                    templates.append(meta)

        return templates

    def get_template(self, name: str, version: str | None = None) -> Template | None:
        """Get a template by name."""
        # Check cache
        cache_key = f"{name}:{version or 'latest'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        template_path = self.base_path / name
        if not template_path.exists():
            return None

        template = self._load_template(template_path)
        if template:
            self._cache[cache_key] = template

        return template

    def search(self, query: str) -> list[TemplateMetadata]:
        """Search for templates."""
        query_lower = query.lower()
        results: list[TemplateMetadata] = []

        for meta in self.list_templates():
            if (
                query_lower in meta.name.lower()
                or query_lower in meta.description.lower()
                or any(query_lower in tag.lower() for tag in meta.tags)
            ):
                results.append(meta)

        return results

    def add_template(self, template: Template) -> None:
        """Add a template to the registry.

        Args:
            template: Template to add
        """
        template_path = self.base_path / template.metadata.name
        template.save_to_directory(template_path)

        # Clear cache
        self._cache = {}

    def remove_template(self, name: str) -> bool:
        """Remove a template from the registry.

        Args:
            name: Template name

        Returns:
            True if removed
        """
        import shutil

        template_path = self.base_path / name
        if template_path.exists():
            shutil.rmtree(template_path)
            self._cache = {}
            return True
        return False

    def _load_template(self, path: Path) -> Template | None:
        """Load a template from a directory."""
        try:
            return Template.from_directory(path)
        except Exception:
            return None


@dataclass
class RemoteRegistryConfig:
    """Configuration for remote registry."""

    url: str
    api_key: str | None = None
    timeout: float = 30.0
    cache_ttl: int = 3600  # 1 hour


class RemoteRegistry(TemplateRegistry):
    """Remote template registry (e.g., community marketplace).

    Example:
        registry = RemoteRegistry(
            RemoteRegistryConfig(url="https://templates.fastagentic.dev")
        )
        templates = registry.list_templates()
    """

    def __init__(self, config: RemoteRegistryConfig) -> None:
        """Initialize the remote registry.

        Args:
            config: Registry configuration
        """
        self.config = config
        self._cache: dict[str, Any] = {}
        self._cache_time: dict[str, datetime] = {}

    def list_templates(
        self,
        category: TemplateCategory | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
    ) -> list[TemplateMetadata]:
        """List available templates."""
        # Build query params
        params: dict[str, str] = {}
        if category:
            params["category"] = category.value
        if framework:
            params["framework"] = framework
        if tags:
            params["tags"] = ",".join(tags)

        # Try cache
        cache_key = f"list:{json.dumps(params, sort_keys=True)}"
        cached = self._get_cached(cache_key)
        if cached:
            return [TemplateMetadata.from_dict(t) for t in cached]

        # Fetch from remote
        data = self._request("GET", "/templates", params=params)
        if data:
            self._set_cached(cache_key, data)
            return [TemplateMetadata.from_dict(t) for t in data]

        return []

    def get_template(self, name: str, version: str | None = None) -> Template | None:
        """Get a template by name."""
        cache_key = f"template:{name}:{version or 'latest'}"
        cached = self._get_cached(cache_key)
        if cached:
            return Template.from_dict(cached)

        # Build URL
        url = f"/templates/{name}"
        if version:
            url += f"?version={version}"

        data = self._request("GET", url)
        if data:
            self._set_cached(cache_key, data)
            return Template.from_dict(data)

        return None

    def search(self, query: str) -> list[TemplateMetadata]:
        """Search for templates."""
        data = self._request("GET", "/templates/search", params={"q": query})
        if data:
            return [TemplateMetadata.from_dict(t) for t in data]
        return []

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any | None:
        """Make an HTTP request to the registry."""
        import urllib.error
        import urllib.parse
        import urllib.request

        url = f"{self.config.url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request_data = json.dumps(data).encode() if data else None

        try:
            req = urllib.request.Request(
                url,
                data=request_data,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode())

        except urllib.error.URLError:
            return None
        except json.JSONDecodeError:
            return None

    def _get_cached(self, key: str) -> Any | None:
        """Get a cached value."""
        if key not in self._cache:
            return None

        cache_time = self._cache_time.get(key)
        if cache_time:
            age = (datetime.now() - cache_time).total_seconds()
            if age > self.config.cache_ttl:
                del self._cache[key]
                del self._cache_time[key]
                return None

        return self._cache[key]

    def _set_cached(self, key: str, value: Any) -> None:
        """Set a cached value."""
        self._cache[key] = value
        self._cache_time[key] = datetime.now()


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise registry."""

    url: str
    api_key: str
    organization: str
    timeout: float = 30.0
    cache_ttl: int = 3600
    require_approval: bool = True
    allowed_categories: list[TemplateCategory] | None = None


class EnterpriseRegistry(TemplateRegistry):
    """Enterprise template registry with access control.

    Provides private template management for organizations with:
    - Access control and permissions
    - Template approval workflows
    - Usage tracking and auditing
    - Category restrictions

    Example:
        registry = EnterpriseRegistry(
            EnterpriseConfig(
                url="https://templates.mycompany.com",
                api_key="ent-key",
                organization="mycompany",
            )
        )
    """

    def __init__(self, config: EnterpriseConfig) -> None:
        """Initialize the enterprise registry.

        Args:
            config: Registry configuration
        """
        self.config = config
        self._remote = RemoteRegistry(
            RemoteRegistryConfig(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout,
                cache_ttl=config.cache_ttl,
            )
        )

    def list_templates(
        self,
        category: TemplateCategory | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
    ) -> list[TemplateMetadata]:
        """List available templates."""
        # Check category restrictions
        if (
            self.config.allowed_categories
            and category
            and category not in self.config.allowed_categories
        ):
            return []

        templates = self._remote.list_templates(
            category=category,
            framework=framework,
            tags=tags,
        )

        # Filter by allowed categories
        if self.config.allowed_categories:
            templates = [t for t in templates if t.category in self.config.allowed_categories]

        return templates

    def get_template(self, name: str, version: str | None = None) -> Template | None:
        """Get a template by name."""
        template = self._remote.get_template(name, version)

        # Check category restrictions
        if (
            template
            and self.config.allowed_categories
            and template.metadata.category not in self.config.allowed_categories
        ):
            return None

        return template

    def search(self, query: str) -> list[TemplateMetadata]:
        """Search for templates."""
        results = self._remote.search(query)

        # Filter by allowed categories
        if self.config.allowed_categories:
            results = [t for t in results if t.category in self.config.allowed_categories]

        return results

    def publish_template(
        self,
        template: Template,
        *,
        private: bool = True,
        require_approval: bool | None = None,
    ) -> dict[str, Any]:
        """Publish a template to the enterprise registry.

        Args:
            template: Template to publish
            private: Whether the template is private to the organization
            require_approval: Override approval requirement

        Returns:
            Publication result
        """
        needs_approval = (
            require_approval if require_approval is not None else self.config.require_approval
        )

        data = {
            "template": template.to_dict(),
            "organization": self.config.organization,
            "private": private,
            "status": "pending" if needs_approval else "published",
        }

        result = self._remote._request("POST", "/templates", data=data)
        return result or {"status": "error", "message": "Failed to publish template"}

    def get_usage_stats(self, template_name: str) -> dict[str, Any]:
        """Get usage statistics for a template.

        Args:
            template_name: Template name

        Returns:
            Usage statistics
        """
        result = self._remote._request(
            "GET",
            f"/templates/{template_name}/stats",
            params={"organization": self.config.organization},
        )
        return result or {}

    def get_audit_log(
        self,
        template_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get audit log for template operations.

        Args:
            template_name: Optional filter by template
            limit: Maximum entries to return

        Returns:
            List of audit log entries
        """
        params: dict[str, str] = {
            "organization": self.config.organization,
            "limit": str(limit),
        }
        if template_name:
            params["template"] = template_name

        result = self._remote._request("GET", "/audit", params=params)
        return result or []
