"""Template marketplace for FastAgentic.

Provides community marketplace features including ratings, reviews,
and discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastagentic.templates.base import Template, TemplateCategory, TemplateMetadata
from fastagentic.templates.registry import RemoteRegistry, RemoteRegistryConfig


@dataclass
class TemplateRating:
    """A rating for a template.

    Example:
        rating = TemplateRating(
            template_name="rag-chatbot",
            user_id="user-123",
            score=5,
        )
    """

    template_name: str
    user_id: str
    score: int  # 1-5
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate rating score."""
        if not 1 <= self.score <= 5:
            raise ValueError("Rating score must be between 1 and 5")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_name": self.template_name,
            "user_id": self.user_id,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateRating:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class TemplateReview:
    """A review for a template.

    Example:
        review = TemplateReview(
            template_name="rag-chatbot",
            user_id="user-123",
            title="Great template!",
            content="Easy to use and well documented.",
            rating=5,
        )
    """

    template_name: str
    user_id: str
    title: str
    content: str
    rating: int  # 1-5
    helpful_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    verified_user: bool = False

    def __post_init__(self) -> None:
        """Validate review."""
        if not 1 <= self.rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        if len(self.title) < 3:
            raise ValueError("Title must be at least 3 characters")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_name": self.template_name,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "rating": self.rating,
            "helpful_count": self.helpful_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "verified_user": self.verified_user,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateReview:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class MarketplaceConfig:
    """Configuration for marketplace."""

    url: str = "https://templates.fastagentic.dev"
    api_key: str | None = None
    timeout: float = 30.0
    cache_ttl: int = 3600


class Marketplace:
    """Community template marketplace.

    Provides template discovery, ratings, reviews, and publishing.

    Example:
        marketplace = Marketplace()

        # Browse templates
        popular = marketplace.get_popular()
        recent = marketplace.get_recent()

        # Search
        results = marketplace.search("rag chatbot")

        # Get reviews
        reviews = marketplace.get_reviews("rag-chatbot")

        # Rate a template
        marketplace.rate_template("rag-chatbot", 5)
    """

    def __init__(self, config: MarketplaceConfig | None = None) -> None:
        """Initialize the marketplace.

        Args:
            config: Optional marketplace configuration
        """
        self.config = config or MarketplaceConfig()
        self._registry = RemoteRegistry(
            RemoteRegistryConfig(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                cache_ttl=self.config.cache_ttl,
            )
        )

    def browse(
        self,
        category: TemplateCategory | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        sort: str = "popular",  # popular, recent, rating
        page: int = 1,
        per_page: int = 20,
    ) -> list[TemplateMetadata]:
        """Browse marketplace templates.

        Args:
            category: Filter by category
            framework: Filter by framework
            tags: Filter by tags
            sort: Sort order
            page: Page number
            per_page: Items per page

        Returns:
            List of template metadata
        """
        templates = self._registry.list_templates(
            category=category,
            framework=framework,
            tags=tags,
        )

        # Sort
        if sort == "popular":
            templates.sort(key=lambda t: t.downloads, reverse=True)
        elif sort == "recent":
            templates.sort(key=lambda t: t.updated_at, reverse=True)
        elif sort == "rating":
            templates.sort(key=lambda t: t.rating, reverse=True)

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page

        return templates[start:end]

    def get_popular(self, limit: int = 10) -> list[TemplateMetadata]:
        """Get popular templates.

        Args:
            limit: Maximum templates to return

        Returns:
            List of popular template metadata
        """
        return self.browse(sort="popular", per_page=limit)

    def get_recent(self, limit: int = 10) -> list[TemplateMetadata]:
        """Get recent templates.

        Args:
            limit: Maximum templates to return

        Returns:
            List of recent template metadata
        """
        return self.browse(sort="recent", per_page=limit)

    def get_top_rated(self, limit: int = 10) -> list[TemplateMetadata]:
        """Get top-rated templates.

        Args:
            limit: Maximum templates to return

        Returns:
            List of top-rated template metadata
        """
        return self.browse(sort="rating", per_page=limit)

    def get_by_framework(self, framework: str, limit: int = 10) -> list[TemplateMetadata]:
        """Get templates for a specific framework.

        Args:
            framework: Framework name
            limit: Maximum templates to return

        Returns:
            List of template metadata
        """
        return self.browse(framework=framework, per_page=limit)

    def get_by_category(
        self, category: TemplateCategory, limit: int = 10
    ) -> list[TemplateMetadata]:
        """Get templates for a specific category.

        Args:
            category: Template category
            limit: Maximum templates to return

        Returns:
            List of template metadata
        """
        return self.browse(category=category, per_page=limit)

    def search(self, query: str, limit: int = 20) -> list[TemplateMetadata]:
        """Search for templates.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching template metadata
        """
        results = self._registry.search(query)
        return results[:limit]

    def get_template(self, name: str, version: str | None = None) -> Template | None:
        """Get a template by name.

        Args:
            name: Template name
            version: Optional specific version

        Returns:
            Template or None if not found
        """
        template = self._registry.get_template(name, version)

        # Track download
        if template:
            self._track_download(name)

        return template

    def get_reviews(
        self,
        template_name: str,
        page: int = 1,
        per_page: int = 10,
        sort: str = "helpful",
    ) -> list[TemplateReview]:
        """Get reviews for a template.

        Args:
            template_name: Template name
            page: Page number
            per_page: Items per page
            sort: Sort order (helpful, recent, rating)

        Returns:
            List of reviews
        """
        params = {
            "page": str(page),
            "per_page": str(per_page),
            "sort": sort,
        }

        result = self._registry._request(
            "GET",
            f"/templates/{template_name}/reviews",
            params=params,
        )

        if result:
            return [TemplateReview.from_dict(r) for r in result]
        return []

    def rate_template(self, template_name: str, score: int) -> bool:
        """Rate a template.

        Args:
            template_name: Template name
            score: Rating score (1-5)

        Returns:
            True if successful
        """
        if not 1 <= score <= 5:
            raise ValueError("Score must be between 1 and 5")

        result = self._registry._request(
            "POST",
            f"/templates/{template_name}/ratings",
            data={"score": score},
        )

        return result is not None

    def submit_review(self, review: TemplateReview) -> bool:
        """Submit a review for a template.

        Args:
            review: The review to submit

        Returns:
            True if successful
        """
        result = self._registry._request(
            "POST",
            f"/templates/{review.template_name}/reviews",
            data=review.to_dict(),
        )

        return result is not None

    def mark_review_helpful(self, template_name: str, review_id: str) -> bool:
        """Mark a review as helpful.

        Args:
            template_name: Template name
            review_id: Review ID

        Returns:
            True if successful
        """
        result = self._registry._request(
            "POST",
            f"/templates/{template_name}/reviews/{review_id}/helpful",
        )

        return result is not None

    def publish_template(
        self,
        template: Template,
        *,
        publish_files: bool = True,
    ) -> dict[str, Any]:
        """Publish a template to the marketplace.

        Args:
            template: Template to publish
            publish_files: Whether to include template files

        Returns:
            Publication result
        """
        data: dict[str, Any] = {"metadata": template.metadata.to_dict()}

        if publish_files:
            data["template"] = template.to_dict()

        result = self._registry._request("POST", "/templates", data=data)
        return result or {"status": "error", "message": "Failed to publish"}

    def update_template(
        self,
        template_name: str,
        template: Template,
    ) -> dict[str, Any]:
        """Update an existing template.

        Args:
            template_name: Template name
            template: Updated template

        Returns:
            Update result
        """
        result = self._registry._request(
            "PUT",
            f"/templates/{template_name}",
            data=template.to_dict(),
        )

        return result or {"status": "error", "message": "Failed to update"}

    def deprecate_template(
        self,
        template_name: str,
        message: str,
        replacement: str | None = None,
    ) -> bool:
        """Deprecate a template.

        Args:
            template_name: Template name
            message: Deprecation message
            replacement: Optional replacement template name

        Returns:
            True if successful
        """
        result = self._registry._request(
            "POST",
            f"/templates/{template_name}/deprecate",
            data={
                "message": message,
                "replacement": replacement,
            },
        )

        return result is not None

    def get_categories(self) -> list[dict[str, Any]]:
        """Get available categories with counts.

        Returns:
            List of categories with template counts
        """
        result = self._registry._request("GET", "/categories")
        return result or []

    def get_frameworks(self) -> list[dict[str, Any]]:
        """Get available frameworks with counts.

        Returns:
            List of frameworks with template counts
        """
        result = self._registry._request("GET", "/frameworks")
        return result or []

    def get_tags(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get popular tags.

        Args:
            limit: Maximum tags to return

        Returns:
            List of tags with usage counts
        """
        result = self._registry._request("GET", "/tags", params={"limit": str(limit)})
        return result or []

    def _track_download(self, template_name: str) -> None:
        """Track a template download."""
        self._registry._request(
            "POST",
            f"/templates/{template_name}/download",
        )
