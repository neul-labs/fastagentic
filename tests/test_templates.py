"""Tests for template ecosystem."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from fastagentic.templates import (
    CompositionConfig,
    LocalRegistry,
    Marketplace,
    MarketplaceConfig,
    Template,
    TemplateComposer,
    TemplateFile,
    TemplateMetadata,
    TemplateRating,
    TemplateReview,
    TemplateVariable,
    TemplateVersion,
)
from fastagentic.templates.base import TemplateCategory

# ============================================================================
# TemplateVariable Tests
# ============================================================================


class TestTemplateVariable:
    """Tests for TemplateVariable."""

    def test_create_variable(self):
        """Test creating a variable."""
        var = TemplateVariable(
            name="project_name",
            description="Name of the project",
            default="my-project",
        )

        assert var.name == "project_name"
        assert var.default == "my-project"
        assert var.required is True

    def test_validate_required(self):
        """Test validating required variable."""
        var = TemplateVariable(
            name="name",
            description="Required name",
            required=True,
        )

        errors = var.validate(None)
        assert len(errors) == 1
        assert "required" in errors[0]

    def test_validate_string_length(self):
        """Test validating string length."""
        var = TemplateVariable(
            name="name",
            description="Name",
            type="string",
            min_length=3,
            max_length=10,
        )

        assert var.validate("ab") != []
        assert var.validate("abc") == []
        assert var.validate("abcdefghijk") != []

    def test_validate_choice(self):
        """Test validating choice variable."""
        var = TemplateVariable(
            name="framework",
            description="Framework",
            type="choice",
            choices=["pydanticai", "langgraph"],
        )

        assert var.validate("pydanticai") == []
        assert var.validate("invalid") != []

    def test_to_dict(self):
        """Test serialization."""
        var = TemplateVariable(
            name="name",
            description="Name",
            default="value",
        )

        data = var.to_dict()
        assert data["name"] == "name"
        assert data["default"] == "value"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "name": "name",
            "description": "Description",
            "type": "string",
            "default": "value",
            "required": False,
        }

        var = TemplateVariable.from_dict(data)
        assert var.name == "name"
        assert var.default == "value"


# ============================================================================
# TemplateFile Tests
# ============================================================================


class TestTemplateFile:
    """Tests for TemplateFile."""

    def test_create_file(self):
        """Test creating a file."""
        file = TemplateFile(
            path="src/main.py",
            content="print('hello')",
        )

        assert file.path == "src/main.py"
        assert file.is_template is True

    def test_render_template(self):
        """Test rendering template variables."""
        file = TemplateFile(
            path="src/main.py",
            content="PROJECT = '{{ project_name }}'",
        )

        rendered = file.render({"project_name": "my-project"})
        assert rendered == "PROJECT = 'my-project'"

    def test_render_non_template(self):
        """Test rendering non-template file."""
        file = TemplateFile(
            path="data.txt",
            content="{{ not_replaced }}",
            is_template=False,
        )

        rendered = file.render({"not_replaced": "value"})
        assert rendered == "{{ not_replaced }}"


# ============================================================================
# TemplateVersion Tests
# ============================================================================


class TestTemplateVersion:
    """Tests for TemplateVersion."""

    def test_create_version(self):
        """Test creating a version."""
        version = TemplateVersion(
            version="1.0.0",
            release_date=datetime.now(),
            changelog="Initial release",
        )

        assert version.version == "1.0.0"
        assert version.deprecated is False

    def test_compatibility_check(self):
        """Test version compatibility."""
        version = TemplateVersion(
            version="1.0.0",
            release_date=datetime.now(),
            min_fastagentic_version="1.0.0",
        )

        assert version.is_compatible("1.0.0")
        assert version.is_compatible("1.1.0")
        assert not version.is_compatible("0.5.0")


# ============================================================================
# TemplateMetadata Tests
# ============================================================================


class TestTemplateMetadata:
    """Tests for TemplateMetadata."""

    def test_create_metadata(self):
        """Test creating metadata."""
        meta = TemplateMetadata(
            name="my-template",
            description="A test template",
            author="Test Author",
            category=TemplateCategory.AGENT,
        )

        assert meta.name == "my-template"
        assert meta.category == TemplateCategory.AGENT

    def test_to_dict(self):
        """Test serialization."""
        meta = TemplateMetadata(
            name="my-template",
            description="Test",
            author="Author",
            tags=["test", "example"],
        )

        data = meta.to_dict()
        assert data["name"] == "my-template"
        assert data["tags"] == ["test", "example"]


# ============================================================================
# Template Tests
# ============================================================================


class TestTemplate:
    """Tests for Template."""

    def test_create_template(self):
        """Test creating a template."""
        template = Template(
            metadata=TemplateMetadata(
                name="test",
                description="Test",
                author="Author",
            ),
            variables=[
                TemplateVariable(name="name", description="Name"),
            ],
            files=[
                TemplateFile(path="main.py", content="print('{{ name }}')"),
            ],
        )

        assert template.metadata.name == "test"
        assert len(template.variables) == 1
        assert len(template.files) == 1

    def test_get_defaults(self):
        """Test getting default values."""
        template = Template(
            metadata=TemplateMetadata(name="test", description="Test", author="Author"),
            variables=[
                TemplateVariable(name="name", description="Name", default="default"),
                TemplateVariable(name="version", description="Version", default="1.0"),
            ],
        )

        defaults = template.get_defaults()
        assert defaults["name"] == "default"
        assert defaults["version"] == "1.0"

    def test_validate_variables(self):
        """Test validating variables."""
        template = Template(
            metadata=TemplateMetadata(name="test", description="Test", author="Author"),
            variables=[
                TemplateVariable(name="name", description="Name", required=True),
            ],
        )

        errors = template.validate_variables({})
        assert len(errors) == 1

        errors = template.validate_variables({"name": "value"})
        assert len(errors) == 0

    def test_render_template(self):
        """Test rendering template."""
        template = Template(
            metadata=TemplateMetadata(name="test", description="Test", author="Author"),
            variables=[
                TemplateVariable(name="name", description="Name", default="project"),
            ],
            files=[
                TemplateFile(path="README.md", content="# {{ name }}"),
                TemplateFile(path="src/main.py", content="NAME = '{{ name }}'"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = template.render(tmpdir, {"name": "my-project"})

            assert len(files) == 2

            readme = Path(tmpdir) / "README.md"
            assert readme.read_text() == "# my-project"

            main = Path(tmpdir) / "src" / "main.py"
            assert main.read_text() == "NAME = 'my-project'"

    def test_to_json(self):
        """Test JSON serialization."""
        template = Template(
            metadata=TemplateMetadata(name="test", description="Test", author="Author"),
            files=[TemplateFile(path="main.py", content="print('hi')")],
        )

        json_str = template.to_json()
        assert '"name": "test"' in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        template = Template(
            metadata=TemplateMetadata(name="test", description="Test", author="Author"),
        )

        json_str = template.to_json()
        loaded = Template.from_json(json_str)

        assert loaded.metadata.name == "test"


# ============================================================================
# LocalRegistry Tests
# ============================================================================


class TestLocalRegistry:
    """Tests for LocalRegistry."""

    def test_create_registry(self):
        """Test creating a registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)
            assert registry.base_path == Path(tmpdir)

    def test_list_empty_registry(self):
        """Test listing empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)
            templates = registry.list_templates()
            assert templates == []

    def test_add_and_get_template(self):
        """Test adding and getting a template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)

            template = Template(
                metadata=TemplateMetadata(
                    name="test-template",
                    description="Test",
                    author="Author",
                ),
                files=[TemplateFile(path="main.py", content="print('hi')")],
            )

            registry.add_template(template)

            loaded = registry.get_template("test-template")
            assert loaded is not None
            assert loaded.metadata.name == "test-template"

    def test_search_templates(self):
        """Test searching templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)

            template = Template(
                metadata=TemplateMetadata(
                    name="rag-chatbot",
                    description="RAG-powered chatbot",
                    author="Author",
                    tags=["rag", "chatbot"],
                ),
            )

            registry.add_template(template)

            results = registry.search("rag")
            assert len(results) == 1
            assert results[0].name == "rag-chatbot"

    def test_remove_template(self):
        """Test removing a template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)

            template = Template(
                metadata=TemplateMetadata(
                    name="to-remove",
                    description="Test",
                    author="Author",
                ),
            )

            registry.add_template(template)
            assert registry.template_exists("to-remove")

            registry.remove_template("to-remove")
            assert not registry.template_exists("to-remove")


# ============================================================================
# Template Rating/Review Tests
# ============================================================================


class TestTemplateRating:
    """Tests for TemplateRating."""

    def test_create_rating(self):
        """Test creating a rating."""
        rating = TemplateRating(
            template_name="test",
            user_id="user-1",
            score=5,
        )

        assert rating.score == 5

    def test_invalid_score(self):
        """Test invalid rating score."""
        with pytest.raises(ValueError):
            TemplateRating(
                template_name="test",
                user_id="user-1",
                score=6,
            )


class TestTemplateReview:
    """Tests for TemplateReview."""

    def test_create_review(self):
        """Test creating a review."""
        review = TemplateReview(
            template_name="test",
            user_id="user-1",
            title="Great template!",
            content="Works perfectly.",
            rating=5,
        )

        assert review.title == "Great template!"
        assert review.rating == 5

    def test_invalid_title(self):
        """Test invalid review title."""
        with pytest.raises(ValueError):
            TemplateReview(
                template_name="test",
                user_id="user-1",
                title="Hi",  # Too short
                content="Content",
                rating=5,
            )


# ============================================================================
# Marketplace Tests
# ============================================================================


class TestMarketplace:
    """Tests for Marketplace."""

    def test_create_marketplace(self):
        """Test creating a marketplace."""
        marketplace = Marketplace()
        assert marketplace.config.url == "https://templates.fastagentic.dev"

    def test_custom_config(self):
        """Test marketplace with custom config."""
        config = MarketplaceConfig(
            url="https://custom.example.com",
            api_key="test-key",
        )
        marketplace = Marketplace(config)

        assert marketplace.config.url == "https://custom.example.com"


# ============================================================================
# Template Composer Tests
# ============================================================================


class TestCompositionConfig:
    """Tests for CompositionConfig."""

    def test_create_config(self):
        """Test creating composition config."""
        config = CompositionConfig(
            base_template="base",
            include_templates=["module-a", "module-b"],
        )

        assert config.base_template == "base"
        assert len(config.include_templates) == 2


class TestTemplateComposer:
    """Tests for TemplateComposer."""

    def test_create_composer(self):
        """Test creating a composer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)
            composer = TemplateComposer(registry)

            assert composer.registry is registry

    def test_compose_templates(self):
        """Test composing templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)

            # Create base template
            base = Template(
                metadata=TemplateMetadata(
                    name="base",
                    description="Base template",
                    author="Author",
                ),
                variables=[
                    TemplateVariable(name="name", description="Name"),
                ],
                files=[
                    TemplateFile(path="main.py", content="# Base"),
                ],
            )
            registry.add_template(base)

            # Create module template
            module = Template(
                metadata=TemplateMetadata(
                    name="module",
                    description="Module template",
                    author="Author",
                ),
                files=[
                    TemplateFile(path="utils.py", content="# Utils"),
                ],
            )
            registry.add_template(module)

            # Compose
            composer = TemplateComposer(registry)
            config = CompositionConfig(
                base_template="base",
                include_templates=["module"],
            )

            result = composer.compose(config)

            assert len(result.template.files) == 2
            assert "base" in result.source_templates
            assert "module" in result.source_templates

    def test_compose_with_conflict(self):
        """Test composing with file conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)

            # Create base template
            base = Template(
                metadata=TemplateMetadata(
                    name="base",
                    description="Base",
                    author="Author",
                ),
                files=[
                    TemplateFile(path="config.json", content='{"base": true}'),
                ],
            )
            registry.add_template(base)

            # Create module with same file
            module = Template(
                metadata=TemplateMetadata(
                    name="module",
                    description="Module",
                    author="Author",
                ),
                files=[
                    TemplateFile(path="config.json", content='{"module": true}'),
                ],
            )
            registry.add_template(module)

            # Compose
            composer = TemplateComposer(registry)
            config = CompositionConfig(
                base_template="base",
                include_templates=["module"],
            )

            result = composer.compose(config)

            assert len(result.conflicts) == 1
            assert result.conflicts[0].path == "config.json"

    def test_preview_composition(self):
        """Test previewing composition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalRegistry(tmpdir)

            base = Template(
                metadata=TemplateMetadata(
                    name="base",
                    description="Base",
                    author="Author",
                ),
                variables=[
                    TemplateVariable(name="name", description="Name"),
                ],
                files=[
                    TemplateFile(path="main.py", content="# Main"),
                ],
            )
            registry.add_template(base)

            composer = TemplateComposer(registry)
            config = CompositionConfig(base_template="base")

            preview = composer.preview_composition(config)

            assert preview["base_template"] == "base"
            assert "name" in preview["variables"]
