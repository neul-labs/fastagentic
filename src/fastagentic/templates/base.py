"""Base template classes for FastAgentic.

Defines the core template data structures and interfaces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TemplateCategory(str, Enum):
    """Template categories."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    RAG = "rag"
    CHATBOT = "chatbot"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    UTILITY = "utility"
    ENTERPRISE = "enterprise"


@dataclass
class TemplateVariable:
    """A variable in a template that users must provide.

    Example:
        TemplateVariable(
            name="project_name",
            description="Name of the project",
            default="my-agent",
            required=True,
        )
    """

    name: str
    description: str
    type: str = "string"  # string, number, boolean, choice
    default: Any = None
    required: bool = True
    choices: list[str] | None = None
    pattern: str | None = None  # Regex pattern for validation
    min_length: int | None = None
    max_length: int | None = None

    def validate(self, value: Any) -> list[str]:
        """Validate a value against this variable's constraints.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        if self.required and value is None:
            errors.append(f"{self.name} is required")
            return errors

        if value is None:
            return errors

        if self.type == "string":
            if not isinstance(value, str):
                errors.append(f"{self.name} must be a string")
            else:
                if self.min_length and len(value) < self.min_length:
                    errors.append(f"{self.name} must be at least {self.min_length} characters")
                if self.max_length and len(value) > self.max_length:
                    errors.append(f"{self.name} must be at most {self.max_length} characters")
                if self.pattern and not re.match(self.pattern, value):
                    errors.append(f"{self.name} does not match required pattern")

        elif self.type == "choice":
            if self.choices and value not in self.choices:
                errors.append(f"{self.name} must be one of: {', '.join(self.choices)}")

        elif self.type == "number":
            if not isinstance(value, (int, float)):
                errors.append(f"{self.name} must be a number")

        elif self.type == "boolean":  # noqa: SIM102
            if not isinstance(value, bool):
                errors.append(f"{self.name} must be a boolean")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "default": self.default,
            "required": self.required,
            "choices": self.choices,
            "pattern": self.pattern,
            "min_length": self.min_length,
            "max_length": self.max_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateVariable:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TemplateFile:
    """A file in a template.

    Example:
        TemplateFile(
            path="src/agent.py",
            content="from fastagentic import App\\n...",
            is_template=True,
        )
    """

    path: str
    content: str
    is_template: bool = True  # Whether content contains template variables
    executable: bool = False
    encoding: str = "utf-8"

    def render(self, variables: dict[str, Any]) -> str:
        """Render the file content with variables.

        Args:
            variables: Variable values to substitute

        Returns:
            Rendered content
        """
        if not self.is_template:
            return self.content

        content = self.content

        # Replace {{ variable_name }} patterns
        for name, value in variables.items():
            content = content.replace(f"{{{{ {name} }}}}", str(value))
            content = content.replace(f"{{{{{name}}}}}", str(value))

        return content

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "content": self.content,
            "is_template": self.is_template,
            "executable": self.executable,
            "encoding": self.encoding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateFile:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TemplateVersion:
    """Version information for a template.

    Example:
        TemplateVersion(
            version="1.2.0",
            release_date=datetime.now(),
            changelog="Added streaming support",
        )
    """

    version: str
    release_date: datetime
    changelog: str = ""
    min_fastagentic_version: str | None = None
    deprecated: bool = False
    deprecation_message: str | None = None

    def is_compatible(self, fastagentic_version: str) -> bool:
        """Check if this version is compatible with a FastAgentic version.

        Args:
            fastagentic_version: The FastAgentic version to check

        Returns:
            True if compatible
        """
        if not self.min_fastagentic_version:
            return True

        # Simple version comparison
        def parse_version(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split(".")[:3])

        try:
            min_ver = parse_version(self.min_fastagentic_version)
            current_ver = parse_version(fastagentic_version)
            return current_ver >= min_ver
        except ValueError:
            return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "release_date": self.release_date.isoformat(),
            "changelog": self.changelog,
            "min_fastagentic_version": self.min_fastagentic_version,
            "deprecated": self.deprecated,
            "deprecation_message": self.deprecation_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateVersion:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("release_date"), str):
            data["release_date"] = datetime.fromisoformat(data["release_date"])
        return cls(**data)


@dataclass
class TemplateMetadata:
    """Metadata for a template.

    Example:
        TemplateMetadata(
            name="rag-chatbot",
            description="A RAG-powered chatbot template",
            author="FastAgentic Team",
            category=TemplateCategory.RAG,
        )
    """

    name: str
    description: str
    author: str
    category: TemplateCategory = TemplateCategory.AGENT
    tags: list[str] = field(default_factory=list)
    license: str = "MIT"
    homepage: str | None = None
    repository: str | None = None
    documentation: str | None = None
    framework: str | None = None  # Primary framework (pydanticai, langgraph, etc.)
    frameworks: list[str] = field(default_factory=list)  # All supported frameworks
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "tags": self.tags,
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "documentation": self.documentation,
            "framework": self.framework,
            "frameworks": self.frameworks,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateMetadata:
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("category"), str):
            data["category"] = TemplateCategory(data["category"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class Template:
    """A complete template definition.

    Example:
        template = Template(
            metadata=TemplateMetadata(name="my-template", ...),
            variables=[TemplateVariable(name="project_name", ...)],
            files=[TemplateFile(path="main.py", content="...")],
        )

        # Render to a directory
        template.render("/path/to/output", {"project_name": "my-project"})
    """

    metadata: TemplateMetadata
    variables: list[TemplateVariable] = field(default_factory=list)
    files: list[TemplateFile] = field(default_factory=list)
    versions: list[TemplateVersion] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Other templates
    composable: bool = True  # Can be composed with other templates

    @property
    def current_version(self) -> TemplateVersion | None:
        """Get the current (latest) version."""
        if not self.versions:
            return None
        return sorted(self.versions, key=lambda v: v.release_date, reverse=True)[0]

    def validate_variables(self, values: dict[str, Any]) -> list[str]:
        """Validate variable values.

        Args:
            values: Variable values to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        for var in self.variables:
            value = values.get(var.name, var.default)
            errors.extend(var.validate(value))

        return errors

    def get_defaults(self) -> dict[str, Any]:
        """Get default values for all variables.

        Returns:
            Dictionary of variable defaults
        """
        return {var.name: var.default for var in self.variables}

    def render(
        self,
        output_dir: str | Path,
        variables: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> list[Path]:
        """Render the template to a directory.

        Args:
            output_dir: Directory to write files to
            variables: Variable values
            overwrite: Whether to overwrite existing files

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Merge with defaults
        all_vars = self.get_defaults()
        all_vars.update(variables)

        # Validate
        errors = self.validate_variables(all_vars)
        if errors:
            raise ValueError(f"Variable validation failed: {', '.join(errors)}")

        created_files: list[Path] = []

        for template_file in self.files:
            # Render path (may contain variables)
            file_path = template_file.path
            for name, value in all_vars.items():
                file_path = file_path.replace(f"{{{{ {name} }}}}", str(value))
                file_path = file_path.replace(f"{{{{{name}}}}}", str(value))

            full_path = output_path / file_path

            # Check overwrite
            if full_path.exists() and not overwrite:
                continue

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Render content
            content = template_file.render(all_vars)

            # Write file
            full_path.write_text(content, encoding=template_file.encoding)

            # Set executable if needed
            if template_file.executable:
                full_path.chmod(full_path.stat().st_mode | 0o111)

            created_files.append(full_path)

        return created_files

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "variables": [v.to_dict() for v in self.variables],
            "files": [f.to_dict() for f in self.files],
            "versions": [v.to_dict() for v in self.versions],
            "dependencies": self.dependencies,
            "composable": self.composable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Template:
        """Create from dictionary."""
        return cls(
            metadata=TemplateMetadata.from_dict(data["metadata"]),
            variables=[TemplateVariable.from_dict(v) for v in data.get("variables", [])],
            files=[TemplateFile.from_dict(f) for f in data.get("files", [])],
            versions=[TemplateVersion.from_dict(v) for v in data.get("versions", [])],
            dependencies=data.get("dependencies", []),
            composable=data.get("composable", True),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Template:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_directory(cls, path: str | Path) -> Template:
        """Load a template from a directory.

        Expects a template.json file in the directory.

        Args:
            path: Path to the template directory

        Returns:
            Loaded template
        """
        template_path = Path(path)
        manifest_path = template_path / "template.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"No template.json found in {path}")

        manifest = json.loads(manifest_path.read_text())

        # Load files from disk if not embedded in manifest
        if "files" not in manifest or not manifest["files"]:
            files: list[TemplateFile] = []
            for file_path in template_path.rglob("*"):
                if file_path.is_file() and file_path.name != "template.json":
                    rel_path = file_path.relative_to(template_path)
                    files.append(
                        TemplateFile(
                            path=str(rel_path),
                            content=file_path.read_text(),
                            is_template=True,
                        )
                    )
            manifest["files"] = [f.to_dict() for f in files]

        return cls.from_dict(manifest)

    def save_to_directory(self, path: str | Path) -> None:
        """Save template to a directory.

        Args:
            path: Directory path to save to
        """
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = output_path / "template.json"
        manifest_path.write_text(self.to_json())

        # Save template files
        for template_file in self.files:
            file_path = output_path / template_file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(template_file.content, encoding=template_file.encoding)
