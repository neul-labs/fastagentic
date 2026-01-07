"""Template composition for FastAgentic.

Allows combining multiple templates into a single project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastagentic.templates.base import (
    Template,
    TemplateFile,
    TemplateMetadata,
    TemplateVariable,
    TemplateVersion,
)
from fastagentic.templates.registry import TemplateRegistry


@dataclass
class CompositionConfig:
    """Configuration for template composition.

    Example:
        config = CompositionConfig(
            base_template="pydanticai-agent",
            include_templates=["rag-module", "auth-module"],
            override_strategy="merge",
        )
    """

    base_template: str
    include_templates: list[str] = field(default_factory=list)
    override_strategy: str = "merge"  # merge, replace, skip
    resolve_conflicts: str = "prompt"  # prompt, base_wins, include_wins
    merge_variables: bool = True
    merge_dependencies: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_template": self.base_template,
            "include_templates": self.include_templates,
            "override_strategy": self.override_strategy,
            "resolve_conflicts": self.resolve_conflicts,
            "merge_variables": self.merge_variables,
            "merge_dependencies": self.merge_dependencies,
        }


@dataclass
class FileConflict:
    """Represents a conflict between template files."""

    path: str
    base_file: TemplateFile
    include_file: TemplateFile
    include_template: str
    resolution: str | None = None  # base, include, merge, skip

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "base_template": "base",
            "include_template": self.include_template,
            "resolution": self.resolution,
        }


@dataclass
class ComposedTemplate:
    """Result of template composition.

    Contains the merged template and any conflicts that occurred.
    """

    template: Template
    source_templates: list[str]
    conflicts: list[FileConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    composition_config: CompositionConfig | None = None

    def has_unresolved_conflicts(self) -> bool:
        """Check if there are unresolved conflicts."""
        return any(c.resolution is None for c in self.conflicts)

    def get_unresolved_conflicts(self) -> list[FileConflict]:
        """Get unresolved conflicts."""
        return [c for c in self.conflicts if c.resolution is None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template": self.template.to_dict(),
            "source_templates": self.source_templates,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "warnings": self.warnings,
            "composition_config": (
                self.composition_config.to_dict() if self.composition_config else None
            ),
        }


class TemplateComposer:
    """Composes multiple templates into a single template.

    Example:
        composer = TemplateComposer(registry)

        # Compose templates
        config = CompositionConfig(
            base_template="pydanticai-agent",
            include_templates=["rag-module", "auth-module"],
        )
        result = composer.compose(config)

        # Render composed template
        result.template.render("/path/to/output", {"project_name": "my-project"})
    """

    def __init__(self, registry: TemplateRegistry) -> None:
        """Initialize the composer.

        Args:
            registry: Template registry to load templates from
        """
        self.registry = registry

    def compose(
        self,
        config: CompositionConfig,
        variables: dict[str, Any] | None = None,
    ) -> ComposedTemplate:
        """Compose templates based on configuration.

        Args:
            config: Composition configuration
            variables: Optional variable values for conflict resolution

        Returns:
            Composed template result
        """
        # Load base template
        base = self.registry.get_template(config.base_template)
        if not base:
            raise ValueError(f"Base template not found: {config.base_template}")

        if not base.composable:
            raise ValueError(f"Base template is not composable: {config.base_template}")

        # Load include templates
        includes: list[tuple[str, Template]] = []
        for name in config.include_templates:
            template = self.registry.get_template(name)
            if not template:
                raise ValueError(f"Include template not found: {name}")
            if not template.composable:
                raise ValueError(f"Template is not composable: {name}")
            includes.append((name, template))

        # Compose templates
        return self._compose_templates(base, includes, config, variables or {})

    def _compose_templates(
        self,
        base: Template,
        includes: list[tuple[str, Template]],
        config: CompositionConfig,
        variables: dict[str, Any],
    ) -> ComposedTemplate:
        """Compose multiple templates."""
        conflicts: list[FileConflict] = []
        warnings: list[str] = []
        source_templates = [config.base_template] + [name for name, _ in includes]

        # Start with base template files
        merged_files: dict[str, TemplateFile] = {f.path: f for f in base.files}

        # Merge variables
        merged_variables: dict[str, TemplateVariable] = {v.name: v for v in base.variables}

        # Merge dependencies
        merged_deps: set[str] = set(base.dependencies)

        # Process each include template
        for include_name, include in includes:
            # Merge files
            for file in include.files:
                if file.path in merged_files:
                    # Conflict!
                    conflict = FileConflict(
                        path=file.path,
                        base_file=merged_files[file.path],
                        include_file=file,
                        include_template=include_name,
                    )

                    # Resolve based on strategy
                    resolution = self._resolve_file_conflict(conflict, config, variables)
                    conflict.resolution = resolution

                    if resolution == "include":
                        merged_files[file.path] = file
                    elif resolution == "merge":
                        merged_files[file.path] = self._merge_files(merged_files[file.path], file)
                    # else: keep base (skip)

                    conflicts.append(conflict)
                else:
                    merged_files[file.path] = file

            # Merge variables
            if config.merge_variables:
                for var in include.variables:
                    if var.name in merged_variables:
                        # Check if definitions match
                        existing = merged_variables[var.name]
                        if existing.type != var.type:
                            warnings.append(
                                f"Variable '{var.name}' has conflicting types: "
                                f"{existing.type} vs {var.type}"
                            )
                    else:
                        merged_variables[var.name] = var

            # Merge dependencies
            if config.merge_dependencies:
                merged_deps.update(include.dependencies)

        # Create composed metadata
        composed_metadata = TemplateMetadata(
            name=f"{config.base_template}-composed",
            description=f"Composed template from: {', '.join(source_templates)}",
            author=base.metadata.author,
            category=base.metadata.category,
            tags=list(set(base.metadata.tags)),
            framework=base.metadata.framework,
            frameworks=list(
                set(
                    base.metadata.frameworks
                    + [f for _, t in includes for f in t.metadata.frameworks]
                )
            ),
        )

        # Create composed template
        composed = Template(
            metadata=composed_metadata,
            variables=list(merged_variables.values()),
            files=list(merged_files.values()),
            versions=[
                TemplateVersion(
                    version="1.0.0",
                    release_date=datetime.now(),
                    changelog="Composed template",
                )
            ],
            dependencies=list(merged_deps),
            composable=True,
        )

        return ComposedTemplate(
            template=composed,
            source_templates=source_templates,
            conflicts=conflicts,
            warnings=warnings,
            composition_config=config,
        )

    def _resolve_file_conflict(
        self,
        conflict: FileConflict,
        config: CompositionConfig,
        _variables: dict[str, Any],
    ) -> str:
        """Resolve a file conflict.

        Returns:
            Resolution: "base", "include", "merge", or "skip"
        """
        if config.resolve_conflicts == "base_wins":
            return "base"
        elif config.resolve_conflicts == "include_wins":
            return "include"
        elif config.override_strategy == "merge":
            # Try to merge if files are compatible
            if self._can_merge_files(conflict.base_file, conflict.include_file):
                return "merge"
            return "base"
        elif config.override_strategy == "replace":
            return "include"
        else:
            return "skip"

    def _can_merge_files(self, file1: TemplateFile, _file2: TemplateFile) -> bool:
        """Check if two files can be merged.

        Files can be merged if they're both Python files or both config files.
        """
        path = file1.path

        # Python files - can merge imports and add to end
        if path.endswith(".py"):
            return True

        # Config files - can merge dicts
        return path.endswith((".json", ".yaml", ".yml", ".toml"))

    def _merge_files(self, base_file: TemplateFile, include_file: TemplateFile) -> TemplateFile:
        """Merge two template files."""
        path = base_file.path

        if path.endswith(".py"):
            merged_content = self._merge_python_files(base_file.content, include_file.content)
        elif path.endswith(".json"):
            merged_content = self._merge_json_files(base_file.content, include_file.content)
        elif path.endswith((".yaml", ".yml")):
            merged_content = self._merge_yaml_files(base_file.content, include_file.content)
        else:
            # Fallback: concatenate
            merged_content = base_file.content + "\n\n" + include_file.content

        return TemplateFile(
            path=base_file.path,
            content=merged_content,
            is_template=base_file.is_template or include_file.is_template,
        )

    def _merge_python_files(self, base: str, include: str) -> str:
        """Merge two Python files."""
        import re

        # Extract imports from both files
        import_pattern = r"^(from\s+\S+\s+import\s+.+|import\s+.+)$"

        base_imports = set(re.findall(import_pattern, base, re.MULTILINE))
        include_imports = set(re.findall(import_pattern, include, re.MULTILINE))

        # Remove imports from content
        base_content = re.sub(import_pattern, "", base, flags=re.MULTILINE).strip()
        include_content = re.sub(import_pattern, "", include, flags=re.MULTILINE).strip()

        # Merge imports
        all_imports = sorted(base_imports | include_imports)

        # Combine
        merged = "\n".join(all_imports) + "\n\n" + base_content + "\n\n" + include_content

        return merged.strip()

    def _merge_json_files(self, base: str, include: str) -> str:
        """Merge two JSON files."""
        import json

        try:
            base_data = json.loads(base)
            include_data = json.loads(include)

            # Deep merge dictionaries
            merged = self._deep_merge(base_data, include_data)

            return json.dumps(merged, indent=2)
        except json.JSONDecodeError:
            return base

    def _merge_yaml_files(self, base: str, include: str) -> str:
        """Merge two YAML files."""
        try:
            import yaml

            base_data = yaml.safe_load(base) or {}
            include_data = yaml.safe_load(include) or {}

            merged = self._deep_merge(base_data, include_data)

            return yaml.dump(merged, default_flow_style=False)
        except (yaml.YAMLError, TypeError):
            return base

    def _deep_merge(self, base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value

        return result

    def preview_composition(
        self,
        config: CompositionConfig,
    ) -> dict[str, Any]:
        """Preview a composition without actually composing.

        Args:
            config: Composition configuration

        Returns:
            Preview of what would be composed
        """
        # Load templates
        base = self.registry.get_template(config.base_template)
        if not base:
            raise ValueError(f"Base template not found: {config.base_template}")

        includes = []
        for name in config.include_templates:
            template = self.registry.get_template(name)
            if template:
                includes.append((name, template))

        # Find potential conflicts
        base_files = {f.path for f in base.files}
        conflicts = []

        for name, template in includes:
            for file in template.files:
                if file.path in base_files:
                    conflicts.append(
                        {
                            "path": file.path,
                            "templates": [config.base_template, name],
                        }
                    )

        # Collect all variables
        variables = {v.name: v.to_dict() for v in base.variables}
        for _name, template in includes:
            for var in template.variables:
                if var.name not in variables:
                    variables[var.name] = var.to_dict()

        return {
            "base_template": config.base_template,
            "include_templates": config.include_templates,
            "total_files": len(base_files) + sum(len(t.files) for _, t in includes),
            "potential_conflicts": conflicts,
            "variables": variables,
        }

    def save_composition(
        self,
        composed: ComposedTemplate,
        output_dir: str | Path,
    ) -> Path:
        """Save a composed template to a directory.

        Args:
            composed: Composed template
            output_dir: Directory to save to

        Returns:
            Path to saved template
        """
        output_path = Path(output_dir)
        composed.template.save_to_directory(output_path)

        # Save composition metadata
        composition_meta = output_path / "composition.json"
        import json

        composition_meta.write_text(json.dumps(composed.to_dict(), indent=2, default=str))

        return output_path
