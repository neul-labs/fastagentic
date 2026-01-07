"""Prompt template system for FastAgentic."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VariableType(str, Enum):
    """Types of template variables."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    ANY = "any"


@dataclass
class PromptVariable:
    """Definition of a template variable.

    Attributes:
        name: Variable name (used in template as {{name}})
        type: Expected type of the value
        description: Human-readable description
        required: Whether the variable must be provided
        default: Default value if not provided
        validator: Optional validation function
    """

    name: str
    type: VariableType = VariableType.STRING
    description: str = ""
    required: bool = True
    default: Any = None
    validator: Callable[[Any], bool] | None = None

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """Validate a value for this variable.

        Args:
            value: The value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required and self.default is None:
                return False, f"Variable '{self.name}' is required"
            return True, None

        # Type checking
        type_valid = self._check_type(value)
        if not type_valid:
            return (
                False,
                f"Variable '{self.name}' expected {self.type.value}, got {type(value).__name__}",
            )

        # Custom validation
        if self.validator is not None:
            try:
                if not self.validator(value):
                    return False, f"Variable '{self.name}' failed custom validation"
            except Exception as e:
                return False, f"Variable '{self.name}' validation error: {e}"

        return True, None

    def _check_type(self, value: Any) -> bool:
        """Check if value matches expected type."""
        if self.type == VariableType.ANY:
            return True
        if self.type == VariableType.STRING:
            return isinstance(value, str)
        if self.type == VariableType.NUMBER:
            return isinstance(value, (int, float))
        if self.type == VariableType.BOOLEAN:
            return isinstance(value, bool)
        if self.type == VariableType.LIST:
            return isinstance(value, (list, tuple))
        if self.type == VariableType.OBJECT:
            return isinstance(value, dict)
        return True


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution.

    Supports {{variable}} syntax with optional formatting.

    Example:
        template = PromptTemplate(
            name="triage_prompt",
            content=\"\"\"
            You are a support agent for {{company}}.
            The user's name is {{user_name}}.
            Their issue category is: {{category}}

            {{#if is_premium}}
            This is a premium customer - prioritize their request.
            {{/if}}
            \"\"\",
            variables=[
                PromptVariable(name="company", required=True),
                PromptVariable(name="user_name", required=True),
                PromptVariable(name="category", default="general"),
                PromptVariable(name="is_premium", type=VariableType.BOOLEAN, default=False),
            ]
        )

        rendered = template.render(
            company="Acme Corp",
            user_name="Alice",
            is_premium=True,
        )
    """

    name: str
    content: str
    variables: list[PromptVariable] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build variable lookup after initialization."""
        self._var_map: dict[str, PromptVariable] = {v.name: v for v in self.variables}

    def get_variable(self, name: str) -> PromptVariable | None:
        """Get a variable definition by name."""
        return self._var_map.get(name)

    def get_required_variables(self) -> list[str]:
        """Get list of required variable names."""
        return [v.name for v in self.variables if v.required and v.default is None]

    def validate(self, **kwargs: Any) -> tuple[bool, list[str]]:
        """Validate all variables.

        Args:
            **kwargs: Variable values to validate

        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors: list[str] = []

        for var in self.variables:
            value = kwargs.get(var.name, var.default)
            valid, error = var.validate(value)
            if not valid and error:
                errors.append(error)

        return len(errors) == 0, errors

    def render(self, **kwargs: Any) -> str:
        """Render the template with variable substitution.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If validation fails
        """
        # Validate first
        valid, errors = self.validate(**kwargs)
        if not valid:
            raise ValueError(f"Template validation failed: {'; '.join(errors)}")

        # Build context with defaults
        context: dict[str, Any] = {}
        for var in self.variables:
            context[var.name] = kwargs.get(var.name, var.default)

        # Add any extra kwargs
        for key, value in kwargs.items():
            if key not in context:
                context[key] = value

        return render_template(self.content, context)

    def render_safe(self, **kwargs: Any) -> tuple[str | None, list[str]]:
        """Render template without raising exceptions.

        Args:
            **kwargs: Variable values

        Returns:
            Tuple of (rendered_string_or_none, errors)
        """
        valid, errors = self.validate(**kwargs)
        if not valid:
            return None, errors

        try:
            result = self.render(**kwargs)
            return result, []
        except Exception as e:
            return None, [str(e)]


def render_template(template: str, context: dict[str, Any]) -> str:
    """Render a template string with variable substitution.

    Supports:
    - {{variable}} - Simple substitution
    - {{#if condition}}...{{/if}} - Conditional blocks
    - {{#each items}}...{{/each}} - List iteration
    - {{variable | filter}} - Filters (upper, lower, capitalize, trim)

    Args:
        template: Template string
        context: Variable values

    Returns:
        Rendered string
    """
    result = template

    # Process conditionals first
    result = _process_conditionals(result, context)

    # Process each loops
    result = _process_each(result, context)

    # Process simple variables with optional filters
    result = _process_variables(result, context)

    return result


def _process_conditionals(template: str, context: dict[str, Any]) -> str:
    """Process {{#if condition}}...{{/if}} blocks."""
    pattern = r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}"

    def replacer(match: re.Match[str]) -> str:
        condition_var = match.group(1)
        content = match.group(2)

        # Check if condition is truthy
        value = context.get(condition_var)
        if value:
            return content.strip()
        return ""

    return re.sub(pattern, replacer, template, flags=re.DOTALL)


def _process_each(template: str, context: dict[str, Any]) -> str:
    """Process {{#each items}}...{{/each}} blocks."""
    pattern = r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}"

    def replacer(match: re.Match[str]) -> str:
        list_var = match.group(1)
        item_template = match.group(2)

        items = context.get(list_var, [])
        if not isinstance(items, (list, tuple)):
            return ""

        results = []
        for i, item in enumerate(items):
            item_context = {
                **context,
                "this": item,
                "index": i,
                "first": i == 0,
                "last": i == len(items) - 1,
            }
            # If item is a dict, add its keys to context
            if isinstance(item, dict):
                item_context.update(item)

            rendered = _process_variables(item_template, item_context)
            results.append(rendered.strip())

        return "\n".join(results)

    return re.sub(pattern, replacer, template, flags=re.DOTALL)


def _process_variables(template: str, context: dict[str, Any]) -> str:
    """Process {{variable}} and {{variable | filter}} substitutions."""
    pattern = r"\{\{\s*(\w+)(?:\s*\|\s*(\w+))?\s*\}\}"

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        filter_name = match.group(2)

        value = context.get(var_name, "")

        # Convert to string
        if value is None:
            str_value = ""
        elif isinstance(value, bool):
            str_value = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            import json

            str_value = json.dumps(value)
        else:
            str_value = str(value)

        # Apply filter if specified
        if filter_name:
            str_value = _apply_filter(str_value, filter_name)

        return str_value

    return re.sub(pattern, replacer, template)


def _apply_filter(value: str, filter_name: str) -> str:
    """Apply a filter to a string value."""
    filters = {
        "upper": str.upper,
        "lower": str.lower,
        "capitalize": str.capitalize,
        "title": str.title,
        "trim": str.strip,
        "strip": str.strip,
    }

    filter_func = filters.get(filter_name)
    if filter_func:
        return filter_func(value)

    return value
