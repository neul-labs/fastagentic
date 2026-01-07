"""Decorators for defining tools, resources, prompts, and agent endpoints."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

from pydantic import BaseModel

from fastagentic.types import (
    EndpointDefinition,
    PromptDefinition,
    ResourceDefinition,
    ToolDefinition,
)

P = ParamSpec("P")
T = TypeVar("T")

# Global registries for decorated functions
_tools: dict[str, tuple[ToolDefinition, Callable[..., Any]]] = {}
_resources: dict[str, tuple[ResourceDefinition, Callable[..., Any]]] = {}
_prompts: dict[str, tuple[PromptDefinition, Callable[..., Any]]] = {}
_endpoints: dict[str, tuple[EndpointDefinition, Callable[..., Any]]] = {}


def get_tools() -> dict[str, tuple[ToolDefinition, Callable[..., Any]]]:
    """Get all registered tools."""
    return _tools.copy()


def get_resources() -> dict[str, tuple[ResourceDefinition, Callable[..., Any]]]:
    """Get all registered resources."""
    return _resources.copy()


def get_prompts() -> dict[str, tuple[PromptDefinition, Callable[..., Any]]]:
    """Get all registered prompts."""
    return _prompts.copy()


def get_endpoints() -> dict[str, tuple[EndpointDefinition, Callable[..., Any]]]:
    """Get all registered endpoints."""
    return _endpoints.copy()


def reset_registries() -> None:
    """Reset all decorator registries.

    This is primarily useful for testing to ensure a clean state
    between test cases. In production, decorators are typically
    used once at module load time.
    """
    global _tools, _resources, _prompts, _endpoints
    _tools = {}
    _resources = {}
    _prompts = {}
    _endpoints = {}


def _extract_parameters_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON Schema parameters from function signature."""
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = func.__annotations__
    except AttributeError:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls", "ctx", "context"):
            continue

        param_type = hints.get(name, Any)
        param_schema = _type_to_json_schema(param_type)

        properties[name] = param_schema

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert a Python type to JSON Schema."""
    if python_type is None or python_type is type(None):
        return {"type": "null"}

    if python_type is str:
        return {"type": "string"}
    if python_type is int:
        return {"type": "integer"}
    if python_type is float:
        return {"type": "number"}
    if python_type is bool:
        return {"type": "boolean"}

    # Handle Optional, Union, list, dict via origin
    origin = getattr(python_type, "__origin__", None)

    if origin is list:
        args = getattr(python_type, "__args__", (Any,))
        return {"type": "array", "items": _type_to_json_schema(args[0])}

    if origin is dict:
        return {"type": "object"}

    # Check if it's a Pydantic model
    if isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return python_type.model_json_schema()

    # Default fallback
    return {"type": "string"}


@overload
def tool(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    scopes: list[str] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def tool(
    func: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    scopes: list[str] | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to register a function as an MCP tool.

    Can be used with or without arguments:

        @tool
        async def my_tool(x: int) -> str:
            ...

        @tool(name="custom_name", description="Does something")
        async def my_tool(x: int) -> str:
            ...
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""

        # Extract parameters from signature
        parameters = _extract_parameters_from_signature(fn)

        definition = ToolDefinition(
            name=tool_name,
            description=tool_description.strip(),
            parameters=parameters,
            scopes=scopes or [],
        )

        _tools[tool_name] = (definition, fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(*args, **kwargs)

        # Attach metadata to wrapper
        wrapper._fastagentic_tool = definition  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


@overload
def resource(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def resource(
    *,
    name: str | None = None,
    uri: str | None = None,
    description: str | None = None,
    mime_type: str = "application/json",
    scopes: list[str] | None = None,
    cache_ttl: int | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def resource(
    func: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    uri: str | None = None,
    description: str | None = None,
    mime_type: str = "application/json",
    scopes: list[str] | None = None,
    cache_ttl: int | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to register a function as an MCP resource.

    Resources provide data that LLMs can read.

        @resource(name="user-profile", uri="users/{user_id}/profile")
        async def get_user(user_id: str) -> dict:
            ...
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        resource_name = name or fn.__name__
        resource_uri = uri or fn.__name__
        resource_description = description or fn.__doc__ or ""

        definition = ResourceDefinition(
            name=resource_name,
            uri=resource_uri,
            description=resource_description.strip(),
            mime_type=mime_type,
            scopes=scopes or [],
            cache_ttl=cache_ttl,
        )

        _resources[resource_name] = (definition, fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(*args, **kwargs)

        wrapper._fastagentic_resource = definition  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


@overload
def prompt(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def prompt(
    *,
    name: str | None = None,
    description: str | None = None,
    arguments: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def prompt(
    func: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arguments: list[dict[str, Any]] | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to register a function as an MCP prompt template.

    Prompts define reusable templates that appear in MCP discovery.

        @prompt(name="triage_prompt", description="System prompt for triage")
        def triage_prompt(ticket: dict) -> str:
            return f"Triage this ticket: {ticket}"
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        prompt_name = name or fn.__name__
        prompt_description = description or fn.__doc__ or ""

        # Build arguments from function signature if not provided
        prompt_arguments = arguments
        if prompt_arguments is None:
            sig = inspect.signature(fn)
            prompt_arguments = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                prompt_arguments.append(
                    {
                        "name": param_name,
                        "description": "",
                        "required": param.default is inspect.Parameter.empty,
                    }
                )

        definition = PromptDefinition(
            name=prompt_name,
            description=prompt_description.strip(),
            arguments=prompt_arguments,
        )

        _prompts[prompt_name] = (definition, fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(*args, **kwargs)

        wrapper._fastagentic_prompt = definition  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def agent_endpoint(
    path: str,
    *,
    runnable: Any = None,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    output_model: type[BaseModel] | None = None,
    stream: bool = False,
    durable: bool = False,
    mcp_tool: str | None = None,
    a2a_skill: str | None = None,
    scopes: list[str] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to register a function as an agent endpoint.

    Agent endpoints are the primary way to expose agent functionality
    via REST, MCP, and A2A protocols.

        @agent_endpoint(
            path="/triage",
            runnable=LangGraphAdapter(graph),
            input_model=TicketIn,
            output_model=TicketOut,
            stream=True,
            durable=True,
        )
        async def triage(ticket: TicketIn) -> TicketOut:
            ...
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        endpoint_name = name or fn.__name__
        endpoint_description = description or fn.__doc__ or ""

        definition = EndpointDefinition(
            path=path,
            name=endpoint_name,
            description=endpoint_description.strip(),
            input_model=input_model,
            output_model=output_model,
            stream=stream,
            durable=durable,
            mcp_tool=mcp_tool,
            a2a_skill=a2a_skill,
            scopes=scopes or [],
        )

        _endpoints[path] = (definition, fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(*args, **kwargs)

        # Attach metadata
        wrapper._fastagentic_endpoint = definition  # type: ignore[attr-defined]
        wrapper._fastagentic_runnable = runnable  # type: ignore[attr-defined]
        return wrapper

    return decorator
