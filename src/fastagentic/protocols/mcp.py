"""MCP (Model Context Protocol) implementation.

Implements the MCP 2025-11-25 specification for exposing tools,
resources, and prompts to LLM hosts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Path, Request
from fastapi.responses import JSONResponse

from fastagentic.decorators import get_prompts, get_resources, get_tools

if TYPE_CHECKING:
    from fastagentic.app import App

# MCP Protocol Version
MCP_VERSION = "2025-11-25"


def configure_mcp(
    app: App,
    *,
    enabled: bool = True,
    path_prefix: str = "/mcp",
    _require_auth: bool = False,
    capabilities: dict[str, bool] | None = None,
) -> None:
    """Configure MCP protocol routes on an App.

    This function adds MCP-compliant endpoints for tool, resource,
    and prompt discovery and invocation.

    Args:
        app: The FastAgentic App instance
        enabled: Whether to enable MCP routes
        path_prefix: URL prefix for MCP routes
        require_auth: Whether to require authentication
        capabilities: Override default capabilities

    Example:
        from fastagentic import App
        from fastagentic.protocols.mcp import configure_mcp

        app = App(title="My Agent")
        configure_mcp(app, path_prefix="/mcp")
    """
    if not enabled:
        return

    fastapi = app.fastapi

    default_capabilities = {
        "tools": True,
        "resources": True,
        "prompts": True,
        "sampling": False,
    }
    caps = {**default_capabilities, **(capabilities or {})}

    # MCP Discovery endpoint
    @fastapi.get(f"{path_prefix}/discovery")
    async def mcp_discovery() -> dict[str, Any]:
        """MCP discovery endpoint."""
        return {
            "protocolVersion": MCP_VERSION,
            "serverInfo": {
                "name": app.config.title,
                "version": app.config.version,
            },
            "capabilities": caps,
        }

    # Tools listing
    @fastapi.get(f"{path_prefix}/tools")
    async def mcp_list_tools() -> dict[str, Any]:
        """List available MCP tools."""
        tools = get_tools()
        return {
            "tools": [
                {
                    "name": defn.name,
                    "description": defn.description,
                    "inputSchema": defn.parameters,
                }
                for defn, _ in tools.values()
            ]
        }

    # Tool invocation
    @fastapi.post(f"{path_prefix}/tools/{{tool_name}}", response_model=None)
    async def mcp_call_tool(
        tool_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """Invoke an MCP tool."""
        tools = get_tools()

        if tool_name not in tools:
            return JSONResponse(
                status_code=404,
                content={"error": f"Tool '{tool_name}' not found"},
            )

        defn, func = tools[tool_name]

        try:
            body = await request.json()
            arguments = body.get("arguments", {})

            # Call the tool function
            if hasattr(func, "__wrapped__"):
                result = await func.__wrapped__(**arguments)
            else:
                result = await func(**arguments)

            return JSONResponse(
                content={
                    "content": [
                        {
                            "type": "text",
                            "text": str(result) if not isinstance(result, dict) else None,
                            **(result if isinstance(result, dict) else {}),
                        }
                    ]
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    # Resources listing
    @fastapi.get(f"{path_prefix}/resources")
    async def mcp_list_resources() -> dict[str, Any]:
        """List available MCP resources."""
        resources = get_resources()
        return {
            "resources": [
                {
                    "name": defn.name,
                    "uri": defn.uri,
                    "description": defn.description,
                    "mimeType": defn.mime_type,
                }
                for defn, _ in resources.values()
            ]
        }

    # Resource reading
    @fastapi.get(f"{path_prefix}/resources/{{resource_name}}", response_model=None)
    async def mcp_read_resource(
        resource_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """Read an MCP resource."""
        resources = get_resources()

        if resource_name not in resources:
            return JSONResponse(
                status_code=404,
                content={"error": f"Resource '{resource_name}' not found"},
            )

        defn, func = resources[resource_name]

        try:
            # Extract path parameters from query
            params = dict(request.query_params)

            if hasattr(func, "__wrapped__"):
                result = await func.__wrapped__(**params)
            else:
                result = await func(**params)

            return JSONResponse(
                content={
                    "contents": [
                        {
                            "uri": defn.uri,
                            "mimeType": defn.mime_type,
                            "text": str(result) if not isinstance(result, dict) else None,
                            **(result if isinstance(result, dict) else {}),
                        }
                    ]
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    # Prompts listing
    @fastapi.get(f"{path_prefix}/prompts")
    async def mcp_list_prompts() -> dict[str, Any]:
        """List available MCP prompts."""
        prompts = get_prompts()
        return {
            "prompts": [
                {
                    "name": defn.name,
                    "description": defn.description,
                    "arguments": defn.arguments,
                }
                for defn, _ in prompts.values()
            ]
        }

    # Prompt rendering
    @fastapi.post(f"{path_prefix}/prompts/{{prompt_name}}", response_model=None)
    async def mcp_get_prompt(
        prompt_name: str = Path(..., pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """Get a rendered MCP prompt."""
        prompts = get_prompts()

        if prompt_name not in prompts:
            return JSONResponse(
                status_code=404,
                content={"error": f"Prompt '{prompt_name}' not found"},
            )

        defn, func = prompts[prompt_name]

        try:
            body = await request.json()
            arguments = body.get("arguments", {})

            if hasattr(func, "__wrapped__"):
                result = func.__wrapped__(**arguments)
            else:
                result = func(**arguments)

            # Handle async functions
            if hasattr(result, "__await__"):
                result = await result

            return JSONResponse(
                content={
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": str(result),
                            },
                        }
                    ]
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )
