"""MCP stdio transport implementation.

Enables FastAgentic apps to communicate via stdio for MCP clients
like Claude Desktop, VS Code extensions, etc.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastagentic.app import App


@dataclass
class MCPMessage:
    """MCP JSON-RPC message."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            d["id"] = self.id
        if self.method is not None:
            d["method"] = self.method
        if self.params is not None:
            d["params"] = self.params
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPMessage:
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class MCPStdioTransport:
    """MCP stdio transport for FastAgentic apps.

    Implements the MCP 2025-11-25 protocol over stdio,
    allowing FastAgentic apps to be used as MCP servers.

    Example:
        app = App(name="my-agent")

        @app.tool()
        def search(query: str) -> str:
            return f"Results for {query}"

        if __name__ == "__main__":
            transport = MCPStdioTransport(app)
            asyncio.run(transport.run())

    Or via CLI:
        fastagentic mcp serve app:app
    """

    app: App
    _running: bool = field(default=False, init=False)
    _capabilities: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize capabilities based on registered components."""
        self._capabilities = {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": True},
            "prompts": {"listChanged": True},
        }

    async def _read_message(self) -> MCPMessage | None:
        """Read a JSON-RPC message from stdin."""
        loop = asyncio.get_event_loop()

        try:
            # Read line from stdin (blocking, run in executor)
            line = await loop.run_in_executor(None, sys.stdin.readline)

            if not line:
                return None

            line = line.strip()
            if not line:
                return None

            data = json.loads(line)
            return MCPMessage.from_dict(data)

        except json.JSONDecodeError as e:
            # Send parse error
            await self._send_error(None, -32700, f"Parse error: {e}")
            return None
        except Exception:
            return None

    async def _send_message(self, message: MCPMessage) -> None:
        """Send a JSON-RPC message to stdout."""
        loop = asyncio.get_event_loop()
        line = json.dumps(message.to_dict()) + "\n"
        await loop.run_in_executor(None, lambda: sys.stdout.write(line))
        await loop.run_in_executor(None, sys.stdout.flush)

    async def _send_result(self, id: int | str | None, result: Any) -> None:
        """Send a successful response."""
        await self._send_message(MCPMessage(id=id, result=result))

    async def _send_error(
        self,
        id: int | str | None,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        """Send an error response."""
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        await self._send_message(MCPMessage(id=id, error=error))

    async def _handle_initialize(
        self,
        id: int | str | None,
        _params: dict[str, Any] | None,
    ) -> None:
        """Handle initialize request."""
        result = {
            "protocolVersion": "2025-11-25",
            "capabilities": self._capabilities,
            "serverInfo": {
                "name": self.app.config.title,
                "version": self.app.config.version,
            },
        }
        await self._send_result(id, result)

    async def _handle_tools_list(self, id: int | str | None) -> None:
        """Handle tools/list request."""
        from fastagentic.decorators import get_tools

        tools = []
        for _name, (definition, _) in get_tools().items():
            tool_schema = {
                "name": definition.name,
                "description": definition.description,
                "inputSchema": definition.parameters,
            }
            tools.append(tool_schema)

        await self._send_result(id, {"tools": tools})

    async def _handle_tools_call(
        self,
        id: int | str | None,
        params: dict[str, Any] | None,
    ) -> None:
        """Handle tools/call request."""
        from fastagentic.decorators import get_tools

        if not params:
            await self._send_error(id, -32602, "Missing params")
            return

        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        tools = get_tools()
        if tool_name not in tools:
            await self._send_error(id, -32601, f"Tool not found: {tool_name}")
            return

        _, func = tools[tool_name]

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            # Format as MCP tool result
            content = [{"type": "text", "text": str(result)}]
            await self._send_result(id, {"content": content})

        except Exception as e:
            await self._send_result(
                id,
                {"content": [{"type": "text", "text": str(e)}], "isError": True},
            )

    async def _handle_resources_list(self, id: int | str | None) -> None:
        """Handle resources/list request."""
        from fastagentic.decorators import get_resources

        resources = []
        for name, (definition, _) in get_resources().items():
            resource_schema = {
                "uri": definition.uri or f"resource://{name}",
                "name": definition.name,
                "description": definition.description,
                "mimeType": definition.mime_type,
            }
            resources.append(resource_schema)

        await self._send_result(id, {"resources": resources})

    async def _handle_resources_read(
        self,
        id: int | str | None,
        params: dict[str, Any] | None,
    ) -> None:
        """Handle resources/read request."""
        from fastagentic.decorators import get_resources

        if not params:
            await self._send_error(id, -32602, "Missing params")
            return

        uri = params.get("uri")

        # Find resource by URI
        resource_def = None
        resource_func = None
        for name, (definition, func) in get_resources().items():
            if definition.uri == uri or f"resource://{name}" == uri:
                resource_def = definition
                resource_func = func
                break

        if not resource_def or not resource_func:
            await self._send_error(id, -32601, f"Resource not found: {uri}")
            return

        try:
            if asyncio.iscoroutinefunction(resource_func):
                result = await resource_func()
            else:
                result = resource_func()

            content = {
                "uri": uri,
                "mimeType": resource_def.mime_type,
                "text": json.dumps(result) if not isinstance(result, str) else result,
            }
            await self._send_result(id, {"contents": [content]})

        except Exception as e:
            await self._send_error(id, -32603, f"Resource error: {e}")

    async def _handle_prompts_list(self, id: int | str | None) -> None:
        """Handle prompts/list request."""
        from fastagentic.decorators import get_prompts

        prompts = []
        for _name, (definition, _) in get_prompts().items():
            prompt_schema = {
                "name": definition.name,
                "description": definition.description,
                "arguments": definition.arguments or [],
            }
            prompts.append(prompt_schema)

        await self._send_result(id, {"prompts": prompts})

    async def _handle_prompts_get(
        self,
        id: int | str | None,
        params: dict[str, Any] | None,
    ) -> None:
        """Handle prompts/get request."""
        from fastagentic.decorators import get_prompts

        if not params:
            await self._send_error(id, -32602, "Missing params")
            return

        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        prompts = get_prompts()
        if prompt_name not in prompts:
            await self._send_error(id, -32601, f"Prompt not found: {prompt_name}")
            return

        _, func = prompts[prompt_name]

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            messages = [{"role": "user", "content": {"type": "text", "text": str(result)}}]
            await self._send_result(id, {"messages": messages})

        except Exception as e:
            await self._send_error(id, -32603, f"Prompt error: {e}")

    async def _handle_message(self, message: MCPMessage) -> None:
        """Handle an incoming JSON-RPC message."""
        method = message.method

        if method == "initialize":
            await self._handle_initialize(message.id, message.params)
        elif method == "initialized":
            # Notification, no response needed
            pass
        elif method == "tools/list":
            await self._handle_tools_list(message.id)
        elif method == "tools/call":
            await self._handle_tools_call(message.id, message.params)
        elif method == "resources/list":
            await self._handle_resources_list(message.id)
        elif method == "resources/read":
            await self._handle_resources_read(message.id, message.params)
        elif method == "prompts/list":
            await self._handle_prompts_list(message.id)
        elif method == "prompts/get":
            await self._handle_prompts_get(message.id, message.params)
        elif method == "ping":
            await self._send_result(message.id, {})
        else:
            await self._send_error(
                message.id,
                -32601,
                f"Method not found: {method}",
            )

    async def run(self) -> None:
        """Run the stdio transport.

        Reads JSON-RPC messages from stdin and writes responses to stdout.
        """
        self._running = True

        while self._running:
            message = await self._read_message()
            if message is None:
                # EOF or error
                break

            await self._handle_message(message)

    def stop(self) -> None:
        """Stop the transport."""
        self._running = False


async def serve_stdio(app: App) -> None:
    """Serve a FastAgentic app via MCP stdio transport.

    Args:
        app: The FastAgentic app to serve
    """
    transport = MCPStdioTransport(app)
    await transport.run()
