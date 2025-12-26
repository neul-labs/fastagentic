"""Protocol implementations for MCP and A2A."""

from fastagentic.protocols.a2a import configure_a2a
from fastagentic.protocols.mcp import configure_mcp
from fastagentic.protocols.mcp_stdio import MCPStdioTransport, serve_stdio

__all__ = ["configure_mcp", "configure_a2a", "MCPStdioTransport", "serve_stdio"]
