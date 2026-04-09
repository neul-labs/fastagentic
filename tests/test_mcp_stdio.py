"""Tests for MCP stdio transport."""

import pytest

from fastagentic import App
from fastagentic.protocols.mcp_stdio import MCPMessage, MCPStdioTransport


class TestMCPMessage:
    """Tests for MCPMessage."""

    def test_to_dict_request(self):
        """Test converting request to dict."""
        msg = MCPMessage(
            id=1,
            method="tools/list",
            params={"cursor": None},
        )
        d = msg.to_dict()

        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["method"] == "tools/list"
        assert d["params"] == {"cursor": None}

    def test_to_dict_response(self):
        """Test converting response to dict."""
        msg = MCPMessage(
            id=1,
            result={"tools": []},
        )
        d = msg.to_dict()

        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["result"] == {"tools": []}

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25"},
        }
        msg = MCPMessage.from_dict(data)

        assert msg.id == 1
        assert msg.method == "initialize"
        assert msg.params == {"protocolVersion": "2025-11-25"}


class TestMCPStdioTransport:
    """Tests for MCPStdioTransport."""

    @pytest.fixture
    def app(self):
        """Create a test app."""
        return App(title="test-app", version="1.0.0")

    @pytest.fixture
    def transport(self, app):
        """Create a test transport."""
        return MCPStdioTransport(app)

    @pytest.mark.asyncio
    async def test_handle_initialize(self, transport):
        """Test handling initialize request."""
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        await transport._handle_initialize(1, {"protocolVersion": "2025-11-25"})

        assert len(responses) == 1
        result = responses[0].result
        assert result["protocolVersion"] == "2025-11-25"
        assert result["serverInfo"]["name"] == "test-app"
        assert result["serverInfo"]["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, transport):
        """Test handling tools/list request."""
        from fastagentic.decorators import _tools
        from fastagentic.types import ToolDefinition

        # Register a test tool
        _tools["test_tool"] = (
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            ),
            lambda: "test",
        )

        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        await transport._handle_tools_list(1)

        assert len(responses) == 1
        tools = responses[0].result["tools"]
        assert any(t["name"] == "test_tool" for t in tools)

        # Cleanup
        del _tools["test_tool"]

    @pytest.mark.asyncio
    async def test_handle_tools_call(self, transport):
        """Test handling tools/call request."""
        from fastagentic.decorators import _tools
        from fastagentic.types import ToolDefinition

        # Register a test tool
        async def test_func(name: str) -> str:
            return f"Hello, {name}!"

        _tools["greet"] = (
            ToolDefinition(
                name="greet",
                description="Greet someone",
                parameters={"type": "object", "properties": {"name": {"type": "string"}}},
            ),
            test_func,
        )

        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        await transport._handle_tools_call(
            1,
            {"name": "greet", "arguments": {"name": "World"}},
        )

        assert len(responses) == 1
        content = responses[0].result["content"]
        assert content[0]["type"] == "text"
        assert "Hello, World!" in content[0]["text"]

        # Cleanup
        del _tools["greet"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_not_found(self, transport):
        """Test handling tools/call for non-existent tool."""
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        await transport._handle_tools_call(
            1,
            {"name": "nonexistent", "arguments": {}},
        )

        assert len(responses) == 1
        assert responses[0].error is not None
        assert responses[0].error["code"] == -32601

    @pytest.mark.asyncio
    async def test_handle_resources_list(self, transport):
        """Test handling resources/list request."""
        from fastagentic.decorators import _resources
        from fastagentic.types import ResourceDefinition

        # Register a test resource
        _resources["test_resource"] = (
            ResourceDefinition(
                name="test_resource",
                uri="test://resource",
                description="A test resource",
            ),
            lambda: {"data": "test"},
        )

        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        await transport._handle_resources_list(1)

        assert len(responses) == 1
        resources = responses[0].result["resources"]
        assert any(r["name"] == "test_resource" for r in resources)

        # Cleanup
        del _resources["test_resource"]

    @pytest.mark.asyncio
    async def test_handle_prompts_list(self, transport):
        """Test handling prompts/list request."""
        from fastagentic.decorators import _prompts
        from fastagentic.types import PromptDefinition

        # Register a test prompt
        _prompts["test_prompt"] = (
            PromptDefinition(
                name="test_prompt",
                description="A test prompt",
                arguments=[{"name": "topic", "required": True}],
            ),
            lambda topic: f"Tell me about {topic}",
        )

        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        await transport._handle_prompts_list(1)

        assert len(responses) == 1
        prompts = responses[0].result["prompts"]
        assert any(p["name"] == "test_prompt" for p in prompts)

        # Cleanup
        del _prompts["test_prompt"]

    @pytest.mark.asyncio
    async def test_handle_ping(self, transport):
        """Test handling ping request."""
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        message = MCPMessage(id=1, method="ping")
        await transport._handle_message(message)

        assert len(responses) == 1
        assert responses[0].result == {}

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, transport):
        """Test handling unknown method."""
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        transport._send_message = mock_send

        message = MCPMessage(id=1, method="unknown/method")
        await transport._handle_message(message)

        assert len(responses) == 1
        assert responses[0].error is not None
        assert responses[0].error["code"] == -32601

    def test_stop(self, transport):
        """Test stopping the transport."""
        transport._running = True
        transport.stop()
        assert transport._running is False
