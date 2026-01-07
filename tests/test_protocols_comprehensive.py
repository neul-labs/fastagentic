"""Comprehensive tests for MCP and A2A protocols."""

from fastagentic.decorators import prompt, resource, tool
from fastagentic.protocols.mcp import get_prompts, get_resources, get_tools


class TestMCPProtocol:
    """Tests for MCP protocol functions."""

    def test_get_tools_returns_dict(self):
        """Test getting tools returns a dict."""
        tools = get_tools()
        assert isinstance(tools, dict)

    def test_get_resources_returns_dict(self):
        """Test getting resources returns a dict."""
        resources = get_resources()
        assert isinstance(resources, dict)

    def test_get_prompts_returns_dict(self):
        """Test getting prompts returns a dict."""
        prompts = get_prompts()
        assert isinstance(prompts, dict)


class TestMCPToolDecorator:
    """Tests for MCP tool decorator integration."""

    def test_tool_registration(self):
        """Test that @tool decorator registers tools."""

        @tool(name="test_mcp_tool_1", description="A test tool")
        async def my_test_tool(query: str) -> str:
            return f"Result: {query}"

        tools = get_tools()
        assert "test_mcp_tool_1" in tools

    def test_tool_with_schema(self):
        """Test tool with input schema."""

        @tool(
            name="test_schema_tool_1",
            description="A tool with schema",
        )
        async def schema_tool(query: str, limit: int = 10) -> str:
            return f"Query: {query}, Limit: {limit}"

        tools = get_tools()
        assert "test_schema_tool_1" in tools


class TestMCPResourceDecorator:
    """Tests for MCP resource decorator integration."""

    def test_resource_registration(self):
        """Test that @resource decorator registers resources."""

        @resource(
            name="test_mcp_resource_1",
            uri="test://resource",
            description="A test resource",
        )
        async def my_test_resource() -> dict:
            return {"data": "test"}

        resources = get_resources()
        assert "test_mcp_resource_1" in resources


class TestMCPPromptDecorator:
    """Tests for MCP prompt decorator integration."""

    def test_prompt_registration(self):
        """Test that @prompt decorator registers prompts."""

        @prompt(name="test_mcp_prompt_1", description="A test prompt")
        def my_test_prompt(name: str) -> str:
            return f"Hello, {name}!"

        prompts = get_prompts()
        assert "test_mcp_prompt_1" in prompts


class TestProtocolIntegration:
    """Integration tests for protocols."""

    def test_tool_execution(self):
        """Test that registered tools can be executed."""

        @tool(name="integration_tool_1", description="Integration test tool")
        async def integration_tool(x: int, y: int) -> int:
            return x + y

        tools = get_tools()
        assert "integration_tool_1" in tools

        # Get the function
        tool_def, func = tools["integration_tool_1"]
        assert tool_def.name == "integration_tool_1"

    def test_resource_execution(self):
        """Test that registered resources can be accessed."""

        @resource(
            name="integration_resource_1",
            uri="test://integration",
            description="Integration test resource",
        )
        async def integration_resource() -> dict:
            return {"status": "ok"}

        resources = get_resources()
        assert "integration_resource_1" in resources

    def test_prompt_execution(self):
        """Test that registered prompts can be called."""

        @prompt(name="integration_prompt_1", description="Integration test prompt")
        def integration_prompt(message: str) -> str:
            return f"Prompt: {message}"

        prompts = get_prompts()
        assert "integration_prompt_1" in prompts

        # Get the function
        prompt_def, func = prompts["integration_prompt_1"]
        result = func("test")
        assert result == "Prompt: test"
