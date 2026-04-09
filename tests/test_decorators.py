"""Tests for FastAgentic decorators."""

from fastagentic import prompt, resource, tool
from fastagentic.decorators import get_prompts, get_resources, get_tools


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_with_args(self) -> None:
        """Test tool decorator with arguments."""

        @tool(name="test_tool", description="A test tool")
        async def my_tool(x: int, y: str) -> str:
            return f"{x}: {y}"

        tools = get_tools()
        assert "test_tool" in tools

        defn, func = tools["test_tool"]
        assert defn.name == "test_tool"
        assert defn.description == "A test tool"
        assert "x" in defn.parameters["properties"]
        assert "y" in defn.parameters["properties"]

    def test_tool_without_args(self) -> None:
        """Test tool decorator without arguments."""

        @tool
        async def simple_tool(text: str) -> str:
            """A simple tool."""
            return text.upper()

        tools = get_tools()
        assert "simple_tool" in tools

        defn, _ = tools["simple_tool"]
        assert defn.name == "simple_tool"
        assert defn.description == "A simple tool."


class TestResourceDecorator:
    """Tests for the @resource decorator."""

    def test_resource_decorator(self) -> None:
        """Test resource decorator."""

        @resource(
            name="test_resource",
            uri="items/{item_id}",
            description="Get an item",
            cache_ttl=60,
        )
        async def get_item(item_id: str) -> dict:
            return {"id": item_id}

        resources = get_resources()
        assert "test_resource" in resources

        defn, _ = resources["test_resource"]
        assert defn.name == "test_resource"
        assert defn.uri == "items/{item_id}"
        assert defn.cache_ttl == 60


class TestPromptDecorator:
    """Tests for the @prompt decorator."""

    def test_prompt_decorator(self) -> None:
        """Test prompt decorator."""

        @prompt(name="test_prompt", description="A test prompt")
        def my_prompt(context: str) -> str:
            return f"Context: {context}"

        prompts = get_prompts()
        assert "test_prompt" in prompts

        defn, func = prompts["test_prompt"]
        assert defn.name == "test_prompt"
        assert defn.description == "A test prompt"
        assert len(defn.arguments) == 1
        assert defn.arguments[0]["name"] == "context"
