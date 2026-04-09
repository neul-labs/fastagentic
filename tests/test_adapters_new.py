"""Tests for new adapters (Semantic Kernel, AutoGen, LlamaIndex, DSPy)."""

import pytest

from fastagentic.adapters import (
    AutoGenAdapter,
    DSPyAdapter,
    DSPyProgramAdapter,
    LlamaIndexAdapter,
    SemanticKernelAdapter,
)
from fastagentic.adapters.sdk import (
    AdapterMetadata,
    AdapterRegistry,
    CommunityAdapter,
    SimpleAdapter,
)
from fastagentic.types import StreamEvent, StreamEventType

# ============================================================================
# Semantic Kernel Adapter Tests
# ============================================================================


class TestSemanticKernelAdapter:
    """Tests for SemanticKernelAdapter."""

    def test_create_adapter(self):
        """Test creating adapter."""
        # Mock kernel
        kernel = type("Kernel", (), {})()
        adapter = SemanticKernelAdapter(kernel, function_name="chat")

        assert adapter.kernel is kernel
        assert adapter.function_name == "chat"

    def test_with_function(self):
        """Test creating adapter with different function."""
        kernel = type("Kernel", (), {})()
        adapter = SemanticKernelAdapter(kernel, function_name="chat")

        new_adapter = adapter.with_function("summarize", "text")
        assert new_adapter.function_name == "summarize"
        assert new_adapter.plugin_name == "text"

    def test_build_arguments_dict(self):
        """Test building arguments from dict input."""
        kernel = type("Kernel", (), {})()
        adapter = SemanticKernelAdapter(kernel)

        args = adapter._build_arguments({"message": "hello"})
        assert args == {"message": "hello"}

    def test_build_arguments_string(self):
        """Test building arguments from string input."""
        kernel = type("Kernel", (), {})()
        adapter = SemanticKernelAdapter(kernel)

        args = adapter._build_arguments("hello")
        assert args == {"message": "hello"}


# ============================================================================
# AutoGen Adapter Tests
# ============================================================================


class TestAutoGenAdapter:
    """Tests for AutoGenAdapter."""

    def test_create_adapter(self):
        """Test creating adapter."""
        initiator = type("Agent", (), {"name": "user"})()
        recipient = type("Agent", (), {"name": "assistant"})()

        adapter = AutoGenAdapter(initiator, recipient)

        assert adapter.initiator is initiator
        assert adapter.recipient is recipient
        assert adapter.max_turns is None

    def test_with_max_turns(self):
        """Test creating adapter with max turns."""
        initiator = type("Agent", (), {})()
        recipient = type("Agent", (), {})()

        adapter = AutoGenAdapter(initiator, recipient)
        new_adapter = adapter.with_max_turns(5)

        assert new_adapter.max_turns == 5

    def test_extract_message_string(self):
        """Test extracting message from string."""
        initiator = type("Agent", (), {})()
        recipient = type("Agent", (), {})()
        adapter = AutoGenAdapter(initiator, recipient)

        msg = adapter._extract_message("hello")
        assert msg == "hello"

    def test_extract_message_dict(self):
        """Test extracting message from dict."""
        initiator = type("Agent", (), {})()
        recipient = type("Agent", (), {})()
        adapter = AutoGenAdapter(initiator, recipient)

        msg = adapter._extract_message({"message": "hello"})
        assert msg == "hello"


# ============================================================================
# LlamaIndex Adapter Tests
# ============================================================================


class TestLlamaIndexAdapter:
    """Tests for LlamaIndexAdapter."""

    def test_create_with_agent(self):
        """Test creating adapter with agent."""
        agent = type("Agent", (), {})()
        adapter = LlamaIndexAdapter(agent=agent)

        assert adapter.agent is agent
        assert adapter.query_engine is None

    def test_create_with_query_engine(self):
        """Test creating adapter with query engine."""
        engine = type("QueryEngine", (), {})()
        adapter = LlamaIndexAdapter(query_engine=engine)

        assert adapter.query_engine is engine
        assert adapter.agent is None

    def test_requires_one_component(self):
        """Test that at least one component is required."""
        with pytest.raises(ValueError, match="Must provide at least one"):
            LlamaIndexAdapter()

    def test_extract_query_string(self):
        """Test extracting query from string."""
        agent = type("Agent", (), {})()
        adapter = LlamaIndexAdapter(agent=agent)

        query = adapter._extract_query("what is AI?")
        assert query == "what is AI?"

    def test_extract_query_dict(self):
        """Test extracting query from dict."""
        agent = type("Agent", (), {})()
        adapter = LlamaIndexAdapter(agent=agent)

        query = adapter._extract_query({"query": "what is AI?"})
        assert query == "what is AI?"

    def test_with_query_engine(self):
        """Test creating adapter with different query engine."""
        agent = type("Agent", (), {})()
        adapter = LlamaIndexAdapter(agent=agent)

        engine = type("QueryEngine", (), {})()
        new_adapter = adapter.with_query_engine(engine)

        assert new_adapter.query_engine is engine


# ============================================================================
# DSPy Adapter Tests
# ============================================================================


class TestDSPyAdapter:
    """Tests for DSPyAdapter."""

    def test_create_adapter(self):
        """Test creating adapter."""
        module = type("Module", (), {})()
        adapter = DSPyAdapter(module)

        assert adapter.module is module
        assert adapter.trace is False

    def test_with_trace(self):
        """Test creating adapter with trace."""
        module = type("Module", (), {})()
        adapter = DSPyAdapter(module)

        new_adapter = adapter.with_trace(True)
        assert new_adapter.trace is True

    def test_build_kwargs_dict(self):
        """Test building kwargs from dict."""
        module = type("Module", (), {})()
        adapter = DSPyAdapter(module)

        kwargs = adapter._build_kwargs({"question": "what is AI?"})
        assert kwargs == {"question": "what is AI?"}

    def test_build_kwargs_string(self):
        """Test building kwargs from string."""
        module = type("Module", (), {"signature": None})()
        adapter = DSPyAdapter(module)

        kwargs = adapter._build_kwargs("what is AI?")
        assert "question" in kwargs


class TestDSPyProgramAdapter:
    """Tests for DSPyProgramAdapter."""

    def test_create_adapter(self):
        """Test creating program adapter."""
        program = type("Program", (), {})()
        adapter = DSPyProgramAdapter(program)

        assert adapter.module is program
        assert adapter.include_retrieval is True


# ============================================================================
# Community Adapter SDK Tests
# ============================================================================


class TestAdapterMetadata:
    """Tests for AdapterMetadata."""

    def test_create_metadata(self):
        """Test creating metadata."""
        meta = AdapterMetadata(
            name="my-adapter",
            version="1.0.0",
            description="My adapter",
            author="Test",
            framework="custom",
        )

        assert meta.name == "my-adapter"
        assert meta.version == "1.0.0"

    def test_to_dict(self):
        """Test serialization."""
        meta = AdapterMetadata(
            name="my-adapter",
            version="1.0.0",
            description="My adapter",
            author="Test",
            framework="custom",
            tags=["test"],
        )

        data = meta.to_dict()
        assert data["name"] == "my-adapter"
        assert data["tags"] == ["test"]


class TestSimpleAdapter:
    """Tests for SimpleAdapter."""

    def test_create_simple_adapter(self):
        """Test creating simple adapter."""

        def my_fn(input: str) -> str:
            return f"Response: {input}"

        adapter = SimpleAdapter(invoke_fn=my_fn)
        assert adapter.invoke_fn is my_fn

    @pytest.mark.asyncio
    async def test_invoke_sync_function(self):
        """Test invoking sync function."""

        def my_fn(input: str) -> str:
            return f"Response: {input}"

        adapter = SimpleAdapter(invoke_fn=my_fn)

        # Create mock context using AdapterContext
        from fastagentic.adapters.base import AdapterContext
        from fastagentic.context import AgentContext, RunContext

        run_ctx = RunContext(run_id="test-run", endpoint="/test")
        mock_app = type("App", (), {})()
        agent_ctx = AgentContext(run=run_ctx, app=mock_app)
        ctx = AdapterContext(agent_ctx=agent_ctx)

        result = await adapter.invoke("hello", ctx)
        assert result == "Response: hello"

    @pytest.mark.asyncio
    async def test_invoke_async_function(self):
        """Test invoking async function."""

        async def my_fn(input: str) -> str:
            return f"Response: {input}"

        adapter = SimpleAdapter(invoke_fn=my_fn)

        from fastagentic.adapters.base import AdapterContext
        from fastagentic.context import AgentContext, RunContext

        run_ctx = RunContext(run_id="test-run", endpoint="/test")
        mock_app = type("App", (), {})()
        agent_ctx = AgentContext(run=run_ctx, app=mock_app)
        ctx = AdapterContext(agent_ctx=agent_ctx)

        result = await adapter.invoke("hello", ctx)
        assert result == "Response: hello"


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_create_registry(self):
        """Test creating registry."""
        registry = AdapterRegistry()
        assert len(registry.list_adapters()) == 0

    def test_register_adapter(self):
        """Test registering adapter."""

        class TestAdapter(CommunityAdapter):
            metadata = AdapterMetadata(
                name="test-adapter",
                version="1.0.0",
                description="Test",
                author="Test",
                framework="test",
            )

            async def invoke(self, input, ctx):
                return input

            async def stream(self, input, ctx):
                yield StreamEvent(type=StreamEventType.DONE, data={})

        registry = AdapterRegistry()
        registry.register(TestAdapter)

        adapters = registry.list_adapters()
        assert len(adapters) == 1
        assert adapters[0].name == "test-adapter"

    def test_get_adapter(self):
        """Test getting adapter by name."""

        class TestAdapter(CommunityAdapter):
            metadata = AdapterMetadata(
                name="test-adapter-get",
                version="1.0.0",
                description="Test",
                author="Test",
                framework="test",
            )

            async def invoke(self, input, ctx):
                return input

            async def stream(self, input, ctx):
                yield StreamEvent(type=StreamEventType.DONE, data={})

        registry = AdapterRegistry()
        registry.register(TestAdapter)

        adapter_cls = registry.get("test-adapter-get")
        assert adapter_cls is TestAdapter

    def test_find_by_framework(self):
        """Test finding adapters by framework."""

        class TestAdapter1(CommunityAdapter):
            metadata = AdapterMetadata(
                name="test-1",
                version="1.0.0",
                description="Test",
                author="Test",
                framework="framework-a",
            )

            async def invoke(self, input, ctx):
                return input

            async def stream(self, input, ctx):
                yield StreamEvent(type=StreamEventType.DONE, data={})

        class TestAdapter2(CommunityAdapter):
            metadata = AdapterMetadata(
                name="test-2",
                version="1.0.0",
                description="Test",
                author="Test",
                framework="framework-b",
            )

            async def invoke(self, input, ctx):
                return input

            async def stream(self, input, ctx):
                yield StreamEvent(type=StreamEventType.DONE, data={})

        registry = AdapterRegistry()
        registry.register(TestAdapter1)
        registry.register(TestAdapter2)

        found = registry.find_by_framework("framework-a")
        assert len(found) == 1
        assert found[0] is TestAdapter1
