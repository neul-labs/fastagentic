"""Tests for FastAgentic context classes."""

from unittest.mock import MagicMock

import pytest

from fastagentic.context import AgentContext, RunContext, UsageInfo, UserInfo


class TestUserInfo:
    """Tests for UserInfo."""

    def test_user_info_required_fields(self):
        """Test UserInfo with required fields only."""
        user = UserInfo(id="user-123")
        assert user.id == "user-123"
        assert user.email is None
        assert user.name is None
        assert user.scopes == []
        assert user.metadata == {}

    def test_user_info_all_fields(self):
        """Test UserInfo with all fields."""
        user = UserInfo(
            id="user-123",
            email="user@example.com",
            name="Test User",
            scopes=["read", "write"],
            metadata={"role": "admin"},
        )
        assert user.id == "user-123"
        assert user.email == "user@example.com"
        assert user.name == "Test User"
        assert user.scopes == ["read", "write"]
        assert user.metadata["role"] == "admin"


class TestUsageInfo:
    """Tests for UsageInfo."""

    def test_usage_info_defaults(self):
        """Test UsageInfo default values."""
        usage = UsageInfo()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost == 0.0
        assert usage.model is None

    def test_usage_info_custom_values(self):
        """Test UsageInfo with custom values."""
        usage = UsageInfo(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.003,
            model="gpt-4",
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost == 0.003
        assert usage.model == "gpt-4"

    def test_usage_info_add(self):
        """Test adding UsageInfo objects."""
        usage1 = UsageInfo(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.003,
        )
        usage2 = UsageInfo(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            cost=0.006,
        )

        usage1.add(usage2)

        assert usage1.input_tokens == 300
        assert usage1.output_tokens == 150
        assert usage1.total_tokens == 450
        assert usage1.cost == pytest.approx(0.009)


class TestRunContext:
    """Tests for RunContext."""

    def test_run_context_required_fields(self):
        """Test RunContext with required fields."""
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
        )
        assert ctx.run_id == "run-123"
        assert ctx.endpoint == "/triage"
        assert ctx.user is None
        assert ctx.metadata == {}
        assert ctx.usage is not None

    def test_run_context_with_user(self):
        """Test RunContext with user."""
        user = UserInfo(id="user-123")
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
            user=user,
        )
        assert ctx.user == user
        assert ctx.is_authenticated is True

    def test_run_context_not_authenticated(self):
        """Test is_authenticated when no user."""
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
        )
        assert ctx.is_authenticated is False

    def test_run_context_is_resumed_default(self):
        """Test is_resumed default value."""
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
        )
        assert ctx.is_resumed is False

    def test_run_context_is_resumed_true(self):
        """Test is_resumed when set."""
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
            _is_resumed=True,
        )
        assert ctx.is_resumed is True

    def test_run_context_add_checkpoint(self):
        """Test adding checkpoints."""
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
        )

        state1 = {"step": 1, "data": "first"}
        state2 = {"step": 2, "data": "second"}

        ctx.add_checkpoint(state1)
        ctx.add_checkpoint(state2)

        assert len(ctx._checkpoints) == 2
        assert ctx._checkpoints[0] == state1
        assert ctx._checkpoints[1] == state2

    def test_run_context_metadata(self):
        """Test metadata dictionary."""
        ctx = RunContext(
            run_id="run-123",
            endpoint="/triage",
            metadata={"key": "value"},
        )
        assert ctx.metadata["key"] == "value"

        # Should be mutable
        ctx.metadata["new_key"] = "new_value"
        assert ctx.metadata["new_key"] == "new_value"


class TestAgentContext:
    """Tests for AgentContext."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock App."""
        app = MagicMock()
        app.memory = None
        return app

    @pytest.fixture
    def run_context(self):
        """Create a RunContext for testing."""
        return RunContext(
            run_id="run-123",
            endpoint="/triage",
            user=UserInfo(id="user-123"),
            metadata={"key": "value"},
        )

    def test_agent_context_creation(self, mock_app, run_context):
        """Test AgentContext creation."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.run == run_context
        assert ctx.app == mock_app

    def test_agent_context_run_id_shortcut(self, mock_app, run_context):
        """Test run_id property shortcut."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.run_id == "run-123"

    def test_agent_context_user_shortcut(self, mock_app, run_context):
        """Test user property shortcut."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.user.id == "user-123"

    def test_agent_context_usage_shortcut(self, mock_app, run_context):
        """Test usage property shortcut."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.usage == run_context.usage

    def test_agent_context_metadata_shortcut(self, mock_app, run_context):
        """Test metadata property shortcut."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.metadata["key"] == "value"

    def test_agent_context_memory_from_app(self, run_context):
        """Test memory property returns app memory."""
        mock_app = MagicMock()
        mock_memory = MagicMock()
        mock_app.memory = mock_memory

        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.memory == mock_memory

    def test_agent_context_memory_none_when_no_app(self, run_context):
        """Test memory returns None when no app."""
        ctx = AgentContext(
            run=run_context,
            app=None,
        )
        assert ctx.memory is None

    def test_agent_context_memories_default(self, mock_app, run_context):
        """Test memories returns empty list by default."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )
        assert ctx.memories == []

    def test_agent_context_set_memories(self, mock_app, run_context):
        """Test setting memories."""
        ctx = AgentContext(
            run=run_context,
            app=mock_app,
        )

        memories = [
            {"id": "mem-1", "content": "User likes coffee"},
            {"id": "mem-2", "content": "User is from NYC"},
        ]

        ctx.set_memories(memories)

        assert ctx.memories == memories
        assert len(ctx.memories) == 2

    def test_agent_context_with_request(self, mock_app, run_context):
        """Test AgentContext with request object."""
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer token"}

        ctx = AgentContext(
            run=run_context,
            app=mock_app,
            request=mock_request,
        )

        assert ctx.request == mock_request
        assert ctx.request.headers["Authorization"] == "Bearer token"
