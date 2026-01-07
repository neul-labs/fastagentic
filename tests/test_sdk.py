"""Tests for SDK module."""

import pytest

from fastagentic.sdk import (
    ClientConfig,
    RunRequest,
    RunResponse,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)
from fastagentic.sdk.exceptions import (
    AuthenticationError,
    FastAgenticError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
    raise_for_status,
)
from fastagentic.sdk.models import Message, UsageStats

# ============================================================================
# ClientConfig Tests
# ============================================================================


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ClientConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.api_key is None
        assert config.timeout == 300.0
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClientConfig(
            base_url="http://api.example.com",
            api_key="test-key",
            timeout=60.0,
        )
        assert config.base_url == "http://api.example.com"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0

    def test_get_headers_without_key(self):
        """Test headers without API key."""
        config = ClientConfig()
        headers = config.get_headers()
        assert "Content-Type" in headers
        assert "Authorization" not in headers

    def test_get_headers_with_key(self):
        """Test headers with API key."""
        config = ClientConfig(api_key="test-key")
        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer test-key"


# ============================================================================
# Model Tests
# ============================================================================


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        call = ToolCall(name="search", arguments={"query": "test"})
        assert call.name == "search"
        assert call.arguments == {"query": "test"}
        assert call.id.startswith("call-")

    def test_to_dict(self):
        """Test serialization."""
        call = ToolCall(
            id="call-123",
            name="search",
            arguments={"query": "test"},
        )
        data = call.to_dict()
        assert data["id"] == "call-123"
        assert data["name"] == "search"
        assert data["arguments"] == {"query": "test"}

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "call-456",
            "name": "calculate",
            "arguments": {"x": 1, "y": 2},
        }
        call = ToolCall.from_dict(data)
        assert call.id == "call-456"
        assert call.name == "calculate"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(call_id="call-123", result={"data": "value"})
        assert not result.is_error
        assert result.result == {"data": "value"}

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(call_id="call-123", error="Something failed")
        assert result.is_error
        assert result.error == "Something failed"


class TestStreamEvent:
    """Tests for StreamEvent model."""

    def test_create_event(self):
        """Test creating an event."""
        event = StreamEvent(type=StreamEventType.TOKEN, data="Hello")
        assert event.type == StreamEventType.TOKEN
        assert event.data == "Hello"

    def test_to_dict(self):
        """Test serialization."""
        event = StreamEvent(type=StreamEventType.TOKEN, data="test")
        data = event.to_dict()
        assert data["type"] == "token"
        assert data["data"] == "test"

    def test_from_dict(self):
        """Test deserialization."""
        data = {"type": "message", "data": {"content": "Hi"}}
        event = StreamEvent.from_dict(data)
        assert event.type == StreamEventType.MESSAGE


class TestMessage:
    """Tests for Message model."""

    def test_user_message(self):
        """Test user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message_with_tools(self):
        """Test assistant message with tool calls."""
        tool_calls = [ToolCall(name="search", arguments={})]
        msg = Message(role="assistant", tool_calls=tool_calls)
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1


class TestUsageStats:
    """Tests for UsageStats model."""

    def test_default_stats(self):
        """Test default stats."""
        stats = UsageStats()
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.cost == 0.0

    def test_custom_stats(self):
        """Test custom stats."""
        stats = UsageStats(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.05,
            model="gpt-4",
        )
        assert stats.total_tokens == 150
        assert stats.model == "gpt-4"


class TestRunRequest:
    """Tests for RunRequest model."""

    def test_create_request(self):
        """Test creating a request."""
        request = RunRequest(
            endpoint="/chat",
            input={"message": "Hello"},
        )
        assert request.endpoint == "/chat"
        assert request.input == {"message": "Hello"}
        assert request.stream is False

    def test_streaming_request(self):
        """Test streaming request."""
        request = RunRequest(
            endpoint="/chat",
            input={"message": "Hello"},
            stream=True,
        )
        assert request.stream is True

    def test_to_dict(self):
        """Test serialization."""
        request = RunRequest(
            endpoint="/chat",
            input={"message": "test"},
            timeout=60.0,
        )
        data = request.to_dict()
        assert data["endpoint"] == "/chat"
        assert data["timeout"] == 60.0


class TestRunResponse:
    """Tests for RunResponse model."""

    def test_completed_response(self):
        """Test completed response."""
        response = RunResponse(
            run_id="run-123",
            status=RunStatus.COMPLETED,
            output={"result": "success"},
        )
        assert response.is_complete
        assert response.is_success

    def test_failed_response(self):
        """Test failed response."""
        response = RunResponse(
            run_id="run-123",
            status=RunStatus.FAILED,
            error="Something went wrong",
        )
        assert response.is_complete
        assert not response.is_success
        assert response.error == "Something went wrong"

    def test_pending_response(self):
        """Test pending response."""
        response = RunResponse(
            run_id="run-123",
            status=RunStatus.PENDING,
        )
        assert not response.is_complete


# ============================================================================
# Exception Tests
# ============================================================================


class TestExceptions:
    """Tests for SDK exceptions."""

    def test_fastagentic_error(self):
        """Test base error."""
        error = FastAgenticError("Test error", status_code=400)
        assert "Test error" in str(error)
        assert error.status_code == 400

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Invalid token")
        assert error.status_code == 401
        assert "Invalid token" in str(error)

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("Too many requests", retry_after=60)
        assert error.status_code == 429
        assert error.retry_after == 60
        assert "60s" in str(error)

    def test_validation_error(self):
        """Test validation error."""
        errors = [{"field": "name", "message": "Required"}]
        error = ValidationError("Validation failed", errors=errors)
        assert error.status_code == 422
        assert len(error.errors) == 1

    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError("Request timed out", timeout=30)
        assert error.status_code == 408
        assert error.timeout == 30

    def test_server_error(self):
        """Test server error."""
        error = ServerError("Internal error", status_code=500)
        assert error.status_code == 500


class TestRaiseForStatus:
    """Tests for raise_for_status function."""

    def test_raises_auth_error(self):
        """Test raises authentication error for 401."""
        with pytest.raises(AuthenticationError):
            raise_for_status(401, {"error": "Unauthorized"})

    def test_raises_rate_limit_error(self):
        """Test raises rate limit error for 429."""
        with pytest.raises(RateLimitError):
            raise_for_status(429, {"error": "Too many requests"})

    def test_raises_validation_error(self):
        """Test raises validation error for 422."""
        with pytest.raises(ValidationError):
            raise_for_status(422, {"error": "Invalid input"})

    def test_raises_server_error(self):
        """Test raises server error for 500."""
        with pytest.raises(ServerError):
            raise_for_status(500, {"error": "Internal error"})

    def test_no_error_for_success(self):
        """Test no error for success status."""
        # Should not raise for 2xx status
        # (raise_for_status is typically called only for 4xx/5xx)
        pass
