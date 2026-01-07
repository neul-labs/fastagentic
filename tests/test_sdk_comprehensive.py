"""Comprehensive tests for the SDK client."""

import pytest

from fastagentic.sdk import (
    AsyncFastAgenticClient,
    AuthenticationError,
    ClientConfig,
    FastAgenticClient,
    FastAgenticError,
    RateLimitError,
    RunRequest,
    RunResponse,
    RunStatus,
    ServerError,
    StreamEvent,
    StreamEventType,
    ValidationError,
)
from fastagentic.sdk import TimeoutError as SDKTimeoutError


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_config_creation(self):
        config = ClientConfig(base_url="http://localhost:8000")
        assert config.base_url == "http://localhost:8000"

    def test_config_with_api_key(self):
        config = ClientConfig(
            base_url="http://localhost:8000",
            api_key="test-key",
        )
        assert config.api_key == "test-key"

    def test_config_with_custom_timeout(self):
        config = ClientConfig(
            base_url="http://localhost:8000",
            timeout=60.0,
        )
        assert config.timeout == 60.0


class TestRunRequest:
    """Tests for RunRequest model."""

    def test_basic_request(self):
        request = RunRequest(
            endpoint="/chat",
            input={"message": "Hello"},
        )
        assert request.endpoint == "/chat"
        assert request.input == {"message": "Hello"}

    def test_request_with_options(self):
        request = RunRequest(
            endpoint="/chat",
            input={"message": "Hello"},
            stream=True,
            timeout=60.0,
        )
        assert request.stream is True
        assert request.timeout == 60.0


class TestRunResponse:
    """Tests for RunResponse model."""

    def test_basic_response(self):
        response = RunResponse(
            run_id="run-123",
            status=RunStatus.COMPLETED,
            output={"response": "Hello!"},
        )
        assert response.run_id == "run-123"
        assert response.status == RunStatus.COMPLETED
        assert response.output == {"response": "Hello!"}


class TestStreamEvent:
    """Tests for StreamEvent model."""

    def test_token_event(self):
        event = StreamEvent(
            type=StreamEventType.TOKEN,
            data={"content": "Hello"},
        )
        assert event.type == StreamEventType.TOKEN
        assert event.data["content"] == "Hello"

    def test_tool_call_event(self):
        event = StreamEvent(
            type=StreamEventType.TOOL_CALL,
            data={"name": "search", "input": {"query": "test"}},
        )
        assert event.type == StreamEventType.TOOL_CALL

    def test_end_event(self):
        event = StreamEvent(
            type=StreamEventType.END,
            data={"result": "Complete"},
        )
        assert event.type == StreamEventType.END


class TestExceptions:
    """Tests for SDK exceptions."""

    def test_fastagentic_error(self):
        error = FastAgenticError("Something went wrong")
        assert "Something went wrong" in str(error)

    def test_authentication_error(self):
        error = AuthenticationError("Invalid API key")
        assert "Invalid API key" in str(error)

    def test_rate_limit_error(self):
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert error.retry_after == 60

    def test_validation_error(self):
        error = ValidationError("Invalid input", errors=["field1 required"])
        assert error.errors == ["field1 required"]

    def test_timeout_error(self):
        error = SDKTimeoutError("Request timed out")
        assert "timed out" in str(error)

    def test_server_error(self):
        error = ServerError("Internal server error", status_code=500)
        assert error.status_code == 500


class TestAsyncFastAgenticClient:
    """Tests for AsyncFastAgenticClient."""

    @pytest.fixture
    def config(self):
        return ClientConfig(base_url="http://localhost:8000")

    def test_client_creation(self, config):
        client = AsyncFastAgenticClient(config)
        assert client.config is not None


class TestSyncFastAgenticClient:
    """Tests for synchronous FastAgenticClient."""

    @pytest.fixture
    def config(self):
        return ClientConfig(base_url="http://localhost:8000")

    def test_client_creation(self, config):
        client = FastAgenticClient(config)
        assert client.config is not None


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_status_values(self):
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_event_types(self):
        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.TOOL_CALL.value == "tool_call"
        assert StreamEventType.TOOL_RESULT.value == "tool_result"
        assert StreamEventType.END.value == "end"
        assert StreamEventType.ERROR.value == "error"
