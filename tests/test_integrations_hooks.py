"""Tests for integration hooks behavior."""

import pytest

from fastagentic.hooks.base import HookContext, HookResult, HookResultAction


class TestHookContext:
    """Tests for HookContext."""

    def test_hook_context_creation(self):
        """Test creating a hook context."""
        context = HookContext(
            run_id="run-123",
            endpoint="/test",
            messages=[{"role": "user", "content": "test"}],
        )
        assert context.run_id == "run-123"
        assert context.endpoint == "/test"
        assert len(context.messages) == 1

    def test_hook_context_with_messages(self):
        """Test hook context with messages list."""
        context = HookContext(
            run_id="run-456",
            endpoint="/test",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )
        assert len(context.messages) == 2

    def test_hook_context_with_metadata(self):
        """Test hook context with request/response."""
        context = HookContext(
            run_id="run-789",
            endpoint="/test",
            request={"input": "test"},
            response={"output": "result"},
        )
        assert context.request == {"input": "test"}
        assert context.response == {"output": "result"}


class TestHookResult:
    """Tests for HookResult."""

    def test_hook_result_default(self):
        """Test default HookResult values."""
        result = HookResult()
        assert result.action == HookResultAction.PROCEED
        assert result.data is None
        assert result.message is None
        assert result.retry_count == 0
        assert result.max_retries == 3

    def test_hook_result_proceed_classmethod(self):
        """Test HookResult.proceed() classmethod."""
        result = HookResult.proceed()
        assert result.action == HookResultAction.PROCEED

    def test_hook_result_modify_classmethod(self):
        """Test HookResult.modify() classmethod."""
        result = HookResult.modify({"modified": True})
        assert result.action == HookResultAction.MODIFY
        assert result.data == {"modified": True}

    def test_hook_result_skip_classmethod(self):
        """Test HookResult.skip() classmethod."""
        result = HookResult.skip(message="Skipping operation")
        assert result.action == HookResultAction.SKIP
        assert result.message == "Skipping operation"

    def test_hook_result_reject_classmethod(self):
        """Test HookResult.reject() classmethod."""
        result = HookResult.reject(message="Request rejected")
        assert result.action == HookResultAction.REJECT
        assert result.message == "Request rejected"

    def test_hook_result_retry_classmethod(self):
        """Test HookResult.retry() classmethod."""
        result = HookResult.retry(max_retries=5)
        assert result.action == HookResultAction.RETRY
        assert result.max_retries == 5


class TestHookResultAction:
    """Tests for HookResultAction enum."""

    def test_action_values(self):
        """Test HookResultAction enum values."""
        assert HookResultAction.PROCEED.value == "proceed"
        assert HookResultAction.MODIFY.value == "modify"
        assert HookResultAction.SKIP.value == "skip"
        assert HookResultAction.REJECT.value == "reject"
        assert HookResultAction.RETRY.value == "retry"

    def test_action_count(self):
        """Test there are 5 action types."""
        actions = list(HookResultAction)
        assert len(actions) == 5


class TestHookBehavior:
    """Tests for hook behavior patterns."""

    def test_hook_context_run_id_required(self):
        """Test that run_id is required in HookContext."""
        with pytest.raises(TypeError):
            HookContext()

    def test_hook_context_endpoint_required(self):
        """Test that endpoint is required in HookContext."""
        with pytest.raises(TypeError):
            HookContext(run_id="test")

    def test_hook_context_optional_fields(self):
        """Test HookContext optional fields."""
        context = HookContext(
            run_id="run-123",
            endpoint="/test",
        )
        assert context.messages == []
        assert context.request is None
        assert context.response is None

    def test_hook_result_action_is_string_enum(self):
        """Test HookResultAction is a string enum."""
        assert isinstance(HookResultAction.PROCEED, str)
        assert isinstance(HookResultAction.MODIFY, str)
