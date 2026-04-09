"""Tests for FastAgentic hooks module."""

from unittest.mock import MagicMock

import pytest

from fastagentic.context import UsageInfo, UserInfo
from fastagentic.hooks.base import (
    Hook,
    HookContext,
    HookResult,
    HookResultAction,
    _hook_registry,
    get_registered_hooks,
    hook,
)


class TestHookResultAction:
    """Tests for HookResultAction enum."""

    def test_proceed_value(self):
        """Test PROCEED action value."""
        assert HookResultAction.PROCEED.value == "proceed"

    def test_modify_value(self):
        """Test MODIFY action value."""
        assert HookResultAction.MODIFY.value == "modify"

    def test_skip_value(self):
        """Test SKIP action value."""
        assert HookResultAction.SKIP.value == "skip"

    def test_reject_value(self):
        """Test REJECT action value."""
        assert HookResultAction.REJECT.value == "reject"

    def test_retry_value(self):
        """Test RETRY action value."""
        assert HookResultAction.RETRY.value == "retry"


class TestHookResult:
    """Tests for HookResult class."""

    def test_default_result(self):
        """Test default HookResult values."""
        result = HookResult()
        assert result.action == HookResultAction.PROCEED
        assert result.data is None
        assert result.message is None

    def test_proceed_factory(self):
        """Test proceed() factory method."""
        result = HookResult.proceed()
        assert result.action == HookResultAction.PROCEED

    def test_modify_factory(self):
        """Test modify() factory method."""
        data = {"modified": True}
        result = HookResult.modify(data)
        assert result.action == HookResultAction.MODIFY
        assert result.data == data

    def test_skip_factory(self):
        """Test skip() factory method."""
        result = HookResult.skip("Skipping for reason")
        assert result.action == HookResultAction.SKIP
        assert result.message == "Skipping for reason"

    def test_skip_factory_no_message(self):
        """Test skip() factory without message."""
        result = HookResult.skip()
        assert result.action == HookResultAction.SKIP
        assert result.message is None

    def test_reject_factory(self):
        """Test reject() factory method."""
        result = HookResult.reject("Access denied")
        assert result.action == HookResultAction.REJECT
        assert result.message == "Access denied"

    def test_retry_factory(self):
        """Test retry() factory method."""
        result = HookResult.retry(max_retries=5)
        assert result.action == HookResultAction.RETRY
        assert result.max_retries == 5

    def test_retry_factory_default(self):
        """Test retry() factory with default max_retries."""
        result = HookResult.retry()
        assert result.action == HookResultAction.RETRY
        assert result.max_retries == 3


class TestHookContext:
    """Tests for HookContext class."""

    def test_required_fields(self):
        """Test HookContext with required fields."""
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
        )
        assert ctx.run_id == "run-123"
        assert ctx.endpoint == "/test"

    def test_optional_fields_defaults(self):
        """Test HookContext optional field defaults."""
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
        )
        assert ctx.request is None
        assert ctx.response is None
        assert ctx.messages == []
        assert ctx.user is None
        assert ctx.usage is None
        assert ctx.tool_name is None
        assert ctx.error is None
        assert ctx.metadata == {}

    def test_with_user(self):
        """Test HookContext with user info."""
        user = UserInfo(id="user-123", email="user@example.com")
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            user=user,
        )
        assert ctx.user == user
        assert ctx.user.id == "user-123"

    def test_with_usage(self):
        """Test HookContext with usage info."""
        usage = UsageInfo(input_tokens=100, output_tokens=50)
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            usage=usage,
        )
        assert ctx.usage == usage
        assert ctx.usage.input_tokens == 100

    def test_with_tool_info(self):
        """Test HookContext with tool info."""
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            tool_name="search",
            tool_input={"query": "test"},
            tool_output={"results": []},
        )
        assert ctx.tool_name == "search"
        assert ctx.tool_input["query"] == "test"
        assert ctx.tool_output["results"] == []

    def test_with_node_info(self):
        """Test HookContext with node info (LangGraph)."""
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            node_name="triage",
            node_input={"ticket": "help"},
            node_output={"priority": "high"},
        )
        assert ctx.node_name == "triage"
        assert ctx.node_input["ticket"] == "help"
        assert ctx.node_output["priority"] == "high"

    def test_with_error(self):
        """Test HookContext with error."""
        error = ValueError("Something went wrong")
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            error=error,
            retry_count=2,
        )
        assert ctx.error == error
        assert ctx.retry_count == 2

    def test_with_memory_info(self):
        """Test HookContext with memory info."""
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            memory_content="User prefers dark mode",
            memory_query="preferences",
            memory_results=[{"id": "mem-1", "content": "..."}],
        )
        assert ctx.memory_content == "User prefers dark mode"
        assert ctx.memory_query == "preferences"
        assert len(ctx.memory_results) == 1

    def test_metadata_mutable(self):
        """Test that metadata can be modified."""
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
        )
        ctx.metadata["custom"] = "value"
        assert ctx.metadata["custom"] == "value"


class TestHook:
    """Tests for Hook base class."""

    @pytest.mark.asyncio
    async def test_hook_default_methods(self):
        """Test Hook default method implementations."""
        hook_instance = Hook()
        ctx = HookContext(run_id="run-123", endpoint="/test")

        # All default methods should return proceed
        result = await hook_instance.on_request(ctx)
        assert result.action == HookResultAction.PROCEED

        result = await hook_instance.on_response(ctx)
        assert result.action == HookResultAction.PROCEED

        result = await hook_instance.on_llm_start(ctx)
        assert result.action == HookResultAction.PROCEED

        result = await hook_instance.on_llm_end(ctx)
        assert result.action == HookResultAction.PROCEED

        result = await hook_instance.on_tool_call(ctx)
        assert result.action == HookResultAction.PROCEED

        result = await hook_instance.on_tool_result(ctx)
        assert result.action == HookResultAction.PROCEED

        result = await hook_instance.on_error(ctx)
        assert result.action == HookResultAction.PROCEED

    @pytest.mark.asyncio
    async def test_hook_startup_shutdown(self):
        """Test Hook startup/shutdown methods."""
        hook_instance = Hook()
        mock_app = MagicMock()

        # Should not raise
        await hook_instance.on_startup(mock_app)
        await hook_instance.on_shutdown(mock_app)


class TestCustomHook:
    """Tests for custom hook implementations."""

    @pytest.mark.asyncio
    async def test_custom_on_request(self):
        """Test custom on_request implementation."""

        class LoggingHook(Hook):
            def __init__(self):
                self.logged_requests = []

            async def on_request(self, ctx: HookContext) -> HookResult:
                self.logged_requests.append(ctx.run_id)
                return HookResult.proceed()

        hook_instance = LoggingHook()
        ctx = HookContext(run_id="run-123", endpoint="/test")

        result = await hook_instance.on_request(ctx)

        assert result.action == HookResultAction.PROCEED
        assert "run-123" in hook_instance.logged_requests

    @pytest.mark.asyncio
    async def test_custom_reject_hook(self):
        """Test hook that rejects requests."""

        class AuthHook(Hook):
            async def on_request(self, ctx: HookContext) -> HookResult:
                if ctx.user is None:
                    return HookResult.reject("Authentication required")
                return HookResult.proceed()

        hook_instance = AuthHook()

        # Without user - should reject
        ctx = HookContext(run_id="run-123", endpoint="/test")
        result = await hook_instance.on_request(ctx)
        assert result.action == HookResultAction.REJECT
        assert result.message == "Authentication required"

        # With user - should proceed
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            user=UserInfo(id="user-123"),
        )
        result = await hook_instance.on_request(ctx)
        assert result.action == HookResultAction.PROCEED

    @pytest.mark.asyncio
    async def test_custom_modify_hook(self):
        """Test hook that modifies data."""

        class ModifyHook(Hook):
            async def on_llm_start(self, ctx: HookContext) -> HookResult:
                # Add system message
                modified_messages = [
                    {"role": "system", "content": "Be helpful."},
                    *ctx.messages,
                ]
                return HookResult.modify(modified_messages)

        hook_instance = ModifyHook()
        ctx = HookContext(
            run_id="run-123",
            endpoint="/test",
            messages=[{"role": "user", "content": "Hello"}],
        )

        result = await hook_instance.on_llm_start(ctx)

        assert result.action == HookResultAction.MODIFY
        assert len(result.data) == 2
        assert result.data[0]["role"] == "system"


class TestHookDecorator:
    """Tests for @hook decorator."""

    def test_hook_decorator_registers(self):
        """Test that @hook decorator registers functions."""
        # Clear registry for test
        _hook_registry.clear()

        @hook("on_custom_event")
        async def my_hook(ctx: HookContext) -> HookResult:
            return HookResult.proceed()

        registered = get_registered_hooks("on_custom_event")
        assert len(registered) == 1
        assert registered[0] == my_hook

    def test_hook_decorator_multiple(self):
        """Test registering multiple hooks for same event."""
        _hook_registry.clear()

        @hook("on_another_event")
        async def hook1(ctx: HookContext) -> HookResult:
            return HookResult.proceed()

        @hook("on_another_event")
        async def hook2(ctx: HookContext) -> HookResult:
            return HookResult.proceed()

        registered = get_registered_hooks("on_another_event")
        assert len(registered) == 2

    def test_get_registered_hooks_empty(self):
        """Test getting hooks for unregistered event."""
        registered = get_registered_hooks("non_existent_event")
        assert registered == []
