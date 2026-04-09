"""Tests for FastAgentic integrations."""

from unittest.mock import patch

import pytest

from fastagentic.hooks.base import HookContext, HookResultAction
from fastagentic.integrations.base import IntegrationConfig


class TestIntegrationBase:
    """Tests for base Integration class."""

    def test_integration_config_defaults(self):
        """Test default configuration values."""
        config = IntegrationConfig()
        assert config.enabled is True
        assert config.api_key is None
        assert config.base_url is None
        assert config.extra == {}

    def test_integration_config_custom(self):
        """Test custom configuration."""
        config = IntegrationConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com",
            extra={"custom": "value"},
        )
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"
        assert config.extra["custom"] == "value"


class TestLangfuseIntegration:
    """Tests for Langfuse integration."""

    def test_langfuse_config(self):
        """Test Langfuse configuration."""
        from fastagentic.integrations.langfuse import LangfuseConfig

        config = LangfuseConfig(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://custom.langfuse.com",
            release="v1.0.0",
        )
        assert config.public_key == "pk-test"
        assert config.secret_key == "sk-test"
        assert config.host == "https://custom.langfuse.com"

    def test_langfuse_integration_name(self):
        """Test integration name property."""
        from fastagentic.integrations.langfuse import LangfuseIntegration

        integration = LangfuseIntegration()
        assert integration.name == "langfuse"

    def test_langfuse_validate_config_missing_keys(self):
        """Test validation fails without keys."""
        from fastagentic.integrations.langfuse import LangfuseIntegration

        integration = LangfuseIntegration()

        with patch.dict("os.environ", {}, clear=True):
            errors = integration.validate_config()
            # Should have errors about missing keys
            assert any("public_key" in e for e in errors) or not integration.is_available()

    def test_langfuse_get_hooks(self):
        """Test hooks are returned."""
        from fastagentic.integrations.langfuse import LangfuseIntegration

        integration = LangfuseIntegration(public_key="pk", secret_key="sk")
        hooks = integration.get_hooks()
        assert len(hooks) == 1


class TestPortkeyIntegration:
    """Tests for Portkey integration."""

    def test_portkey_config(self):
        """Test Portkey configuration."""
        from fastagentic.integrations.portkey import PortkeyConfig

        config = PortkeyConfig(
            api_key="pk-test",
            virtual_key="vk-test",
            retry_attempts=5,
        )
        assert config.api_key == "pk-test"
        assert config.virtual_key == "vk-test"
        assert config.retry_attempts == 5

    def test_portkey_integration_name(self):
        """Test integration name property."""
        from fastagentic.integrations.portkey import PortkeyIntegration

        integration = PortkeyIntegration()
        assert integration.name == "portkey"

    def test_portkey_get_hooks(self):
        """Test hooks are returned."""
        from fastagentic.integrations.portkey import PortkeyIntegration

        integration = PortkeyIntegration(api_key="pk")
        hooks = integration.get_hooks()
        assert len(hooks) == 1


class TestLakeraIntegration:
    """Tests for Lakera integration."""

    def test_lakera_config(self):
        """Test Lakera configuration."""
        from fastagentic.integrations.lakera import LakeraConfig

        config = LakeraConfig(
            api_key="lak-test",
            block_on_detect=False,
            confidence_threshold=0.9,
            categories=["prompt_injection"],
        )
        assert config.api_key == "lak-test"
        assert config.block_on_detect is False
        assert config.confidence_threshold == 0.9
        assert config.categories == ["prompt_injection"]

    def test_lakera_integration_name(self):
        """Test integration name property."""
        from fastagentic.integrations.lakera import LakeraIntegration

        integration = LakeraIntegration()
        assert integration.name == "lakera"

    def test_lakera_always_available(self):
        """Test Lakera is always available (HTTP-based)."""
        from fastagentic.integrations.lakera import LakeraIntegration

        integration = LakeraIntegration()
        assert integration.is_available() is True

    def test_lakera_get_hooks(self):
        """Test hooks are returned."""
        from fastagentic.integrations.lakera import LakeraIntegration

        integration = LakeraIntegration(api_key="lak")
        hooks = integration.get_hooks()
        assert len(hooks) == 1

    @pytest.mark.asyncio
    async def test_lakera_hook_format_messages(self):
        """Test message formatting for scanning."""
        from fastagentic.integrations.lakera import LakeraConfig, LakeraHook

        config = LakeraConfig(api_key="test")
        hook = LakeraHook(config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        formatted = hook._format_messages_for_scan(messages)
        assert "system: You are a helpful assistant." in formatted
        assert "user: Hello!" in formatted
        assert "assistant: Hi there!" in formatted


class TestMem0Integration:
    """Tests for Mem0 integration."""

    def test_mem0_config(self):
        """Test Mem0 configuration."""
        from fastagentic.integrations.mem0 import Mem0Config

        config = Mem0Config(
            api_key="m0-test",
            auto_add=False,
            search_limit=10,
        )
        assert config.api_key == "m0-test"
        assert config.auto_add is False
        assert config.search_limit == 10

    def test_mem0_integration_name(self):
        """Test integration name property."""
        from fastagentic.integrations.mem0 import Mem0Integration

        integration = Mem0Integration()
        assert integration.name == "mem0"

    def test_mem0_get_hooks(self):
        """Test hooks are returned."""
        from fastagentic.integrations.mem0 import Mem0Integration

        integration = Mem0Integration(api_key="m0")
        hooks = integration.get_hooks()
        assert len(hooks) == 1


class TestLakeraHookBehavior:
    """Tests for Lakera hook behavior."""

    @pytest.fixture
    def hook(self):
        """Create a Lakera hook for testing."""
        from fastagentic.integrations.lakera import LakeraConfig, LakeraHook

        config = LakeraConfig(
            api_key="test-key",
            block_on_detect=True,
        )
        return LakeraHook(config)

    @pytest.fixture
    def hook_context(self):
        """Create a hook context for testing."""
        return HookContext(
            run_id="test-run",
            endpoint="/test",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

    @pytest.mark.asyncio
    async def test_hook_proceeds_when_not_flagged(self, hook, hook_context):
        """Test hook proceeds when content is not flagged."""

        # Mock the scan to return not flagged
        async def mock_scan(*args, **kwargs):
            return {"flagged": False, "categories": {}}

        hook._scan_content = mock_scan

        result = await hook.on_request(hook_context)
        assert result.action == HookResultAction.PROCEED

    @pytest.mark.asyncio
    async def test_hook_rejects_when_flagged_and_blocking(self, hook, hook_context):
        """Test hook rejects when content is flagged and blocking enabled."""

        # Mock the scan to return flagged
        async def mock_scan(*args, **kwargs):
            return {
                "flagged": True,
                "categories": {
                    "prompt_injection": True,
                },
            }

        hook._scan_content = mock_scan

        result = await hook.on_request(hook_context)
        assert result.action == HookResultAction.REJECT
        assert "prompt_injection" in result.message

    @pytest.mark.asyncio
    async def test_hook_proceeds_when_flagged_but_not_blocking(self, hook_context):
        """Test hook proceeds when flagged but blocking disabled."""
        from fastagentic.integrations.lakera import LakeraConfig, LakeraHook

        config = LakeraConfig(
            api_key="test-key",
            block_on_detect=False,  # Don't block
        )
        hook = LakeraHook(config)

        # Mock the scan to return flagged
        async def mock_scan(*args, **kwargs):
            return {
                "flagged": True,
                "categories": {
                    "prompt_injection": True,
                },
            }

        hook._scan_content = mock_scan

        result = await hook.on_request(hook_context)
        assert result.action == HookResultAction.PROCEED

        # But metadata should be set
        assert hook_context.metadata.get("lakera_input_flagged") is True


class TestIntegrationLazyLoading:
    """Test lazy loading of integrations."""

    def test_lazy_load_langfuse(self):
        """Test lazy loading of LangfuseIntegration."""
        from fastagentic.integrations import LangfuseIntegration

        assert LangfuseIntegration is not None
        assert LangfuseIntegration().name == "langfuse"

    def test_lazy_load_portkey(self):
        """Test lazy loading of PortkeyIntegration."""
        from fastagentic.integrations import PortkeyIntegration

        assert PortkeyIntegration is not None
        assert PortkeyIntegration().name == "portkey"

    def test_lazy_load_lakera(self):
        """Test lazy loading of LakeraIntegration."""
        from fastagentic.integrations import LakeraIntegration

        assert LakeraIntegration is not None
        assert LakeraIntegration().name == "lakera"

    def test_lazy_load_mem0(self):
        """Test lazy loading of Mem0Integration."""
        from fastagentic.integrations import Mem0Integration

        assert Mem0Integration is not None
        assert Mem0Integration().name == "mem0"

    def test_invalid_attribute_raises(self):
        """Test invalid attribute raises AttributeError."""
        import fastagentic.integrations as integrations

        with pytest.raises(AttributeError):
            _ = integrations.NonExistentIntegration
