"""Tests for the Agent CLI."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastagentic.cli.agent import (
    AgentClient,
    AgentConfig,
    AgentREPL,
    Conversation,
    Message,
    run_single_query,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self) -> None:
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_message_to_dict(self) -> None:
        """Test message serialization."""
        msg = Message(role="assistant", content="Hi there", metadata={"tokens": 10})
        data = msg.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"
        assert "timestamp" in data
        assert data["metadata"] == {"tokens": 10}

    def test_message_from_dict(self) -> None:
        """Test message deserialization."""
        data = {
            "role": "user",
            "content": "Test",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {},
        }
        msg = Message.from_dict(data)

        assert msg.role == "user"
        assert msg.content == "Test"
        assert msg.timestamp.year == 2024


class TestConversation:
    """Tests for Conversation class."""

    def test_create_conversation(self) -> None:
        """Test creating a conversation."""
        conv = Conversation(id="test-conv")
        assert conv.id == "test-conv"
        assert conv.messages == []
        assert isinstance(conv.created_at, datetime)

    def test_add_message(self) -> None:
        """Test adding messages."""
        conv = Conversation(id="test")
        msg = conv.add_message("user", "Hello", custom="value")

        assert len(conv.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.metadata == {"custom": "value"}

    def test_get_history(self) -> None:
        """Test getting conversation history."""
        conv = Conversation(id="test")
        conv.add_message("user", "Q1")
        conv.add_message("assistant", "A1")
        conv.add_message("tool", "Tool output")  # Should be excluded
        conv.add_message("user", "Q2")

        history = conv.get_history()
        assert len(history) == 3
        assert history[0] == {"role": "user", "content": "Q1"}
        assert history[1] == {"role": "assistant", "content": "A1"}
        assert history[2] == {"role": "user", "content": "Q2"}

    def test_save_and_load(self) -> None:
        """Test saving and loading conversations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conv.json"

            # Create and save
            conv = Conversation(id="test-save")
            conv.add_message("user", "Hello")
            conv.add_message("assistant", "Hi!")
            conv.save(path)

            # Load and verify
            loaded = Conversation.load(path)
            assert loaded.id == "test-save"
            assert len(loaded.messages) == 2
            assert loaded.messages[0].content == "Hello"
            assert loaded.messages[1].content == "Hi!"


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = AgentConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.endpoint == "/chat"
        assert config.api_key is None
        assert config.stream is True
        assert config.timeout == 300.0

    def test_config_from_environment(self) -> None:
        """Test loading config from environment."""
        with patch.dict(
            "os.environ",
            {
                "FASTAGENTIC_URL": "http://custom:9000",
                "FASTAGENTIC_API_KEY": "test-key",
                "FASTAGENTIC_ENDPOINT": "/custom",
            },
        ):
            config = AgentConfig.load()
            assert config.base_url == "http://custom:9000"
            assert config.api_key == "test-key"
            assert config.endpoint == "/custom"

    def test_config_save_and_load(self) -> None:
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            # Create and save
            config = AgentConfig()
            config.base_url = "http://test:8080"
            config.endpoint = "/agent"
            config.api_key = "secret"
            config.save(path)

            # Verify file
            data = json.loads(path.read_text())
            assert data["base_url"] == "http://test:8080"
            assert data["endpoint"] == "/agent"


class TestAgentClient:
    """Tests for AgentClient class."""

    @pytest.mark.asyncio
    async def test_client_context_manager(self) -> None:
        """Test client as async context manager."""
        config = AgentConfig()

        async with AgentClient(config) as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_get_headers_without_api_key(self) -> None:
        """Test headers without API key."""
        config = AgentConfig(api_key=None)
        client = AgentClient(config)

        headers = client._get_headers()
        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_get_headers_with_api_key(self) -> None:
        """Test headers with API key."""
        config = AgentConfig(api_key="test-key")
        client = AgentClient(config)

        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_invoke(self) -> None:
        """Test invoke method."""
        config = AgentConfig()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Hello!"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with AgentClient(config) as client:
                result = await client.invoke("Hi")

            assert result == {"response": "Hello!"}

    @pytest.mark.asyncio
    async def test_health_check_healthy(self) -> None:
        """Test health check when healthy."""
        config = AgentConfig()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with AgentClient(config) as client:
                healthy = await client.health_check()

            assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self) -> None:
        """Test health check when unhealthy."""
        config = AgentConfig()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with AgentClient(config) as client:
                healthy = await client.health_check()

            assert healthy is False

    @pytest.mark.asyncio
    async def test_list_endpoints(self) -> None:
        """Test listing endpoints."""
        config = AgentConfig()

        openapi_spec = {
            "paths": {
                "/chat": {
                    "post": {"summary": "Chat endpoint"},
                },
                "/analyze": {
                    "post": {"summary": "Analysis endpoint"},
                    "get": {"summary": "Get analysis"},
                },
            }
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = openapi_spec
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with AgentClient(config) as client:
                endpoints = await client.list_endpoints()

            assert len(endpoints) == 2
            assert any(ep["path"] == "/chat" for ep in endpoints)
            assert any(ep["path"] == "/analyze" for ep in endpoints)


class TestAgentREPL:
    """Tests for AgentREPL class."""

    def test_repl_creation(self) -> None:
        """Test REPL creation."""
        config = AgentConfig()
        repl = AgentREPL(config)

        assert repl.config == config
        assert repl.conversation is None
        assert repl.client is None

    @pytest.mark.asyncio
    async def test_handle_quit_command(self) -> None:
        """Test /quit command."""
        config = AgentConfig()
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        result = await repl._handle_command("/quit")
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_help_command(self) -> None:
        """Test /help command."""
        config = AgentConfig()
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        # Just verify it doesn't crash
        result = await repl._handle_command("/help")
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_clear_command(self) -> None:
        """Test /clear command."""
        config = AgentConfig()
        repl = AgentREPL(config)
        repl.conversation = Conversation(id="old")
        repl.conversation.add_message("user", "old message")
        repl.client = AsyncMock()

        result = await repl._handle_command("/clear")
        assert result is True
        assert repl.conversation is not None
        assert len(repl.conversation.messages) == 0

    @pytest.mark.asyncio
    async def test_handle_use_command(self) -> None:
        """Test /use command."""
        config = AgentConfig(endpoint="/old")
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        await repl._handle_command("/use /new-endpoint")
        assert config.endpoint == "/new-endpoint"

    @pytest.mark.asyncio
    async def test_handle_stream_command(self) -> None:
        """Test /stream command."""
        config = AgentConfig(stream=True)
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        await repl._handle_command("/stream off")
        assert config.stream is False

        await repl._handle_command("/stream on")
        assert config.stream is True

    @pytest.mark.asyncio
    async def test_handle_format_command(self) -> None:
        """Test /format command."""
        config = AgentConfig(output_format="markdown")
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        await repl._handle_command("/format plain")
        assert config.output_format == "plain"

        await repl._handle_command("/format json")
        assert config.output_format == "json"

    @pytest.mark.asyncio
    async def test_handle_compact_command(self) -> None:
        """Test /compact command."""
        config = AgentConfig(show_tools=True, show_thinking=True, show_usage=True)
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        await repl._handle_command("/compact")
        assert config.show_tools is False
        assert config.show_thinking is False
        assert config.show_usage is False

    @pytest.mark.asyncio
    async def test_handle_verbose_command(self) -> None:
        """Test /verbose command."""
        config = AgentConfig(show_tools=False, show_thinking=False, show_usage=False)
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        await repl._handle_command("/verbose")
        assert config.show_tools is True
        assert config.show_thinking is True
        assert config.show_usage is True

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self) -> None:
        """Test unknown command."""
        config = AgentConfig()
        repl = AgentREPL(config)
        repl.client = AsyncMock()

        result = await repl._handle_command("/unknown")
        assert result is True  # Continues REPL


class TestRunSingleQuery:
    """Tests for run_single_query function."""

    @pytest.mark.asyncio
    async def test_run_single_query_non_streaming(self) -> None:
        """Test non-streaming query."""
        config = AgentConfig(stream=False, output_format="plain")

        with patch("fastagentic.cli.agent.AgentClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.invoke = AsyncMock(return_value={"response": "Test response"})
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            # Run without output file
            await run_single_query("Hello", config)

            mock_client.invoke.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_run_single_query_with_output_file(self) -> None:
        """Test query with output file."""
        config = AgentConfig(stream=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"

            with patch("fastagentic.cli.agent.AgentClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.invoke = AsyncMock(return_value={"response": "Saved response"})
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                await run_single_query("Save this", config, output_path)

            assert output_path.exists()
            assert output_path.read_text() == "Saved response"


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_agent_chat_command_exists(self) -> None:
        """Test that agent chat command is registered."""
        from fastagentic.cli.main import agent_app

        # Find the chat command
        commands = {cmd.name: cmd for cmd in agent_app.registered_commands}
        assert "chat" in commands

    def test_agent_query_command_exists(self) -> None:
        """Test that agent query command is registered."""
        from fastagentic.cli.main import agent_app

        commands = {cmd.name: cmd for cmd in agent_app.registered_commands}
        assert "query" in commands

    def test_agent_config_command_exists(self) -> None:
        """Test that agent config command is registered."""
        from fastagentic.cli.main import agent_app

        commands = {cmd.name: cmd for cmd in agent_app.registered_commands}
        assert "config" in commands

    def test_agent_history_command_exists(self) -> None:
        """Test that agent history command is registered."""
        from fastagentic.cli.main import agent_app

        commands = {cmd.name: cmd for cmd in agent_app.registered_commands}
        assert "history" in commands
