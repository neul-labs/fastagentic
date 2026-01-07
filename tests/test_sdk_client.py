"""Tests for SDK client."""

from fastagentic.sdk.client import AsyncFastAgenticClient, ClientConfig, FastAgenticClient


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClientConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.api_key is None
        assert config.timeout == 300.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.verify_ssl is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClientConfig(
            base_url="https://api.example.com",
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
            verify_ssl=False,
        )
        assert config.base_url == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.verify_ssl is False

    def test_get_headers_without_api_key(self):
        """Test getting headers without API key."""
        config = ClientConfig(base_url="http://localhost:8000")
        headers = config.get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_get_headers_with_api_key(self):
        """Test getting headers with API key."""
        config = ClientConfig(base_url="http://localhost:8000", api_key="secret-key")
        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer secret-key"

    def test_get_headers_with_custom_headers(self):
        """Test getting headers with custom headers."""
        config = ClientConfig(base_url="http://localhost:8000", headers={"X-Custom": "value"})
        headers = config.get_headers()
        assert headers["X-Custom"] == "value"

    def test_headers_merge_api_key_and_custom(self):
        """Test that API key Authorization header is added to custom headers."""
        config = ClientConfig(
            base_url="http://localhost:8000", api_key="my-key", headers={"X-Custom": "value"}
        )
        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer my-key"
        assert headers["X-Custom"] == "value"


class TestAsyncFastAgenticClient:
    """Tests for AsyncFastAgenticClient."""

    def test_client_initialization(self):
        """Test client initialization without httpx being imported."""
        client = AsyncFastAgenticClient.__new__(AsyncFastAgenticClient)
        client.config = ClientConfig(base_url="http://test:8000")
        client._client = None
        assert client.config.base_url == "http://test:8000"
        assert client._client is None

    def test_client_with_custom_config(self):
        """Test client with custom config object."""
        config = ClientConfig(
            base_url="https://custom.example.com",
            api_key="test-key",
            timeout=60.0,
        )
        client = AsyncFastAgenticClient(config=config)
        assert client.config.base_url == "https://custom.example.com"
        assert client.config.api_key == "test-key"
        assert client.config.timeout == 60.0

    def test_client_with_kwargs(self):
        """Test client with keyword arguments."""
        client = AsyncFastAgenticClient(
            base_url="https://kwargs.example.com",
            api_key="kwargs-key",
            timeout=120.0,
            max_retries=5,
        )
        assert client.config.base_url == "https://kwargs.example.com"
        assert client.config.api_key == "kwargs-key"
        assert client.config.timeout == 120.0
        assert client.config.max_retries == 5


class TestFastAgenticClient:
    """Tests for synchronous FastAgenticClient."""

    def test_sync_client_initialization(self):
        """Test sync client initialization."""
        client = FastAgenticClient.__new__(FastAgenticClient)
        client.config = ClientConfig(base_url="http://localhost:8000")
        client._client = None
        assert client.config.base_url == "http://localhost:8000"
        assert client._client is None

    def test_sync_client_with_config(self):
        """Test sync client with config object."""
        config = ClientConfig(
            base_url="https://sync.example.com",
            api_key="sync-key",
        )
        client = FastAgenticClient(config=config)
        assert client.config.base_url == "https://sync.example.com"
        assert client.config.api_key == "sync-key"

    def test_sync_client_attributes(self):
        """Test sync client has expected attributes."""
        client = FastAgenticClient(base_url="http://localhost:8000")
        assert hasattr(client, "config")
        assert hasattr(client, "_client")
        assert hasattr(client, "run")
        assert hasattr(client, "stream")
        assert hasattr(client, "close")
        assert hasattr(client, "list_tools")
        assert hasattr(client, "invoke_tool")
        assert hasattr(client, "health")
        assert hasattr(client, "info")
