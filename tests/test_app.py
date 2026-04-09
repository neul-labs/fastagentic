"""Tests for the FastAgentic App class."""

from fastagentic import App
from fastagentic.app import AppConfig


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()
        assert config.title == "FastAgentic App"
        assert config.version == "1.0.0"
        assert config.description == ""
        assert config.mcp_enabled is True
        assert config.a2a_enabled is True
        assert config.telemetry is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = AppConfig(
            title="My Agent",
            version="2.0.0",
            description="A custom agent",
            oidc_issuer="https://auth.example.com",
            mcp_enabled=False,
        )
        assert config.title == "My Agent"
        assert config.version == "2.0.0"
        assert config.description == "A custom agent"
        assert config.oidc_issuer == "https://auth.example.com"
        assert config.mcp_enabled is False


class TestApp:
    """Tests for the App class."""

    def test_app_creation_defaults(self):
        """Test app creation with defaults."""
        app = App()
        assert app.config.title == "FastAgentic App"
        assert app.config.version == "1.0.0"
        assert app.fastapi is not None

    def test_app_creation_custom(self):
        """Test app creation with custom values."""
        app = App(
            title="Support Triage",
            version="1.0.0",
            description="Triage support tickets",
        )
        assert app.config.title == "Support Triage"
        assert app.config.version == "1.0.0"
        assert app.config.description == "Triage support tickets"

    def test_app_has_fastapi_instance(self):
        """Test app exposes FastAPI instance."""
        app = App(title="Test")
        assert hasattr(app, "fastapi")
        # Should be a FastAPI instance
        from fastapi import FastAPI

        assert isinstance(app.fastapi, FastAPI)

    def test_app_with_hooks(self):
        """Test app creation with hooks."""
        from fastagentic.hooks.base import Hook

        class TestHook(Hook):
            pass

        hook = TestHook()
        app = App(title="Test", hooks=[hook])
        assert hook in app._hooks

    def test_app_mcp_disabled(self):
        """Test app with MCP disabled."""
        app = App(title="Test", mcp_enabled=False)
        assert app.config.mcp_enabled is False

    def test_app_a2a_disabled(self):
        """Test app with A2A disabled."""
        app = App(title="Test", a2a_enabled=False)
        assert app.config.a2a_enabled is False

    def test_app_add_hook(self):
        """Test adding hooks after creation."""
        from fastagentic.hooks.base import Hook

        class TestHook(Hook):
            pass

        app = App(title="Test")
        hook = TestHook()
        app.add_hook(hook)
        assert hook in app._hooks


class TestAppRoutes:
    """Tests for App route registration."""

    def test_health_endpoint(self):
        """Test health endpoint is registered."""
        app = App(title="Test")
        routes = [r.path for r in app.fastapi.routes]
        assert "/health" in routes

    def test_ready_endpoint(self):
        """Test ready endpoint is registered."""
        app = App(title="Test")
        routes = [r.path for r in app.fastapi.routes]
        assert "/ready" in routes

    def test_mcp_routes_when_enabled(self):
        """Test MCP routes are registered when enabled."""
        app = App(title="Test", mcp_enabled=True)
        routes = [r.path for r in app.fastapi.routes]
        # Should have MCP schema route
        assert any("/mcp" in r for r in routes)

    def test_mcp_routes_not_registered_when_disabled(self):
        """Test MCP routes are not registered when disabled."""
        app = App(title="Test", mcp_enabled=False)
        routes = [r.path for r in app.fastapi.routes]
        # Should not have MCP routes (except other routes)
        mcp_routes = [r for r in routes if "/mcp/" in r]
        assert len(mcp_routes) == 0

    def test_a2a_routes_when_enabled(self):
        """Test A2A routes are registered when enabled."""
        app = App(title="Test", a2a_enabled=True)
        routes = [r.path for r in app.fastapi.routes]
        # Should have agent card route
        assert any("agent.json" in r for r in routes)

    def test_a2a_routes_not_registered_when_disabled(self):
        """Test A2A routes are not registered when disabled."""
        app = App(title="Test", a2a_enabled=False)
        routes = [r.path for r in app.fastapi.routes]
        # Should not have agent card route
        assert not any("agent.json" in r for r in routes)


class TestAppFastAPIProxies:
    """Tests for FastAPI method proxies."""

    def test_get_proxy(self):
        """Test GET route proxy."""
        app = App(title="Test")

        @app.get("/custom")
        def custom_get():
            return {"method": "GET"}

        routes = [r.path for r in app.fastapi.routes]
        assert "/custom" in routes

    def test_post_proxy(self):
        """Test POST route proxy."""
        app = App(title="Test")

        @app.post("/custom")
        def custom_post():
            return {"method": "POST"}

        routes = [r.path for r in app.fastapi.routes]
        assert "/custom" in routes

    def test_put_proxy(self):
        """Test PUT route proxy."""
        app = App(title="Test")

        @app.put("/custom")
        def custom_put():
            return {"method": "PUT"}

        routes = [r.path for r in app.fastapi.routes]
        assert "/custom" in routes

    def test_delete_proxy(self):
        """Test DELETE route proxy."""
        app = App(title="Test")

        @app.delete("/custom")
        def custom_delete():
            return {"method": "DELETE"}

        routes = [r.path for r in app.fastapi.routes]
        assert "/custom" in routes


class TestAppMemory:
    """Tests for App memory configuration."""

    def test_memory_property_none_by_default(self):
        """Test memory is None by default."""
        app = App(title="Test")
        assert app.memory is None

    def test_session_memory_property_none_by_default(self):
        """Test session_memory is None by default."""
        app = App(title="Test")
        assert app.session_memory is None
