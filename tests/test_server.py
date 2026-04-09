"""Tests for FastAgentic server module."""

import asyncio

import pytest
from starlette.testclient import TestClient

from fastagentic.server.config import PoolConfig, ServerConfig
from fastagentic.server.middleware import ConcurrencyLimitMiddleware, InstanceMetricsMiddleware


class TestPoolConfig:
    """Tests for PoolConfig."""

    def test_default_values(self):
        """Test default pool configuration."""
        config = PoolConfig()
        assert config.redis_pool_size == 10
        assert config.redis_pool_timeout == 5.0
        assert config.db_pool_size == 5
        assert config.db_max_overflow == 10

    def test_custom_values(self):
        """Test custom pool configuration."""
        config = PoolConfig(
            redis_pool_size=20,
            redis_pool_timeout=10.0,
            db_pool_size=15,
            db_max_overflow=30,
        )
        assert config.redis_pool_size == 20
        assert config.redis_pool_timeout == 10.0
        assert config.db_pool_size == 15
        assert config.db_max_overflow == 30

    def test_from_env(self, monkeypatch):
        """Test loading pool config from environment."""
        monkeypatch.setenv("FASTAGENTIC_REDIS_POOL_SIZE", "25")
        monkeypatch.setenv("FASTAGENTIC_DB_POOL_SIZE", "12")

        config = PoolConfig.from_env()
        assert config.redis_pool_size == 25
        assert config.db_pool_size == 12


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        assert config.server == "uvicorn"
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is False
        assert config.max_concurrent is None

    def test_uvicorn_config(self):
        """Test Uvicorn configuration."""
        config = ServerConfig(
            server="uvicorn",
            host="0.0.0.0",
            port=8080,
            workers=4,
        )
        assert config.server == "uvicorn"
        assert config.effective_workers() == 4

    def test_gunicorn_config(self):
        """Test Gunicorn configuration."""
        config = ServerConfig(
            server="gunicorn",
            workers=8,
            timeout_graceful_shutdown=60,
        )
        assert config.server == "gunicorn"
        assert config.workers == 8
        assert config.timeout_graceful_shutdown == 60

    def test_effective_workers_with_reload(self):
        """Test that reload forces single worker."""
        config = ServerConfig(workers=4, reload=True)
        assert config.effective_workers() == 1

    def test_effective_workers_without_reload(self):
        """Test normal worker count without reload."""
        config = ServerConfig(workers=4, reload=False)
        assert config.effective_workers() == 4

    def test_instance_id_generation(self):
        """Test instance ID is generated if not provided."""
        config = ServerConfig()
        instance_id = config.get_instance_id()
        assert instance_id is not None
        assert "-" in instance_id  # hostname-pid format

    def test_instance_id_provided(self):
        """Test instance ID is used when provided."""
        config = ServerConfig(instance_id="my-worker-1")
        assert config.get_instance_id() == "my-worker-1"

    def test_from_env(self, monkeypatch):
        """Test loading config from environment."""
        monkeypatch.setenv("FASTAGENTIC_SERVER", "gunicorn")
        monkeypatch.setenv("FASTAGENTIC_HOST", "0.0.0.0")
        monkeypatch.setenv("FASTAGENTIC_PORT", "9000")
        monkeypatch.setenv("FASTAGENTIC_WORKERS", "4")
        monkeypatch.setenv("FASTAGENTIC_MAX_CONCURRENT", "200")
        monkeypatch.setenv("FASTAGENTIC_INSTANCE_ID", "prod-1")

        config = ServerConfig.from_env()
        assert config.server == "gunicorn"
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.workers == 4
        assert config.max_concurrent == 200
        assert config.instance_id == "prod-1"


class TestConcurrencyLimitMiddleware:
    """Tests for ConcurrencyLimitMiddleware."""

    @pytest.fixture
    def app(self):
        """Create test app with concurrency middleware."""
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            await asyncio.sleep(0.1)
            return {"status": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        app.add_middleware(ConcurrencyLimitMiddleware, max_concurrent=2)
        return app

    def test_allows_requests_under_limit(self, app):
        """Test requests are allowed under the limit."""
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200

    def test_health_bypasses_limit(self, app):
        """Test health endpoints bypass concurrency limit."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200


class TestInstanceMetricsMiddleware:
    """Tests for InstanceMetricsMiddleware."""

    @pytest.fixture
    def app(self):
        """Create test app with instance metrics middleware."""
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(InstanceMetricsMiddleware, instance_id="test-instance-1")
        return app

    def test_adds_instance_header(self, app):
        """Test instance header is added to responses."""
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert response.headers.get("X-FastAgentic-Instance") == "test-instance-1"

    def test_default_instance_id(self):
        """Test default instance ID from environment."""
        middleware = InstanceMetricsMiddleware(app=lambda x: x)
        assert middleware.instance_id is not None

    def test_custom_instance_id(self):
        """Test custom instance ID."""
        middleware = InstanceMetricsMiddleware(
            app=lambda x: x,
            instance_id="custom-worker",
        )
        assert middleware.instance_id == "custom-worker"

    def test_metrics_collection(self):
        """Test metrics are collected."""
        middleware = InstanceMetricsMiddleware(
            app=lambda x: x,
            instance_id="test-worker",
        )
        metrics = middleware.get_metrics()
        assert metrics["instance_id"] == "test-worker"
        assert "request_count" in metrics
        assert "error_count" in metrics


class TestServerRunners:
    """Tests for server runners."""

    def test_run_uvicorn_import(self):
        """Test run_uvicorn can be imported."""
        from fastagentic.server.runners import run_uvicorn

        assert callable(run_uvicorn)

    def test_run_gunicorn_import(self):
        """Test run_gunicorn can be imported."""
        from fastagentic.server.runners import run_gunicorn

        assert callable(run_gunicorn)

    def test_get_recommended_workers(self):
        """Test recommended workers calculation."""
        import multiprocessing

        from fastagentic.server.runners import get_recommended_workers

        expected = (2 * multiprocessing.cpu_count()) + 1
        assert get_recommended_workers() == expected


class TestAppIntegration:
    """Integration tests for App with server configuration."""

    def test_app_with_max_concurrent(self):
        """Test App initializes with max_concurrent."""
        from fastagentic import App

        app = App(title="Test", max_concurrent=100)
        assert app.config.max_concurrent == 100

    def test_app_with_instance_id(self):
        """Test App initializes with instance_id."""
        from fastagentic import App

        app = App(title="Test", instance_id="worker-1")
        assert app.instance_id == "worker-1"

    def test_app_generates_instance_id(self):
        """Test App generates instance ID if not provided."""
        from fastagentic import App

        app = App(title="Test")
        assert app.instance_id is not None
        assert "-" in app.instance_id

    def test_app_pool_config_from_env(self, monkeypatch):
        """Test App reads pool config from environment."""
        monkeypatch.setenv("FASTAGENTIC_REDIS_POOL_SIZE", "50")
        monkeypatch.setenv("FASTAGENTIC_DB_POOL_SIZE", "25")

        from fastagentic import App

        app = App(title="Test")
        assert app.config.redis_pool_size == 50
        assert app.config.db_pool_size == 25

    def test_health_endpoint_includes_instance_id(self):
        """Test health endpoint includes instance ID."""
        from starlette.testclient import TestClient

        from fastagentic import App

        app = App(title="Test", instance_id="health-test-1")
        client = TestClient(app.fastapi)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["instance_id"] == "health-test-1"

    def test_ready_endpoint_includes_instance_id(self):
        """Test ready endpoint includes instance ID."""
        from starlette.testclient import TestClient

        from fastagentic import App

        app = App(title="Test", instance_id="ready-test-1")
        client = TestClient(app.fastapi)

        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["instance_id"] == "ready-test-1"

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns config."""
        from starlette.testclient import TestClient

        from fastagentic import App

        app = App(
            title="Test",
            instance_id="metrics-test-1",
            max_concurrent=100,
        )
        client = TestClient(app.fastapi)

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["instance_id"] == "metrics-test-1"
        assert data["config"]["max_concurrent"] == 100
