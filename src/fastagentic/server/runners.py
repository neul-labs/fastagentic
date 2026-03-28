"""Server runners for Uvicorn and Gunicorn."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastagentic.server.config import ServerConfig


def run_uvicorn(app_path: str, config: ServerConfig) -> None:
    """Run the application with Uvicorn.

    Args:
        app_path: Module:attribute path to the FastAPI app (e.g., "app:app.fastapi")
        config: Server configuration
    """
    import uvicorn

    # Ensure current directory is in Python path for module imports
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    uvicorn.run(
        app_path,
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.effective_workers(),
        timeout_keep_alive=config.timeout_keep_alive,
        log_level="info",
    )


def run_gunicorn(app_path: str, config: ServerConfig) -> None:
    """Run the application with Gunicorn + Uvicorn workers.

    This provides production-grade process management with:
    - Pre-fork worker model for parallel processing
    - Graceful restarts and zero-downtime deployments
    - Worker recycling to prevent memory leaks
    - Signal handling for shutdown

    Args:
        app_path: Module:attribute path to the FastAPI app (e.g., "app:app.fastapi")
        config: Server configuration
    """
    try:
        from gunicorn.app.base import BaseApplication
    except ImportError:
        print("Error: gunicorn is not installed.", file=sys.stderr)
        print("Install it with: pip install gunicorn", file=sys.stderr)
        sys.exit(1)

    class FastAgenticApplication(BaseApplication):
        """Custom Gunicorn application."""

        def __init__(self, app_path: str, options: dict | None = None):
            self.app_path = app_path
            self.options = options or {}
            super().__init__()

        def load_config(self) -> None:
            """Load Gunicorn configuration."""
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self) -> str:
            """Return the ASGI application path."""
            return self.app_path

    # Gunicorn options
    options = {
        "bind": f"{config.host}:{config.port}",
        "workers": config.effective_workers(),
        "worker_class": config.worker_class,
        "timeout": config.timeout_graceful_shutdown,
        "keepalive": config.timeout_keep_alive,
        "graceful_timeout": config.timeout_graceful_shutdown,
        "max_requests": 10000,  # Recycle workers after 10k requests
        "max_requests_jitter": 1000,  # Add jitter to prevent thundering herd
        "preload_app": not config.reload,
        "accesslog": "-",  # Log to stdout
        "errorlog": "-",  # Log to stderr
        "capture_output": True,
        "loglevel": "info",
    }

    # Environment variables for workers
    os.environ["FASTAGENTIC_INSTANCE_ID"] = config.get_instance_id()

    # Pool configuration for workers
    os.environ["FASTAGENTIC_REDIS_POOL_SIZE"] = str(config.pool.redis_pool_size)
    os.environ["FASTAGENTIC_DB_POOL_SIZE"] = str(config.pool.db_pool_size)
    os.environ["FASTAGENTIC_DB_MAX_OVERFLOW"] = str(config.pool.db_max_overflow)

    # Max concurrent requests
    if config.max_concurrent:
        os.environ["FASTAGENTIC_MAX_CONCURRENT"] = str(config.max_concurrent)

    FastAgenticApplication(app_path, options).run()


def get_recommended_workers() -> int:
    """Get recommended worker count based on CPU cores.

    Returns:
        Recommended number of workers (2 * CPU cores + 1)
    """
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    return (2 * cpu_count) + 1
