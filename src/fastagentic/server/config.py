"""Server configuration for production deployments."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PoolConfig:
    """Connection pool configuration for databases and caches.

    Configure pool sizes for Redis and PostgreSQL connections.
    These settings optimize connection reuse and prevent exhaustion
    under high load.

    Example:
        pool = PoolConfig(
            redis_pool_size=20,
            redis_pool_timeout=5.0,
            db_pool_size=10,
            db_max_overflow=20,
        )
    """

    # Redis connection pool
    redis_pool_size: int = 10
    redis_pool_timeout: float = 5.0
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    redis_retry_on_timeout: bool = True

    # PostgreSQL connection pool
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: float = 30.0
    db_pool_recycle: int = 1800  # 30 minutes
    db_pool_pre_ping: bool = True

    @classmethod
    def from_env(cls) -> PoolConfig:
        """Create PoolConfig from environment variables."""
        return cls(
            redis_pool_size=int(os.environ.get("FASTAGENTIC_REDIS_POOL_SIZE", "10")),
            redis_pool_timeout=float(os.environ.get("FASTAGENTIC_REDIS_POOL_TIMEOUT", "5.0")),
            redis_socket_timeout=float(os.environ.get("FASTAGENTIC_REDIS_SOCKET_TIMEOUT", "5.0")),
            db_pool_size=int(os.environ.get("FASTAGENTIC_DB_POOL_SIZE", "5")),
            db_max_overflow=int(os.environ.get("FASTAGENTIC_DB_MAX_OVERFLOW", "10")),
            db_pool_timeout=float(os.environ.get("FASTAGENTIC_DB_POOL_TIMEOUT", "30.0")),
        )


@dataclass
class ServerConfig:
    """Production server configuration.

    Configure server type, workers, concurrency limits, and instance
    identification for cluster deployments.

    Example:
        config = ServerConfig(
            server="gunicorn",
            host="0.0.0.0",
            port=8000,
            workers=4,
            max_concurrent=100,
            instance_id="worker-1",
        )
    """

    # Server type
    server: Literal["uvicorn", "gunicorn"] = "uvicorn"

    # Binding
    host: str = "127.0.0.1"
    port: int = 8000

    # Workers
    workers: int = 1

    # Development
    reload: bool = False

    # Concurrency limits
    max_concurrent: int | None = None  # None = unlimited

    # Cluster identification
    instance_id: str | None = None

    # Timeouts
    timeout_keep_alive: int = 5
    timeout_graceful_shutdown: int = 30

    # Connection pools
    pool: PoolConfig = field(default_factory=PoolConfig)

    # Worker class for Gunicorn
    worker_class: str = "uvicorn.workers.UvicornWorker"

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Create ServerConfig from environment variables."""
        return cls(
            server=os.environ.get("FASTAGENTIC_SERVER", "uvicorn"),  # type: ignore
            host=os.environ.get("FASTAGENTIC_HOST", "127.0.0.1"),
            port=int(os.environ.get("FASTAGENTIC_PORT", "8000")),
            workers=int(os.environ.get("FASTAGENTIC_WORKERS", "1")),
            reload=os.environ.get("FASTAGENTIC_RELOAD", "").lower() == "true",
            max_concurrent=int(mc)
            if (mc := os.environ.get("FASTAGENTIC_MAX_CONCURRENT"))
            else None,
            instance_id=os.environ.get("FASTAGENTIC_INSTANCE_ID") or _generate_instance_id(),
            timeout_keep_alive=int(os.environ.get("FASTAGENTIC_TIMEOUT_KEEP_ALIVE", "5")),
            timeout_graceful_shutdown=int(
                os.environ.get("FASTAGENTIC_TIMEOUT_GRACEFUL_SHUTDOWN", "30")
            ),
            pool=PoolConfig.from_env(),
        )

    def effective_workers(self) -> int:
        """Get effective worker count (1 if reload enabled)."""
        return 1 if self.reload else self.workers

    def get_instance_id(self) -> str:
        """Get or generate instance ID for metrics labeling."""
        return self.instance_id or _generate_instance_id()


def _generate_instance_id() -> str:
    """Generate a unique instance ID for this server."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"
