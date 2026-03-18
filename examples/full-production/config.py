"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Environment
    env: str = "dev"

    # Authentication
    oidc_issuer: str = ""
    oidc_audience: str = ""

    # Storage
    redis_url: str = "redis://localhost:6379"

    # Rate Limits
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    # Budget
    budget_daily_usd: float = 100.0

    # Observability
    otel_endpoint: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Security
    lakera_api_key: str = ""

    class Config:
        env_prefix = "FASTAGENTIC_"
        env_file = ".env"


settings = Settings()
