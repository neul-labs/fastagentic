"""Configuration management commands for FastAgentic CLI."""

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

config_app = typer.Typer(
    name="config",
    help="Manage FastAgentic configuration.",
    no_args_is_help=True,
)


@config_app.command("show")
def config_show(
    env: Annotated[
        str | None,
        typer.Option("--env", "-e", help="Show configuration for specific environment"),
    ] = None,
    secrets: Annotated[
        bool,
        typer.Option("--secrets", "-s", help="Show secrets (use with caution)"),
    ] = False,
) -> None:
    """Display resolved configuration."""
    console.print("[bold]FastAgentic Configuration[/bold]\n")

    # Environment info
    fastagentic_env = env or os.environ.get("FASTAGENTIC_ENV", "unknown")
    console.print(f"Environment: [cyan]{fastagentic_env}[/cyan]")

    # Server configuration
    console.print("\n[bold]Server[/bold]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Server", os.environ.get("FASTAGENTIC_SERVER", "uvicorn"))
    table.add_row("Host", os.environ.get("FASTAGENTIC_HOST", "127.0.0.1"))
    table.add_row("Port", os.environ.get("FASTAGENTIC_PORT", "8000"))
    table.add_row("Workers", os.environ.get("FASTAGENTIC_WORKERS", "1"))
    console.print(table)

    # Connection pools
    console.print("\n[bold]Connection Pools[/bold]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Redis Pool Size", os.environ.get("FASTAGENTIC_REDIS_POOL_SIZE", "10"))
    table.add_row("DB Pool Size", os.environ.get("FASTAGENTIC_DB_POOL_SIZE", "5"))
    table.add_row("DB Max Overflow", os.environ.get("FASTAGENTIC_DB_MAX_OVERFLOW", "10"))
    console.print(table)

    # Limits
    console.print("\n[bold]Limits[/bold]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Max Concurrent", os.environ.get("FASTAGENTIC_MAX_CONCURRENT", "unlimited"))
    table.add_row("Instance ID", os.environ.get("FASTAGENTIC_INSTANCE_ID", "auto-generated"))
    console.print(table)

    # Log level
    console.print("\n[bold]Logging[/bold]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Log Level", os.environ.get("FASTAGENTIC_LOG_LEVEL", "INFO"))
    table.add_row("Log Format", os.environ.get("FASTAGENTIC_LOG_FORMAT", "text"))
    console.print(table)

    # Show secrets if requested
    if secrets:
        console.print("\n[bold red]WARNING: Secrets exposed![/bold red]\n")
        table = Table()
        table.add_column("Variable", style="cyan")
        table.add_column("Value", style="red")
        secret_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "FASTAGENTIC_API_KEY",
        ]
        for var in secret_vars:
            val = os.environ.get(var, "")
            if val:
                masked = val[:4] + "*" * (len(val) - 4) if len(val) > 4 else "****"
                table.add_row(var, masked)
        console.print(table)
    else:
        console.print("\n[dim]Use --secrets to show masked values[/dim]")


@config_app.command("validate")
def config_validate(
    path: Annotated[
        Path,
        typer.Argument(help="Path to configuration file"),
    ],
) -> None:
    """Validate a configuration file."""
    console.print(f"[bold]Validating configuration: {path}[/bold]\n")

    if not path.exists():
        console.print(f"[red]Error: File not found: {path}[/red]")
        raise typer.Exit(1)

    errors: list[str] = []
    warnings: list[str] = []

    try:
        content = path.read_text()
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1) from e

    # Basic YAML validation
    try:
        import yaml

        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML: {e}")
    except Exception as e:
        errors.append(f"Parse error: {e}")

    if errors:
        console.print("[red]Validation failed:[/red]")
        for error in errors:
            console.print(f"  [red]ERROR: {error}[/red]")
        raise typer.Exit(1)

    # Check for required fields
    if config:
        # Check for common structure
        if not isinstance(config, dict):
            warnings.append("Config should be a dictionary")

        # Check for known keys
        known_keys = {"title", "version", "host", "port", "server", "adapters"}
        for key in config:
            if key not in known_keys:
                warnings.append(f"Unknown key: '{key}'")

    if warnings:
        console.print("[yellow]Validation passed with warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]WARNING: {warning}[/yellow]")
    else:
        console.print("[green]Configuration is valid[/green]")

    console.print(f"\n[dim]File: {path}[/dim]")


@config_app.command("init")
def config_init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("settings.yaml"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (yaml, env)"),
    ] = "yaml",
) -> None:
    """Generate a default configuration file."""
    console.print(f"[bold]Generating default configuration: {output}[/bold]\n")

    if format == "yaml":
        config_content = """# FastAgentic Configuration

# Application
title: My FastAgentic App
version: 0.1.0

# Server Configuration
host: 127.0.0.1
port: 8000
server: uvicorn
workers: 1

# Connection Pools
redis_pool_size: 10
db_pool_size: 5
db_max_overflow: 10

# Limits
max_concurrent: null  # unlimited
instance_id: null  # auto-generated

# Logging
log_level: INFO
log_format: text

# Optional: Authentication
# auth:
#   type: none  # none, oidc, api_key

# Optional: Adapters
# adapters:
#   pydanticai:
#     model: openai:gpt-4o-mini
"""
    else:  # env format
        config_content = """# FastAgentic Environment Variables

# Application
FASTAGENTIC_TITLE="My FastAgentic App"
FASTAGENTIC_VERSION="0.1.0"

# Server Configuration
FASTAGENTIC_HOST="127.0.0.1"
FASTAGENTIC_PORT="8000"
FASTAGENTIC_SERVER="uvicorn"
FASTAGENTIC_WORKERS="1"

# Connection Pools
FASTAGENTIC_REDIS_POOL_SIZE="10"
FASTAGENTIC_DB_POOL_SIZE="5"
FASTAGENTIC_DB_MAX_OVERFLOW="10"

# Limits
# FASTAGENTIC_MAX_CONCURRENT="100"
# FASTAGENTIC_INSTANCE_ID="worker-1"

# Logging
FASTAGENTIC_LOG_LEVEL="INFO"
FASTAGENTIC_LOG_FORMAT="text"

# LLM Provider
# OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="sk-ant-..."
"""

    try:
        output.write_text(config_content)
        console.print(f"[green]Configuration written to: {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing file: {e}[/red]")
        raise typer.Exit(1) from e
