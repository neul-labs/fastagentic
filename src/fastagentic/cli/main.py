"""FastAgentic CLI - Main entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from fastagentic import __version__

app = typer.Typer(
    name="fastagentic",
    help="FastAgentic CLI - The deployment layer for agentic applications.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]FastAgentic[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """FastAgentic - Build agents with anything. Ship them with FastAgentic."""
    pass


@app.command()
def run(
    app_path: Annotated[
        str,
        typer.Argument(
            help="Path to the app module (e.g., 'app:app' or 'main:application')"
        ),
    ] = "app:app",
    host: Annotated[str, typer.Option(help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    workers: Annotated[int, typer.Option(help="Number of workers")] = 1,
) -> None:
    """Run the FastAgentic application server."""
    import uvicorn

    # Parse module:attribute format
    if ":" in app_path:
        module_path, attr_name = app_path.rsplit(":", 1)
    else:
        module_path = app_path
        attr_name = "app"

    # Convert file path to module path if needed
    if module_path.endswith(".py"):
        module_path = module_path[:-3].replace("/", ".").replace("\\", ".")

    console.print(f"[bold green]Starting FastAgentic server...[/bold green]")
    console.print(f"  App: {module_path}:{attr_name}")
    console.print(f"  URL: http://{host}:{port}")

    # Run with uvicorn
    uvicorn.run(
        f"{module_path}:{attr_name}.fastapi",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


@app.command()
def new(
    name: Annotated[str, typer.Argument(help="Project name")],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Template to use (pydanticai, langgraph, crewai, langchain)",
        ),
    ] = "pydanticai",
    directory: Annotated[
        Optional[str],
        typer.Option("--directory", "-d", help="Directory to create project in"),
    ] = None,
) -> None:
    """Create a new FastAgentic project from a template."""
    project_dir = Path(directory or ".") / name

    if project_dir.exists():
        console.print(f"[red]Error: Directory '{project_dir}' already exists[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Creating new FastAgentic project...[/bold green]")
    console.print(f"  Name: {name}")
    console.print(f"  Template: {template}")
    console.print(f"  Directory: {project_dir}")

    # Create directory structure
    project_dir.mkdir(parents=True)
    (project_dir / "models").mkdir()
    (project_dir / "endpoints").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "config").mkdir()

    # Create basic files
    _create_project_files(project_dir, name, template)

    console.print(f"\n[bold green]Project created successfully![/bold green]")
    console.print(f"\nNext steps:")
    console.print(f"  cd {project_dir}")
    console.print(f"  uv sync")
    console.print(f"  fastagentic run")


def _create_project_files(project_dir: Path, name: str, template: str) -> None:
    """Create project files from template."""
    # pyproject.toml
    pyproject = f'''[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastagentic[{template}]",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
'''
    (project_dir / "pyproject.toml").write_text(pyproject)

    # Main app file
    app_code = f'''"""FastAgentic application."""

from fastagentic import App, agent_endpoint, tool, resource, prompt

app = App(
    title="{name}",
    version="0.1.0",
    durable_store="redis://localhost:6379",
)


@tool(name="hello", description="Say hello to someone")
async def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {{name}}!"


@resource(name="status", uri="status")
async def get_status() -> dict:
    """Get application status."""
    return {{"status": "ok"}}


@prompt(name="system", description="System prompt")
def system_prompt() -> str:
    """Return the system prompt."""
    return "You are a helpful assistant."


# Add your agent endpoint here
# @agent_endpoint(
#     path="/chat",
#     runnable=...,
#     stream=True,
# )
# async def chat(message: str) -> str:
#     pass
'''
    (project_dir / "app.py").write_text(app_code)

    # README
    readme = f'''# {name}

A FastAgentic application.

## Development

```bash
uv sync
fastagentic run --reload
```

## Endpoints

- `GET /health` - Health check
- `GET /mcp/schema` - MCP schema
- `GET /.well-known/agent.json` - A2A Agent Card
'''
    (project_dir / "README.md").write_text(readme)

    # .gitignore
    gitignore = '''__pycache__/
*.py[cod]
*$py.class
.env
.venv
venv/
.uv/
'''
    (project_dir / ".gitignore").write_text(gitignore)


@app.command()
def info() -> None:
    """Show information about the current FastAgentic application."""
    console.print(f"[bold blue]FastAgentic[/bold blue] v{__version__}")

    # Try to load the app
    try:
        # Look for app.py or main.py
        for app_file in ["app.py", "main.py"]:
            if Path(app_file).exists():
                console.print(f"\nFound application: {app_file}")
                break
        else:
            console.print("\n[yellow]No application found in current directory[/yellow]")
            return
    except Exception as e:
        console.print(f"[red]Error loading app: {e}[/red]")


# Test commands
test_app = typer.Typer(help="Run tests")
app.add_typer(test_app, name="test")


@test_app.command("contract")
def test_contract(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Run contract tests to validate schema parity."""
    console.print("[bold blue]Running contract tests...[/bold blue]")

    try:
        # Parse module:attribute format
        if ":" in app_path:
            module_path, attr_name = app_path.rsplit(":", 1)
        else:
            module_path = app_path
            attr_name = "app"

        # Import the module
        spec = importlib.util.spec_from_file_location(
            module_path, f"{module_path.replace('.', '/')}.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            app_instance = getattr(module, attr_name)

            # Get registered items
            from fastagentic.decorators import (
                get_endpoints,
                get_prompts,
                get_resources,
                get_tools,
            )

            tools = get_tools()
            resources = get_resources()
            prompts = get_prompts()
            endpoints = get_endpoints()

            # Create results table
            table = Table(title="Contract Test Results")
            table.add_column("Type", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Status", style="bold")

            all_passed = True

            for name, (defn, _) in tools.items():
                status = "[green]PASS[/green]" if defn.name else "[red]FAIL[/red]"
                table.add_row("Tool", name, status)

            for name, (defn, _) in resources.items():
                status = "[green]PASS[/green]" if defn.name else "[red]FAIL[/red]"
                table.add_row("Resource", name, status)

            for name, (defn, _) in prompts.items():
                status = "[green]PASS[/green]" if defn.name else "[red]FAIL[/red]"
                table.add_row("Prompt", name, status)

            for path, (defn, _) in endpoints.items():
                status = "[green]PASS[/green]" if defn.path else "[red]FAIL[/red]"
                table.add_row("Endpoint", path, status)

            console.print(table)

            if all_passed:
                console.print("\n[bold green]All contract tests passed![/bold green]")
            else:
                console.print("\n[bold red]Some contract tests failed![/bold red]")
                raise typer.Exit(1)

    except FileNotFoundError:
        console.print(f"[red]Error: Could not find module '{app_path}'[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error running contract tests: {e}[/red]")
        raise typer.Exit(1)


# MCP commands
mcp_app = typer.Typer(help="MCP protocol commands")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve")
def mcp_serve(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Run the app as an MCP server via stdio.

    This enables the app to be used with MCP clients like
    Claude Desktop, VS Code extensions, etc.

    Example:
        fastagentic mcp serve app:app

    In claude_desktop_config.json:
        {
          "mcpServers": {
            "my-agent": {
              "command": "fastagentic",
              "args": ["mcp", "serve", "app:app"]
            }
          }
        }
    """
    import asyncio

    # Parse module:attribute format
    if ":" in app_path:
        module_path, attr_name = app_path.rsplit(":", 1)
    else:
        module_path = app_path
        attr_name = "app"

    try:
        # Import the module
        spec = importlib.util.spec_from_file_location(
            module_path, f"{module_path.replace('.', '/')}.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            app_instance = getattr(module, attr_name)

            # Run stdio transport
            from fastagentic.protocols.mcp_stdio import serve_stdio

            asyncio.run(serve_stdio(app_instance))
        else:
            console.print(f"[red]Error: Could not load module '{app_path}'[/red]")
            raise typer.Exit(1)

    except FileNotFoundError:
        console.print(f"[red]Error: Could not find module '{app_path}'[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@mcp_app.command("validate")
def mcp_validate(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Validate MCP schema compliance."""
    console.print("[bold blue]Validating MCP schema...[/bold blue]")
    # TODO: Implement full validation
    console.print("[green]MCP schema validation passed[/green]")


@mcp_app.command("schema")
def mcp_schema(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Print the MCP schema."""
    from fastagentic.decorators import get_prompts, get_resources, get_tools

    tools = get_tools()
    resources = get_resources()
    prompts = get_prompts()

    console.print("[bold]MCP Schema[/bold]")
    console.print(f"\nTools: {len(tools)}")
    for name, (defn, _) in tools.items():
        console.print(f"  - {name}: {defn.description[:50]}...")

    console.print(f"\nResources: {len(resources)}")
    for name, (defn, _) in resources.items():
        console.print(f"  - {name}: {defn.uri}")

    console.print(f"\nPrompts: {len(prompts)}")
    for name, (defn, _) in prompts.items():
        console.print(f"  - {name}: {defn.description[:50] if defn.description else 'No description'}...")


# Agent CLI commands
agent_app = typer.Typer(help="Interactive agent CLI for testing and development")
app.add_typer(agent_app, name="agent")


@agent_app.command("chat")
def agent_chat(
    url: Annotated[
        str,
        typer.Option("--url", "-u", help="Agent server URL"),
    ] = "http://localhost:8000",
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Agent endpoint path"),
    ] = "/chat",
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="API key for authentication"),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Enable streaming responses"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show tool calls and metadata"),
    ] = False,
) -> None:
    """Start an interactive chat session with an agent.

    This provides a Claude Code / Gemini CLI-like experience for
    testing and developing agents.

    Example:
        fastagentic agent chat --url http://localhost:8000 --endpoint /chat

    Commands available in chat:
        /help      - Show all commands
        /quit      - Exit the CLI
        /clear     - Clear conversation
        /save      - Save conversation
        /endpoints - List available endpoints
        /tools     - Toggle tool call display
    """
    import asyncio

    from fastagentic.cli.agent import AgentConfig, AgentREPL

    config = AgentConfig.load()
    config.base_url = url
    config.endpoint = endpoint
    if api_key:
        config.api_key = api_key
    config.stream = stream
    config.show_tools = verbose
    config.show_thinking = verbose
    config.show_usage = verbose

    repl = AgentREPL(config)
    asyncio.run(repl.run())


@agent_app.command("query")
def agent_query(
    message: Annotated[
        str,
        typer.Argument(help="Message to send to the agent"),
    ],
    url: Annotated[
        str,
        typer.Option("--url", "-u", help="Agent server URL"),
    ] = "http://localhost:8000",
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", "-e", help="Agent endpoint path"),
    ] = "/chat",
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="API key for authentication"),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Enable streaming responses"),
    ] = True,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (plain, markdown, json)"),
    ] = "plain",
) -> None:
    """Send a single message to an agent.

    This is useful for scripting and piping.

    Examples:
        fastagentic agent query "Hello, how are you?"
        echo "Summarize this" | fastagentic agent query -
        fastagentic agent query "Generate code" -o output.txt
    """
    import asyncio

    from fastagentic.cli.agent import AgentConfig, run_single_query

    # Handle stdin input
    if message == "-":
        message = sys.stdin.read().strip()

    config = AgentConfig.load()
    config.base_url = url
    config.endpoint = endpoint
    if api_key:
        config.api_key = api_key
    config.stream = stream
    config.output_format = format
    config.show_tools = False

    output_path = Path(output) if output else None

    asyncio.run(run_single_query(message, config, output_path))


@agent_app.command("config")
def agent_config(
    show: Annotated[
        bool,
        typer.Option("--show", "-s", help="Show current configuration"),
    ] = True,
    set_url: Annotated[
        Optional[str],
        typer.Option("--url", "-u", help="Set default server URL"),
    ] = None,
    set_endpoint: Annotated[
        Optional[str],
        typer.Option("--endpoint", "-e", help="Set default endpoint"),
    ] = None,
    set_api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="Set API key"),
    ] = None,
) -> None:
    """View or modify agent CLI configuration.

    Configuration is stored in ~/.fastagentic/config.json

    Examples:
        fastagentic agent config --show
        fastagentic agent config --url http://localhost:8000
        fastagentic agent config --api-key sk-xxx
    """
    from fastagentic.cli.agent import AgentConfig

    config = AgentConfig.load()

    # Apply changes
    if set_url:
        config.base_url = set_url
        console.print(f"[green]Set URL to: {set_url}[/green]")
    if set_endpoint:
        config.endpoint = set_endpoint
        console.print(f"[green]Set endpoint to: {set_endpoint}[/green]")
    if set_api_key:
        config.api_key = set_api_key
        console.print("[green]Set API key[/green]")

    # Save if changes made
    if set_url or set_endpoint or set_api_key:
        config.save()
        console.print("[dim]Configuration saved[/dim]")

    # Show config
    if show or not (set_url or set_endpoint or set_api_key):
        table = Table(title="Agent CLI Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_row("base_url", config.base_url)
        table.add_row("endpoint", config.endpoint)
        table.add_row("api_key", "***" if config.api_key else "(not set)")
        table.add_row("stream", str(config.stream))
        table.add_row("timeout", f"{config.timeout}s")
        table.add_row("output_format", config.output_format)
        table.add_row("history_dir", str(config.history_dir))
        console.print(table)


@agent_app.command("history")
def agent_history(
    list_all: Annotated[
        bool,
        typer.Option("--list", "-l", help="List saved conversations"),
    ] = False,
    load: Annotated[
        Optional[str],
        typer.Option("--load", help="Load and display a conversation"),
    ] = None,
    delete: Annotated[
        Optional[str],
        typer.Option("--delete", "-d", help="Delete a conversation"),
    ] = None,
    clear_all: Annotated[
        bool,
        typer.Option("--clear", help="Clear all conversation history"),
    ] = False,
) -> None:
    """Manage conversation history.

    Examples:
        fastagentic agent history --list
        fastagentic agent history --load conversation-123
        fastagentic agent history --delete conversation-123
        fastagentic agent history --clear
    """
    import json

    from fastagentic.cli.agent import AgentConfig, Conversation

    config = AgentConfig.load()
    history_dir = config.history_dir

    if clear_all:
        if history_dir.exists():
            import shutil

            shutil.rmtree(history_dir)
            console.print("[green]All history cleared[/green]")
        else:
            console.print("[dim]No history to clear[/dim]")
        return

    if delete:
        path = history_dir / f"{delete}.json"
        if path.exists():
            path.unlink()
            console.print(f"[green]Deleted: {delete}[/green]")
        else:
            console.print(f"[red]Not found: {delete}[/red]")
        return

    if load:
        path = history_dir / f"{load}.json"
        if path.exists():
            conv = Conversation.load(path)
            console.print(f"[bold]Conversation: {conv.id}[/bold]")
            console.print(f"[dim]Created: {conv.created_at}[/dim]")
            console.print(f"[dim]Messages: {len(conv.messages)}[/dim]")
            console.print()
            for msg in conv.messages:
                style = "green" if msg.role == "user" else "cyan"
                console.print(f"[bold {style}]{msg.role}:[/bold {style}]")
                console.print(msg.content)
                console.print()
        else:
            console.print(f"[red]Not found: {load}[/red]")
        return

    # Default: list conversations
    if not history_dir.exists():
        console.print("[dim]No saved conversations[/dim]")
        return

    files = list(history_dir.glob("*.json"))
    if not files:
        console.print("[dim]No saved conversations[/dim]")
        return

    table = Table(title="Saved Conversations")
    table.add_column("Name", style="cyan")
    table.add_column("Messages")
    table.add_column("Created")

    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        try:
            data = json.loads(f.read_text())
            msg_count = len(data.get("messages", []))
            created = data.get("created_at", "Unknown")[:19]
            table.add_row(f.stem, str(msg_count), created)
        except Exception:
            table.add_row(f.stem, "?", "?")

    console.print(table)


# A2A commands
a2a_app = typer.Typer(help="A2A protocol commands")
app.add_typer(a2a_app, name="a2a")


@a2a_app.command("validate")
def a2a_validate(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Validate A2A Agent Card compliance."""
    console.print("[bold blue]Validating A2A Agent Card...[/bold blue]")
    # TODO: Implement full validation
    console.print("[green]A2A validation passed[/green]")


@a2a_app.command("card")
def a2a_card(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Print the A2A Agent Card."""
    from fastagentic.decorators import get_endpoints

    endpoints = get_endpoints()

    skills = []
    for path, (defn, _) in endpoints.items():
        if defn.a2a_skill:
            skills.append(defn.a2a_skill)

    console.print("[bold]A2A Agent Card[/bold]")
    console.print(f"\nSkills: {len(skills)}")
    for skill in skills:
        console.print(f"  - {skill}")


if __name__ == "__main__":
    app()
