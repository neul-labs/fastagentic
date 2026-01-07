"""Template management commands for FastAgentic CLI."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# Built-in templates
BUILTIN_TEMPLATES = {
    "pydanticai": {
        "name": "pydanticai",
        "category": "official",
        "description": "PydanticAI agent with type-safe responses",
        "adapter": "PydanticAI",
    },
    "langgraph": {
        "name": "langgraph",
        "category": "official",
        "description": "LangGraph stateful workflow",
        "adapter": "LangGraph",
    },
    "crewai": {
        "name": "crewai",
        "category": "official",
        "description": "CrewAI multi-agent collaboration",
        "adapter": "CrewAI",
    },
    "langchain": {
        "name": "langchain",
        "category": "official",
        "description": "LangChain chain/runnable",
        "adapter": "LangChain",
    },
}

templates_app = typer.Typer(
    name="templates",
    help="Manage FastAgentic project templates.",
    no_args_is_help=True,
)


def _get_all_templates() -> list[dict]:
    """Get all available templates (builtin + cached)."""
    templates = []

    # Add built-in templates
    for _key, data in BUILTIN_TEMPLATES.items():
        templates.append(
            {
                "name": data["name"],
                "category": data["category"],
                "description": data["description"],
                "source": "builtin",
            }
        )

    # Check for cached remote templates
    cache_dir = Path.home() / ".fastagentic" / "templates"
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            try:
                import json

                data = json.loads(cache_file.read_text())
                templates.append(
                    {
                        "name": data.get("name", cache_file.stem),
                        "category": data.get("category", "community"),
                        "description": data.get("description", ""),
                        "source": "cached",
                    }
                )
            except Exception:
                pass

    return templates


@templates_app.command("list")
def templates_list(
    category: Annotated[
        str,
        typer.Option("--category", "-c", help="Filter by category (official, community, all)"),
    ] = "all",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed information"),
    ] = False,
) -> None:
    """List available project templates."""
    templates = _get_all_templates()

    # Filter by category
    if category != "all":
        templates = [t for t in templates if t["category"] == category]

    if not templates:
        console.print("[yellow]No templates found[/yellow]")
        return

    if verbose:
        table = Table(title="Available Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Source", style="blue")

        for t in templates:
            table.add_row(t["name"], t["category"], t["description"], t["source"])

        console.print(table)
    else:
        table = Table(title="Available Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="green")

        for t in templates:
            table.add_row(t["name"], t["category"], t["description"])

        console.print(table)

    console.print(f"\n[dim]Total: {len(templates)} templates[/dim]")


@templates_app.command("search")
def templates_search(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ],
) -> None:
    """Search for templates by name or description."""
    templates = _get_all_templates()
    query_lower = query.lower()

    # Filter templates matching the query
    matching = [
        t
        for t in templates
        if query_lower in t["name"].lower() or query_lower in t["description"].lower()
    ]

    if not matching:
        console.print(f"[yellow]No templates found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="green")

    for t in matching:
        table.add_row(t["name"], t["category"], t["description"])

    console.print(table)
    console.print(f"\n[dim]Found {len(matching)} matching template(s)[/dim]")


@templates_app.command("info")
def templates_info(
    name: Annotated[
        str,
        typer.Argument(help="Template name"),
    ],
) -> None:
    """Show detailed information about a template."""
    # Check built-in templates first
    if name in BUILTIN_TEMPLATES:
        data = BUILTIN_TEMPLATES[name]
        console.print(f"[bold cyan]{data['name']}[/bold cyan]")
        console.print(f"Category: {data['category']}")
        console.print(f"Adapter: {data['adapter']}")
        console.print(f"Description: {data['description']}")
        console.print("Source: builtin")
        return

    # Check cached templates
    cache_dir = Path.home() / ".fastagentic" / "templates"
    cache_file = cache_dir / f"{name}.json"

    if cache_file.exists():
        try:
            import json

            data = json.loads(cache_file.read_text())
            console.print(f"[bold cyan]{data.get('name', name)}[/bold cyan]")
            console.print(f"Category: {data.get('category', 'community')}")
            console.print(f"Description: {data.get('description', 'N/A')}")
            console.print("Source: cached")
            return
        except Exception as e:
            console.print(f"[red]Error reading cached template: {e}[/red]")
            return

    console.print(f"[red]Template '{name}' not found[/red]")


@templates_app.command("refresh")
def templates_refresh() -> None:
    """Refresh template cache from remote repository."""
    console.print("[yellow]Remote template registry not yet implemented[/yellow]")
    console.print("\n[dim]To create a new project, use:[/dim]")
    console.print("  fastagentic new <name> --template <template-name>")
    console.print("\n[dim]Available templates:[/dim]")
    console.print("  - pydanticai (PydanticAI agents)")
    console.print("  - langgraph (LangGraph workflows)")
    console.print("  - crewai (CrewAI multi-agent)")
    console.print("  - langchain (LangChain chains)")
