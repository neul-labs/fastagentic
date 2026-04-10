#!/usr/bin/env python
"""Demo: Wrap an existing agent with zero code changes.

This demo shows how FastAgentic wraps ANY existing agent to make it
production-safe - without modifying the agent code.

We use local-deep-research as our example agent, but this works with
any Python function.

Usage:
    # Run fresh
    python demo.py "quantum computing applications"

    # Resume (returns cached result if complete)
    python demo.py "quantum computing applications" --run-id research-001
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import the existing agent - UNCHANGED
# This is the key point: we don't modify local_deep_research at all
try:
    from local_deep_research import quick_summary
    HAS_LOCAL_DEEP_RESEARCH = True
except ImportError:
    HAS_LOCAL_DEEP_RESEARCH = False
    quick_summary = None

# Import FastAgentic's opaque wrapper
from fastagentic import run_opaque, FileCheckpointStore

console = Console()


def mock_research_agent(query: str, **kwargs) -> dict:
    """Mock agent for demo when local-deep-research is not installed.

    This simulates a long-running research agent.
    """
    import time

    console.print("[dim]  (Using mock agent - install local-deep-research for real LLM)[/dim]")

    # Simulate work
    for i in range(3):
        time.sleep(1)
        console.print(f"[dim]  Researching... ({i+1}/3)[/dim]")

    return {
        "summary": f"""Research Summary: {query}

This is a mock summary demonstrating FastAgentic's checkpoint and resume capability.
In production, this would be the actual research output from local-deep-research.

Key findings:
1. The query "{query}" was processed successfully
2. Results were cached for instant resume
3. No code changes were required to the underlying agent

Install local-deep-research for real LLM-powered research:
  pip install local-deep-research
""",
        "sources": ["mock-source-1", "mock-source-2"],
    }


async def run_demo(query: str, run_id: str | None) -> None:
    """Run the research demo with opaque wrapping."""
    store = FileCheckpointStore(".checkpoints")

    # Choose agent
    agent = quick_summary if HAS_LOCAL_DEEP_RESEARCH else mock_research_agent
    agent_name = "local_deep_research.quick_summary" if HAS_LOCAL_DEEP_RESEARCH else "mock_agent"

    # Generate run_id if not provided
    if run_id is None:
        import uuid
        run_id = f"research-{uuid.uuid4().hex[:6]}"

    # Display header
    console.print()
    console.print(
        Panel(
            f"[bold]Agent:[/bold] {agent_name}\n"
            f"[bold]Query:[/bold] {query}\n"
            f"[bold]Run ID:[/bold] {run_id}\n\n"
            f"[dim]This is an UNMODIFIED existing agent.[/dim]\n"
            f"[dim]FastAgentic wraps it with zero code changes.[/dim]",
            title="FastAgentic Demo: Wrap Any Agent",
            border_style="green",
        )
    )
    console.print()

    # Progress callback
    def on_progress(msg: str) -> None:
        if "cached" in msg.lower() or "skipping" in msg.lower():
            console.print(f"[bold green]✓[/bold green] {msg}")
        elif "complete" in msg.lower():
            console.print(f"[bold green]✓[/bold green] {msg}")
        else:
            console.print(f"[yellow]●[/yellow] {msg}")

    try:
        # THE MAGIC: One line to wrap any agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running agent...", total=None)

            result = await run_opaque(
                agent,
                query=query,
                run_id=run_id,
                store=store,
                on_progress=on_progress,
            )

            progress.remove_task(task)

        # Display result
        console.print()
        summary = result.get("summary", str(result)) if isinstance(result, dict) else str(result)
        console.print(
            Panel(
                summary[:2000],  # Truncate for display
                title="Research Summary",
                border_style="blue",
            )
        )

        console.print()
        console.print(f"[bold green]Done![/bold green] Run ID: [cyan]{run_id}[/cyan]")
        console.print()
        console.print("[dim]To resume this run (returns cached result):[/dim]")
        console.print(f"[cyan]  python demo.py \"{query}\" --run-id {run_id}[/cyan]")

    except KeyboardInterrupt:
        console.print()
        console.print(
            Panel(
                f"[bold yellow]Run interrupted![/bold yellow]\n\n"
                f"The run was not completed, so no result was cached.\n"
                f"Re-run with the same query to try again.",
                title="Interrupted",
                border_style="yellow",
            )
        )
        sys.exit(130)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FastAgentic Demo: Wrap any existing agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py "quantum computing"
  python demo.py "AI safety research" --run-id my-research
  python demo.py "climate change" --run-id climate-001  # Resume
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="quantum entanglement and its applications in computing",
        help="Research query (default: quantum entanglement)",
    )
    parser.add_argument(
        "--run-id",
        metavar="ID",
        help="Run ID for caching/resume",
    )

    args = parser.parse_args()

    # Show setup info
    if not HAS_LOCAL_DEEP_RESEARCH:
        console.print(
            Panel(
                "[yellow]local-deep-research not installed[/yellow]\n\n"
                "Using mock agent for demo. To use real LLM research:\n"
                "  [cyan]pip install local-deep-research[/cyan]\n"
                "  [cyan]export OPENAI_API_KEY=sk-...[/cyan]",
                title="Setup",
                border_style="yellow",
            )
        )
        console.print()

    asyncio.run(run_demo(args.query, args.run_id))


if __name__ == "__main__":
    main()
