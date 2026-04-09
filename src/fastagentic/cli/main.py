"""FastAgentic CLI - Main entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fastagentic import __version__
from fastagentic.cli.config_cli import config_app

# Import command subgroups at module level to avoid circular imports
from fastagentic.cli.templates import templates_app

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
        bool | None,
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
        typer.Argument(help="Path to the app module (e.g., 'app:app' or 'main:application')"),
    ] = "app:app",
    host: Annotated[str, typer.Option(help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    workers: Annotated[int, typer.Option(help="Number of workers")] = 1,
    server: Annotated[
        str,
        typer.Option(
            "--server",
            "-s",
            help="Server type: uvicorn (default) or gunicorn (production)",
        ),
    ] = "uvicorn",
    max_concurrent: Annotated[
        int | None,
        typer.Option(
            "--max-concurrent",
            help="Maximum concurrent requests (enables backpressure)",
        ),
    ] = None,
    instance_id: Annotated[
        str | None,
        typer.Option(
            "--instance-id",
            help="Instance ID for cluster-aware metrics",
        ),
    ] = None,
    redis_pool_size: Annotated[
        int,
        typer.Option(
            "--redis-pool-size",
            help="Redis connection pool size",
        ),
    ] = 10,
    db_pool_size: Annotated[
        int,
        typer.Option(
            "--db-pool-size",
            help="Database connection pool size",
        ),
    ] = 5,
    db_max_overflow: Annotated[
        int,
        typer.Option(
            "--db-max-overflow",
            help="Database pool max overflow connections",
        ),
    ] = 10,
    timeout_graceful: Annotated[
        int,
        typer.Option(
            "--timeout-graceful",
            help="Graceful shutdown timeout in seconds",
        ),
    ] = 30,
) -> None:
    """Run the FastAgentic application server.

    Supports both development (uvicorn) and production (gunicorn) modes.

    Examples:
        # Development with auto-reload
        fastagentic run --reload

        # Production with Gunicorn (4 workers)
        fastagentic run --server gunicorn --workers 4

        # Production with concurrency limits
        fastagentic run --server gunicorn --workers 4 --max-concurrent 100

        # Cluster deployment with instance ID
        fastagentic run --server gunicorn --instance-id worker-1
    """
    from fastagentic.server.config import PoolConfig, ServerConfig

    # Parse module:attribute format
    if ":" in app_path:
        module_path, attr_name = app_path.rsplit(":", 1)
    else:
        module_path = app_path
        attr_name = "app"

    # Convert file path to module path if needed
    if module_path.endswith(".py"):
        module_path = module_path[:-3].replace("/", ".").replace("\\", ".")

    # Build server configuration
    pool_config = PoolConfig(
        redis_pool_size=redis_pool_size,
        db_pool_size=db_pool_size,
        db_max_overflow=db_max_overflow,
    )

    config = ServerConfig(
        server=server,  # type: ignore
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        max_concurrent=max_concurrent,
        instance_id=instance_id,
        timeout_graceful_shutdown=timeout_graceful,
        pool=pool_config,
    )

    # Display startup info
    console.print("[bold green]Starting FastAgentic server...[/bold green]")
    console.print(f"  App: {module_path}:{attr_name}")
    console.print(f"  Server: {server}")
    console.print(f"  URL: http://{host}:{port}")
    console.print(f"  Workers: {config.effective_workers()}")
    if max_concurrent:
        console.print(f"  Max Concurrent: {max_concurrent}")
    if instance_id:
        console.print(f"  Instance ID: {instance_id}")

    # Set environment variables for the app to pick up
    os.environ["FASTAGENTIC_INSTANCE_ID"] = config.get_instance_id()
    os.environ["FASTAGENTIC_REDIS_POOL_SIZE"] = str(redis_pool_size)
    os.environ["FASTAGENTIC_DB_POOL_SIZE"] = str(db_pool_size)
    os.environ["FASTAGENTIC_DB_MAX_OVERFLOW"] = str(db_max_overflow)
    if max_concurrent:
        os.environ["FASTAGENTIC_MAX_CONCURRENT"] = str(max_concurrent)

    # Run the server
    app_import_path = f"{module_path}:{attr_name}.fastapi"

    if server == "gunicorn":
        from fastagentic.server.runners import run_gunicorn

        run_gunicorn(app_import_path, config)
    else:
        from fastagentic.server.runners import run_uvicorn

        run_uvicorn(app_import_path, config)


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
        str | None,
        typer.Option("--directory", "-d", help="Directory to create project in"),
    ] = None,
) -> None:
    """Create a new FastAgentic project from a template."""
    project_dir = Path(directory or ".") / name

    if project_dir.exists():
        console.print(f"[red]Error: Directory '{project_dir}' already exists[/red]")
        raise typer.Exit(1)

    console.print("[bold green]Creating new FastAgentic project...[/bold green]")
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

    console.print("\n[bold green]Project created successfully![/bold green]")
    console.print("\nNext steps:")
    console.print(f"  cd {project_dir}")
    console.print("  uv sync")
    console.print("  fastagentic run")


def _create_project_files(project_dir: Path, name: str, template: str) -> None:
    """Create project files from template."""
    # Get template-specific content
    template_content = _get_template_content(name, template)

    # pyproject.toml
    (project_dir / "pyproject.toml").write_text(template_content["pyproject"])

    # Main app file
    (project_dir / "app.py").write_text(template_content["app"])

    # Agent/workflow file (if applicable)
    if "agent" in template_content:
        (project_dir / "agent.py").write_text(template_content["agent"])

    # Models
    (project_dir / "models.py").write_text(template_content["models"])

    # CLAUDE.md
    (project_dir / "CLAUDE.md").write_text(template_content["claude_md"])

    # README
    (project_dir / "README.md").write_text(template_content["readme"])

    # .env.example
    (project_dir / ".env.example").write_text(template_content["env_example"])

    # .gitignore
    gitignore = """__pycache__/
*.py[cod]
*$py.class
.env
.venv
venv/
.uv/
"""
    (project_dir / ".gitignore").write_text(gitignore)


def _get_template_content(name: str, template: str) -> dict[str, str]:
    """Get template-specific content based on framework."""

    # Common pyproject.toml
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

[tool.pytest.ini_options]
asyncio_mode = "auto"
'''

    # Common .env.example
    env_example = f"""# {name} - Environment Variables

# LLM Provider (choose one)
OPENAI_API_KEY=sk-your-openai-key
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# FastAgentic
FASTAGENTIC_ENV=dev
FASTAGENTIC_LOG_LEVEL=INFO

# Optional: Durable store
# REDIS_URL=redis://localhost:6379
"""

    if template == "pydanticai":
        return _get_pydanticai_template(name, pyproject, env_example)
    elif template == "langgraph":
        return _get_langgraph_template(name, pyproject, env_example)
    elif template == "crewai":
        return _get_crewai_template(name, pyproject, env_example)
    elif template == "langchain":
        return _get_langchain_template(name, pyproject, env_example)
    else:
        return _get_pydanticai_template(name, pyproject, env_example)


def _get_pydanticai_template(name: str, pyproject: str, env_example: str) -> dict[str, str]:
    """Generate PydanticAI template."""
    return {
        "pyproject": pyproject,
        "env_example": env_example,
        "app": f'''"""FastAgentic application with PydanticAI."""

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from pydantic import BaseModel

from agent import chat_agent
from models import ChatRequest, ChatResponse

app = App(
    title="{name}",
    version="0.1.0",
)


@tool(name="get_time", description="Get current time")
async def get_time() -> str:
    """Return current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()


@resource(name="info", uri="info")
async def get_info() -> dict:
    """Return service info."""
    return {{"name": "{name}", "version": "0.1.0"}}


@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(chat_agent),
    input_model=ChatRequest,
    output_model=ChatResponse,
    stream=True,
    mcp_tool="chat",
)
async def chat(request: ChatRequest) -> ChatResponse:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
''',
        "agent": '''"""PydanticAI agent definition."""

from datetime import datetime
from pydantic_ai import Agent

SYSTEM_PROMPT = """You are a helpful AI assistant.
Be concise and helpful in your responses."""

chat_agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
)


@chat_agent.tool
async def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
''',
        "models": '''"""API models."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1)
    stream: bool = True


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
''',
        "claude_md": f"""# {name} - Claude Code Guide

PydanticAI agent deployed with FastAgentic.

## Commands

```bash
uv sync                           # Install dependencies
uv run fastagentic run            # Start server
uv run fastagentic agent chat     # Test interactively
```

## Structure

- `app.py` - FastAgentic application
- `agent.py` - PydanticAI agent definition
- `models.py` - Request/response models

## Modifying

- **Add tools**: Edit `agent.py`, use `@chat_agent.tool`
- **Change model**: Edit `Agent(model="...")` in `agent.py`
- **Add endpoints**: Use `@agent_endpoint` in `app.py`
""",
        "readme": f"""# {name}

A PydanticAI agent deployed with FastAgentic.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run fastagentic run
```

## Test

```bash
uv run fastagentic agent chat
```

See `CLAUDE.md` for detailed instructions.
""",
    }


def _get_langgraph_template(name: str, pyproject: str, env_example: str) -> dict[str, str]:
    """Generate LangGraph template."""
    pyproject = pyproject.replace(
        "fastagentic[langgraph]", 'fastagentic[langgraph]",\n    "langchain-openai>=0.2.0'
    )
    return {
        "pyproject": pyproject,
        "env_example": env_example,
        "app": f'''"""FastAgentic application with LangGraph."""

from fastagentic import App, agent_endpoint, resource
from fastagentic.adapters.langgraph import LangGraphAdapter
from pydantic import BaseModel

from agent import workflow
from models import WorkflowRequest, WorkflowResponse

app = App(
    title="{name}",
    version="0.1.0",
)


@resource(name="info", uri="info")
async def get_info() -> dict:
    return {{"name": "{name}", "type": "langgraph-workflow"}}


@agent_endpoint(
    path="/run",
    runnable=LangGraphAdapter(workflow),
    input_model=WorkflowRequest,
    output_model=WorkflowResponse,
    stream=True,
    mcp_tool="run_workflow",
)
async def run_workflow(request: WorkflowRequest) -> WorkflowResponse:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
''',
        "agent": '''"""LangGraph workflow definition."""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class State(BaseModel):
    """Workflow state."""
    input: str = ""
    output: str = ""


llm = ChatOpenAI(model="gpt-4o-mini")


async def process_node(state: State) -> dict:
    """Process the input."""
    response = await llm.ainvoke(state.input)
    return {"output": response.content}


# Build graph
graph = StateGraph(State)
graph.add_node("process", process_node)
graph.set_entry_point("process")
graph.add_edge("process", END)

workflow = graph.compile()
''',
        "models": '''"""API models."""

from pydantic import BaseModel, Field


class WorkflowRequest(BaseModel):
    """Workflow request."""
    input: str = Field(..., min_length=1)


class WorkflowResponse(BaseModel):
    """Workflow response."""
    output: str
''',
        "claude_md": f"""# {name} - Claude Code Guide

LangGraph workflow deployed with FastAgentic.

## Commands

```bash
uv sync
uv run fastagentic run
uv run fastagentic agent chat --endpoint /run
```

## Structure

- `app.py` - FastAgentic application
- `agent.py` - LangGraph workflow definition
- `models.py` - Request/response models

## Modifying

- **Add nodes**: Create function, add with `graph.add_node()`
- **Add edges**: Use `graph.add_edge()` or `add_conditional_edges()`
- **Change state**: Update `State` class
""",
        "readme": f"""# {name}

A LangGraph workflow deployed with FastAgentic.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run fastagentic run
```

See `CLAUDE.md` for details.
""",
    }


def _get_crewai_template(name: str, pyproject: str, env_example: str) -> dict[str, str]:
    """Generate CrewAI template."""
    return {
        "pyproject": pyproject,
        "env_example": env_example,
        "app": f'''"""FastAgentic application with CrewAI."""

from fastagentic import App, agent_endpoint, resource
from fastagentic.adapters.crewai import CrewAIAdapter
from pydantic import BaseModel

from agent import crew
from models import CrewRequest, CrewResponse

app = App(
    title="{name}",
    version="0.1.0",
)


@resource(name="info", uri="info")
async def get_info() -> dict:
    return {{"name": "{name}", "type": "crewai-multi-agent"}}


@agent_endpoint(
    path="/run",
    runnable=CrewAIAdapter(crew),
    input_model=CrewRequest,
    output_model=CrewResponse,
    stream=True,
    mcp_tool="run_crew",
)
async def run_crew(request: CrewRequest) -> CrewResponse:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
''',
        "agent": '''"""CrewAI crew definition."""

from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Research the given topic thoroughly",
    backstory="You are an expert researcher.",
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write clear, engaging content",
    backstory="You are a skilled writer.",
    verbose=True,
)

# Define tasks
research_task = Task(
    description="Research: {topic}",
    expected_output="Research findings",
    agent=researcher,
)

write_task = Task(
    description="Write summary based on research",
    expected_output="Written summary",
    agent=writer,
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)
''',
        "models": '''"""API models."""

from pydantic import BaseModel, Field


class CrewRequest(BaseModel):
    """Crew request."""
    topic: str = Field(..., min_length=1)


class CrewResponse(BaseModel):
    """Crew response."""
    result: str
''',
        "claude_md": f"""# {name} - Claude Code Guide

CrewAI multi-agent system deployed with FastAgentic.

## Commands

```bash
uv sync
uv run fastagentic run
uv run fastagentic agent chat --endpoint /run
```

## Structure

- `app.py` - FastAgentic application
- `agent.py` - CrewAI agents, tasks, and crew
- `models.py` - Request/response models

## Modifying

- **Add agents**: Create `Agent()` in `agent.py`
- **Add tasks**: Create `Task()` and assign to agent
- **Change process**: Use `Process.sequential` or `Process.hierarchical`
""",
        "readme": f"""# {name}

A CrewAI multi-agent system deployed with FastAgentic.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run fastagentic run
```

See `CLAUDE.md` for details.
""",
    }


def _get_langchain_template(name: str, pyproject: str, env_example: str) -> dict[str, str]:
    """Generate LangChain template."""
    pyproject = pyproject.replace(
        "fastagentic[langchain]", 'fastagentic[langchain]",\n    "langchain-openai>=0.2.0'
    )
    return {
        "pyproject": pyproject,
        "env_example": env_example,
        "app": f'''"""FastAgentic application with LangChain."""

from fastagentic import App, agent_endpoint, resource
from fastagentic.adapters.langchain import LangChainAdapter
from pydantic import BaseModel

from agent import chain
from models import ChainRequest, ChainResponse

app = App(
    title="{name}",
    version="0.1.0",
)


@resource(name="info", uri="info")
async def get_info() -> dict:
    return {{"name": "{name}", "type": "langchain"}}


@agent_endpoint(
    path="/run",
    runnable=LangChainAdapter(chain),
    input_model=ChainRequest,
    output_model=ChainResponse,
    stream=True,
    mcp_tool="run_chain",
)
async def run_chain(request: ChainRequest) -> ChainResponse:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
''',
        "agent": '''"""LangChain chain definition."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()
''',
        "models": '''"""API models."""

from pydantic import BaseModel, Field


class ChainRequest(BaseModel):
    """Chain request."""
    input: str = Field(..., min_length=1)


class ChainResponse(BaseModel):
    """Chain response."""
    output: str
''',
        "claude_md": f"""# {name} - Claude Code Guide

LangChain application deployed with FastAgentic.

## Commands

```bash
uv sync
uv run fastagentic run
uv run fastagentic agent chat --endpoint /run
```

## Structure

- `app.py` - FastAgentic application
- `agent.py` - LangChain chain definition
- `models.py` - Request/response models

## Modifying

- **Change chain**: Edit LCEL in `agent.py`
- **Add tools**: Use `create_tool_calling_agent()`
- **Change prompt**: Edit `ChatPromptTemplate`
""",
        "readme": f"""# {name}

A LangChain application deployed with FastAgentic.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run fastagentic run
```

See `CLAUDE.md` for details.
""",
    }


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


@app.command()
def inspect(
    list_type: Annotated[
        str | None,
        typer.Option(
            "--list",
            "-l",
            help="List registered items (tools, resources, prompts, agents, endpoints)",
        ),
    ] = None,
    schema: Annotated[
        str | None,
        typer.Option("--schema", "-s", help="Show schema for a specific item"),
    ] = None,
    config: Annotated[
        bool,
        typer.Option("--config", "-c", help="Show configuration"),
    ] = False,
    _json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON (reserved)"),
    ] = False,
) -> None:
    """Inspect registered decorators, schemas, and configuration.

    Use --list to see all registered items, --schema to see details of a specific item.
    """
    from fastagentic.decorators import get_endpoints, get_prompts, get_resources, get_tools

    # Show configuration
    if config:
        import os

        console.print("[bold]Configuration[/bold]\n")
        env_vars = [
            ("FASTAGENTIC_ENV", "Environment"),
            ("FASTAGENTIC_SERVER", "Server"),
            ("FASTAGENTIC_HOST", "Host"),
            ("FASTAGENTIC_PORT", "Port"),
            ("FASTAGENTIC_WORKERS", "Workers"),
            ("FASTAGENTIC_LOG_LEVEL", "Log Level"),
        ]
        table = Table()
        table.add_column("Variable", style="cyan")
        table.add_column("Value", style="green")
        for var, _name in env_vars:
            val = os.environ.get(var, "[yellow]not set[/yellow]")
            table.add_row(var, val)
        console.print(table)
        return

    # Show schema for specific item
    if schema:
        tools = get_tools()
        resources = get_resources()
        prompts = get_prompts()

        if schema in tools:
            tool_defn, _ = tools[schema]
            console.print(f"[bold]Tool: {schema}[/bold]")
            console.print(f"Description: {tool_defn.description}")
            if tool_defn.inputSchema:
                console.print("Input Schema:")
                import json

                console.print(json.dumps(tool_defn.inputSchema, indent=2))
        elif schema in resources:
            resource_defn, _ = resources[schema]
            console.print(f"[bold]Resource: {schema}[/bold]")
            console.print(f"URI: {resource_defn.uri}")
            console.print(f"MIME Type: {resource_defn.mimeType}")
        elif schema in prompts:
            prompt_defn, _ = prompts[schema]
            console.print(f"[bold]Prompt: {schema}[/bold]")
            console.print(f"Description: {prompt_defn.description}")
        else:
            console.print(f"[red]Item '{schema}' not found[/red]")
        return

    # List registered items
    if list_type:
        if list_type == "tools":
            tools = get_tools()
            console.print(f"[bold]Tools ({len(tools)})[/bold]\n")
            table = Table()
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            for name, (tool_defn, _) in tools.items():
                desc = (
                    tool_defn.description[:50] + "..."
                    if tool_defn.description and len(tool_defn.description) > 50
                    else tool_defn.description or ""
                )
                table.add_row(name, desc or "-")
            console.print(table)
        elif list_type == "resources":
            resources = get_resources()
            console.print(f"[bold]Resources ({len(resources)})[/bold]\n")
            table = Table()
            table.add_column("Name", style="cyan")
            table.add_column("URI", style="green")
            for name, (resource_defn, _) in resources.items():
                table.add_row(name, resource_defn.uri)
            console.print(table)
        elif list_type == "prompts":
            prompts = get_prompts()
            console.print(f"[bold]Prompts ({len(prompts)})[/bold]\n")
            table = Table()
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            for name, (prompt_defn, _) in prompts.items():
                desc = (
                    prompt_defn.description[:50] + "..."
                    if prompt_defn.description and len(prompt_defn.description) > 50
                    else prompt_defn.description or ""
                )
                table.add_row(name, desc or "-")
            console.print(table)
        elif list_type == "agents" or list_type == "endpoints":
            endpoints = get_endpoints()
            console.print(f"[bold]Agent Endpoints ({len(endpoints)})[/bold]\n")
            table = Table()
            table.add_column("Path", style="cyan")
            table.add_column("Skill", style="green")
            table.add_column("Stream", style="blue")
            for path, (endpoint_defn, _) in endpoints.items():
                skill = endpoint_defn.a2a_skill or "-"
                stream = "Yes" if endpoint_defn.stream else "No"
                table.add_row(path, skill, stream)
            console.print(table)
        else:
            console.print(f"[red]Unknown type: {list_type}[/red]")
            console.print("Use: tools, resources, prompts, agents, or endpoints")
        return

    # Default: show summary
    console.print(f"[bold blue]FastAgentic Application Inspection[/bold blue] v{__version__}\n")

    tools = get_tools()
    resources = get_resources()
    prompts = get_prompts()
    endpoints = get_endpoints()

    table = Table(title="Registered Items")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Tools", str(len(tools)))
    table.add_row("Resources", str(len(resources)))
    table.add_row("Prompts", str(len(prompts)))
    table.add_row("Agent Endpoints", str(len(endpoints)))

    console.print(table)

    console.print("\n[dim]Use --list <type> to see details, --schema <name> for schema[/dim]")


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
            getattr(module, attr_name)

            # Get registered items
            from fastagentic.decorators import get_endpoints, get_prompts, get_resources, get_tools

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

            for name, (tool_defn, _) in tools.items():
                status = "[green]PASS[/green]" if tool_defn.name else "[red]FAIL[/red]"
                table.add_row("Tool", name, status)

            for name, (resource_defn, _) in resources.items():
                status = "[green]PASS[/green]" if resource_defn.name else "[red]FAIL[/red]"
                table.add_row("Resource", name, status)

            for name, (prompt_defn, _) in prompts.items():
                status = "[green]PASS[/green]" if prompt_defn.name else "[red]FAIL[/red]"
                table.add_row("Prompt", name, status)

            for path, (endpoint_defn, _) in endpoints.items():
                status = "[green]PASS[/green]" if endpoint_defn.path else "[red]FAIL[/red]"
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
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Validate MCP schema compliance."""
    from fastagentic.decorators import get_prompts, get_resources, get_tools

    console.print("[bold blue]Validating MCP schema...[/bold blue]")

    errors: list[str] = []
    warnings: list[str] = []

    # Validate tools
    tools = get_tools()
    for name, (tool_defn, _) in tools.items():
        if not tool_defn.name:
            errors.append(f"Tool '{name}' is missing name")
        if not tool_defn.description:
            warnings.append(f"Tool '{name}' is missing description")
        if not tool_defn.inputSchema:
            errors.append(f"Tool '{name}' is missing inputSchema")

    # Validate resources
    resources = get_resources()
    for name, (resource_defn, _) in resources.items():
        if not resource_defn.uri:
            errors.append(f"Resource '{name}' is missing URI")
        if not resource_defn.name:
            errors.append(f"Resource '{name}' is missing name")

    # Validate prompts
    prompts = get_prompts()
    for name, (prompt_defn, _) in prompts.items():
        if not prompt_defn.name:
            errors.append(f"Prompt '{name}' is missing name")
        if not prompt_defn.description:
            warnings.append(f"Prompt '{name}' is missing description")

    # Report results
    if errors:
        console.print("[red]Validation failed:[/red]")
        for error in errors:
            console.print(f"  [red]ERROR: {error}[/red]")
        raise typer.Exit(1)

    if warnings:
        console.print("[yellow]Validation passed with warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]WARNING: {warning}[/yellow]")
    else:
        console.print("[green]MCP schema validation passed[/green]")

    console.print("\nSummary:")
    console.print(f"  Tools: {len(tools)}")
    console.print(f"  Resources: {len(resources)}")
    console.print(f"  Prompts: {len(prompts)}")


@mcp_app.command("schema")
def mcp_schema(
    _app_path: Annotated[
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
    for name, (tool_defn, _) in tools.items():
        console.print(f"  - {name}: {tool_defn.description[:50]}...")

    console.print(f"\nResources: {len(resources)}")
    for name, (resource_defn, _) in resources.items():
        console.print(f"  - {name}: {resource_defn.uri}")

    console.print(f"\nPrompts: {len(prompts)}")
    for name, (prompt_defn, _) in prompts.items():
        console.print(
            f"  - {name}: {prompt_defn.description[:50] if prompt_defn.description else 'No description'}..."
        )


@mcp_app.command("export")
def mcp_export(
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("mcp_manifest.json"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (json, yaml)"),
    ] = "json",
) -> None:
    """Export MCP manifest to a file."""
    import json

    from fastagentic.decorators import get_prompts, get_resources, get_tools

    tools = get_tools()
    resources = get_resources()
    prompts = get_prompts()

    # Build manifest
    manifest = {
        "name": "FastAgentic Server",
        "version": "1.0.0",
        "tools": [
            {
                "name": name,
                "description": tool_defn.description,
                "inputSchema": tool_defn.inputSchema,
            }
            for name, (tool_defn, _) in tools.items()
        ],
        "resources": [
            {
                "name": name,
                "uri": resource_defn.uri,
                "description": resource_defn.description,
                "mimeType": resource_defn.mimeType,
            }
            for name, (resource_defn, _) in resources.items()
        ],
        "prompts": [
            {
                "name": name,
                "description": prompt_defn.description,
                "arguments": [
                    {"name": arg.name, "description": arg.description, "required": arg.required}
                    for arg in (prompt_defn.arguments or [])
                ]
                if prompt_defn.arguments
                else [],
            }
            for name, (prompt_defn, _) in prompts.items()
        ],
    }

    # Write output
    if format == "yaml":
        try:
            import yaml

            output.write_text(yaml.dump(manifest, default_flow_style=False))
        except ImportError:
            console.print("[red]PyYAML not installed. Use --format json or install pyyaml.[/red]")
            raise typer.Exit(1)
    else:
        output.write_text(json.dumps(manifest, indent=2))

    console.print(f"[green]MCP manifest exported to: {output}[/green]")
    console.print(f"  Tools: {len(tools)}")
    console.print(f"  Resources: {len(resources)}")
    console.print(f"  Prompts: {len(prompts)}")


@mcp_app.command("call")
def mcp_call(
    tool_name: Annotated[
        str,
        typer.Argument(help="Name of the tool to call"),
    ],
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
    input_json: Annotated[
        str | None,
        typer.Option("--input", "-i", help="Input JSON string"),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Input JSON file"),
    ] = None,
) -> None:
    """Call an MCP tool directly."""
    import json

    from fastagentic.decorators import get_tools

    tools = get_tools()

    if tool_name not in tools:
        console.print(f"[red]Tool '{tool_name}' not found[/red]")
        console.print(f"Available tools: {', '.join(tools.keys())}")
        raise typer.Exit(1)

    # Parse input
    input_data = {}
    if input_json:
        try:
            input_data = json.loads(input_json)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON input: {e}[/red]")
            raise typer.Exit(1)
    elif input_file:
        try:
            input_data = json.loads(input_file.read_text())
        except FileNotFoundError:
            console.print(f"[red]File not found: {input_file}[/red]")
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in file: {e}[/red]")
            raise typer.Exit(1)

    # Call tool
    tool_defn, tool_fn = tools[tool_name]
    console.print(f"[bold]Calling tool: {tool_name}[/bold]")
    console.print(f"Input: {json.dumps(input_data, indent=2)}")

    try:
        import asyncio

        result = asyncio.run(tool_fn(**input_data))
        console.print("\n[bold]Result:[/bold]")
        if isinstance(result, str):
            console.print(result)
        else:
            console.print(json.dumps(result, indent=2, default=str))
    except TypeError as e:
        console.print(f"[red]Invalid arguments: {e}[/red]")
        console.print(f"Expected input schema: {json.dumps(tool_defn.inputSchema, indent=2)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error calling tool: {e}[/red]")
        raise typer.Exit(1)


@mcp_app.command("stdio")
def mcp_stdio(
    app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Run interactive stdio session for MCP.

    This is an alias for 'mcp serve' for compatibility.
    """
    # Re-use the serve command logic
    import asyncio

    # Parse module:attribute format
    if ":" in app_path:
        module_path, attr_name = app_path.rsplit(":", 1)
    else:
        module_path = app_path
        attr_name = "app"

    try:
        spec = importlib.util.spec_from_file_location(
            module_path, f"{module_path.replace('.', '/')}.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            app_instance = getattr(module, attr_name)

            from fastagentic.protocols.mcp_stdio import serve_stdio

            console.print("[bold green]Starting MCP stdio session...[/bold green]")
            console.print("[dim]Press Ctrl+C to exit[/dim]")
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
        str | None,
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
        str | None,
        typer.Option("--api-key", "-k", help="API key for authentication"),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Enable streaming responses"),
    ] = True,
    output: Annotated[
        str | None,
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
        str | None,
        typer.Option("--url", "-u", help="Set default server URL"),
    ] = None,
    set_endpoint: Annotated[
        str | None,
        typer.Option("--endpoint", "-e", help="Set default endpoint"),
    ] = None,
    set_api_key: Annotated[
        str | None,
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
    _list_all: Annotated[
        bool,
        typer.Option("--list", "-l", help="List saved conversations"),
    ] = False,
    load: Annotated[
        str | None,
        typer.Option("--load", help="Load and display a conversation"),
    ] = None,
    delete: Annotated[
        str | None,
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
        except (json.JSONDecodeError, OSError):
            table.add_row(f.stem, "?", "?")

    console.print(table)


# A2A commands
a2a_app = typer.Typer(help="A2A protocol commands")
app.add_typer(a2a_app, name="a2a")


@a2a_app.command("validate")
def a2a_validate(
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Validate A2A Agent Card compliance."""
    from fastagentic.decorators import get_endpoints

    console.print("[bold blue]Validating A2A Agent Card...[/bold blue]")

    errors: list[str] = []
    warnings: list[str] = []

    endpoints = get_endpoints()

    # Check endpoints with A2A skills
    skills_seen: set[str] = set()
    for path, (defn, _) in endpoints.items():
        if defn.a2a_skill:
            skill = defn.a2a_skill
            if skill in skills_seen:
                errors.append(f"Duplicate skill name: '{skill}'")
            skills_seen.add(skill)

            if not defn.description:
                warnings.append(f"Endpoint '{path}' with skill '{skill}' is missing description")

    # Check for required A2A endpoint fields
    for path, (defn, _) in endpoints.items():
        if defn.a2a_skill and not defn.http_method:
            errors.append(f"Endpoint '{path}' is missing HTTP method")

    # Report results
    if errors:
        console.print("[red]Validation failed:[/red]")
        for error in errors:
            console.print(f"  [red]ERROR: {error}[/red]")
        raise typer.Exit(1)

    if warnings:
        console.print("[yellow]Validation passed with warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]WARNING: {warning}[/yellow]")
    else:
        console.print("[green]A2A validation passed[/green]")

    console.print("\nSummary:")
    console.print(f"  Endpoints with A2A skills: {len(skills_seen)}")
    console.print(f"  Skills: {', '.join(sorted(skills_seen))}")


@a2a_app.command("card")
def a2a_card(
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """Print the A2A Agent Card."""
    from fastagentic.decorators import get_endpoints

    endpoints = get_endpoints()

    skills = []
    for _path, (defn, _) in endpoints.items():
        if defn.a2a_skill:
            skills.append(defn.a2a_skill)

    console.print("[bold]A2A Agent Card[/bold]")
    console.print(f"\nSkills: {len(skills)}")
    for skill in skills:
        console.print(f"  - {skill}")


@a2a_app.command("list")
def a2a_list(
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
) -> None:
    """List registered A2A skills."""
    from fastagentic.decorators import get_endpoints

    endpoints = get_endpoints()

    console.print("[bold]Registered A2A Skills[/bold]\n")

    skills_found = False
    table = Table()
    table.add_column("Skill", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Description", style="blue")
    table.add_column("Stream", style="magenta")

    for path, (defn, _) in endpoints.items():
        if defn.a2a_skill:
            skills_found = True
            desc = defn.description[:50] if defn.description else "-"
            stream = "Yes" if defn.stream else "No"
            table.add_row(defn.a2a_skill, path, desc, stream)

    if skills_found:
        console.print(table)
    else:
        console.print("[yellow]No A2A skills registered[/yellow]")
        console.print("[dim]Add a2a_skill to agent_endpoint decorators to register skills[/dim]")


@a2a_app.command("export")
def a2a_export(
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("agent_card.json"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (json, yaml)"),
    ] = "json",
) -> None:
    """Export A2A Agent Card to a file."""
    import json

    from fastagentic.decorators import get_endpoints

    endpoints = get_endpoints()

    # Build agent card
    skills = []
    for path, (defn, _) in endpoints.items():
        if defn.a2a_skill:
            skills.append(
                {
                    "id": defn.a2a_skill,
                    "name": defn.a2a_skill,
                    "description": defn.description,
                    "url": f"http://localhost:8000{path}",
                    "capabilities": {
                        "streaming": defn.stream,
                    },
                }
            )

    agent_card = {
        "name": "FastAgentic Server",
        "version": "1.0.0",
        "skills": skills,
    }

    # Write output
    if format == "yaml":
        try:
            import yaml

            output.write_text(yaml.dump(agent_card, default_flow_style=False))
        except ImportError:
            console.print("[red]PyYAML not installed. Use --format json or install pyyaml.[/red]")
            raise typer.Exit(1)
    else:
        output.write_text(json.dumps(agent_card, indent=2))

    console.print(f"[green]Agent Card exported to: {output}[/green]")
    console.print(f"  Skills: {len(skills)}")


@a2a_app.command("ping")
def a2a_ping(
    url: Annotated[
        str,
        typer.Argument(help="Agent URL to ping"),
    ],
) -> None:
    """Check connectivity to an external A2A agent."""
    import json

    console.print(f"[bold]Pinging A2A agent: {url}[/bold]\n")

    try:
        import httpx

        # Get agent card
        card_url = f"{url.rstrip('/')}/.well-known/agent.json"

        with httpx.Client(timeout=10.0) as client:
            response = client.get(card_url)
            if response.status_code == 200:
                agent_card = response.json()
                console.print("[green]Agent found![/green]\n")
                console.print(f"Name: {agent_card.get('name', 'Unknown')}")
                console.print(f"Version: {agent_card.get('version', 'Unknown')}")
                skills = agent_card.get("skills", [])
                console.print(f"Skills: {len(skills)}")
                for skill in skills[:5]:
                    console.print(
                        f"  - {skill.get('id')}: {skill.get('description', 'No description')[:50]}"
                    )
                if len(skills) > 5:
                    console.print(f"  ... and {len(skills) - 5} more")
            else:
                console.print(f"[red]Agent not responding: HTTP {response.status_code}[/red]")
                raise typer.Exit(1)
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError:
        console.print("[red]Invalid agent card format[/red]")
        raise typer.Exit(1)


@a2a_app.command("invoke")
def a2a_invoke(
    skill: Annotated[
        str,
        typer.Argument(help="Skill ID to invoke"),
    ],
    _app_path: Annotated[
        str,
        typer.Argument(help="Path to the app module"),
    ] = "app:app",
    input_json: Annotated[
        str | None,
        typer.Option("--input", "-i", help="Input JSON string"),
    ] = None,
    _stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Enable streaming response"),
    ] = False,
) -> None:
    """Invoke a local A2A skill."""
    import json

    from fastagentic.decorators import get_endpoints

    endpoints = get_endpoints()

    # Find the endpoint with matching skill
    target_endpoint = None
    for path, (defn, _) in endpoints.items():
        if defn.a2a_skill == skill:
            target_endpoint = (path, defn)
            break

    if not target_endpoint:
        console.print(f"[red]Skill '{skill}' not found[/red]")
        console.print("Available skills:")
        for path, (defn, _) in endpoints.items():
            if defn.a2a_skill:
                console.print(f"  - {defn.a2a_skill} at {path}")
        raise typer.Exit(1)

    path, defn = target_endpoint

    # Parse input
    input_data = {}
    if input_json:
        try:
            input_data = json.loads(input_json)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON input: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Invoking skill: {skill}[/bold]")
    console.print(f"Endpoint: {path}")
    console.print(f"Input: {json.dumps(input_data, indent=2)}\n")

    # For now, just show how to invoke (actual invocation requires HTTP call)
    console.print("[yellow]Local skill invocation requires HTTP server running[/yellow]")
    console.print(f"\nTo invoke this skill, make a POST request to: http://localhost:8000{path}")
    console.print("Headers: Content-Type: application/json")
    console.print(f"Body: {json.dumps(input_data, indent=2)}")


# Register command subgroups
app.add_typer(templates_app, name="templates")
app.add_typer(config_app, name="config")

if __name__ == "__main__":
    app()
