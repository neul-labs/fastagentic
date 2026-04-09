"""Interactive Agent CLI for FastAgentic.

Provides a Claude Code / Gemini CLI-like experience for building and testing agents.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Custom theme for agent CLI
AGENT_THEME = Theme(
    {
        "agent": "bold cyan",
        "user": "bold green",
        "tool": "bold yellow",
        "tool_result": "dim yellow",
        "error": "bold red",
        "info": "dim",
        "success": "bold green",
        "thinking": "dim italic",
        "code": "bright_black on grey23",
    }
)

console = Console(theme=AGENT_THEME)


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # user, assistant, tool, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Conversation:
    """A conversation session."""

    id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata: Any) -> Message:
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history for API calls."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role in ("user", "assistant")
        ]

    def save(self, path: Path) -> None:
        """Save conversation to file."""
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Conversation:
        """Load conversation from file."""
        data = json.loads(path.read_text())
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            messages=[Message.from_dict(m) for m in data["messages"]],
        )


@dataclass
class AgentConfig:
    """Configuration for the agent CLI."""

    base_url: str = "http://localhost:8000"
    endpoint: str = "/chat"
    api_key: str | None = None
    timeout: float = 300.0
    stream: bool = True
    show_tools: bool = True
    show_thinking: bool = True
    show_usage: bool = True
    output_format: str = "markdown"  # markdown, plain, json
    history_dir: Path = field(default_factory=lambda: Path.home() / ".fastagentic" / "history")
    max_history: int = 100

    @classmethod
    def load(cls, path: Path | None = None) -> AgentConfig:
        """Load config from file or environment."""
        config_path = path or Path.home() / ".fastagentic" / "config.json"

        config = cls()

        # Load from file
        if config_path.exists():
            data = json.loads(config_path.read_text())
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Override with environment
        if os.environ.get("FASTAGENTIC_URL"):
            config.base_url = os.environ["FASTAGENTIC_URL"]
        if os.environ.get("FASTAGENTIC_API_KEY"):
            config.api_key = os.environ["FASTAGENTIC_API_KEY"]
        if os.environ.get("FASTAGENTIC_ENDPOINT"):
            config.endpoint = os.environ["FASTAGENTIC_ENDPOINT"]

        return config

    def save(self, path: Path | None = None) -> None:
        """Save config to file."""
        config_path = path or Path.home() / ".fastagentic" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "stream": self.stream,
            "show_tools": self.show_tools,
            "show_thinking": self.show_thinking,
            "show_usage": self.show_usage,
            "output_format": self.output_format,
            "max_history": self.max_history,
        }
        config_path.write_text(json.dumps(data, indent=2))


class AgentClient:
    """Client for interacting with FastAgentic agents."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client: Any = None

    async def __aenter__(self) -> AgentClient:
        import httpx

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def invoke(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Invoke agent and get response."""
        url = f"{self.config.base_url}{self.config.endpoint}"

        payload: dict[str, Any] = {
            "message": message,
            "stream": False,
        }
        if history:
            payload["history"] = history

        response = await self._client.post(
            url,
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream response from agent."""

        url = f"{self.config.base_url}{self.config.endpoint}"

        payload: dict[str, Any] = {
            "message": message,
            "stream": True,
        }
        if history:
            payload["history"] = history

        async with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=self._get_headers(),
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def list_endpoints(self) -> list[dict[str, Any]]:
        """List available agent endpoints."""
        url = f"{self.config.base_url}/openapi.json"

        try:
            response = await self._client.get(url)
            if response.status_code == 200:
                spec = response.json()
                endpoints = []
                for path, methods in spec.get("paths", {}).items():
                    for method, details in methods.items():
                        if method.upper() == "POST":
                            endpoints.append(
                                {
                                    "path": path,
                                    "method": method.upper(),
                                    "summary": details.get("summary", ""),
                                    "description": details.get("description", ""),
                                }
                            )
                return endpoints
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    async def health_check(self) -> bool:
        """Check if the agent server is healthy."""
        import httpx

        url = f"{self.config.base_url}/health"
        try:
            response = await self._client.get(url)
            return bool(response.status_code == 200)
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False


class AgentREPL:
    """Interactive REPL for agent conversations."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.conversation: Conversation | None = None
        self.client: AgentClient | None = None
        self._running = False

    def _print_welcome(self) -> None:
        """Print welcome message."""
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]FastAgentic Agent CLI[/bold cyan]\n"
                "Interactive agent testing and development\n\n"
                f"[dim]Server:[/dim] {self.config.base_url}\n"
                f"[dim]Endpoint:[/dim] {self.config.endpoint}",
                title="[bold]Welcome[/bold]",
                border_style="cyan",
            )
        )
        console.print()
        console.print("[dim]Commands:[/dim]")
        console.print("  [bold]/help[/bold]      - Show all commands")
        console.print("  [bold]/quit[/bold]      - Exit the CLI")
        console.print("  [bold]/clear[/bold]     - Clear conversation")
        console.print("  [bold]/save[/bold]      - Save conversation")
        console.print()

    def _print_help(self) -> None:
        """Print help message."""
        table = Table(title="Commands", show_header=True)
        table.add_column("Command", style="bold cyan")
        table.add_column("Description")

        commands = [
            ("/help", "Show this help message"),
            ("/quit, /exit, /q", "Exit the CLI"),
            ("/clear", "Clear conversation history"),
            ("/save [name]", "Save conversation to file"),
            ("/load [name]", "Load conversation from file"),
            ("/history", "Show conversation history"),
            ("/endpoints", "List available agent endpoints"),
            ("/use <endpoint>", "Switch to a different endpoint"),
            ("/config", "Show current configuration"),
            ("/set <key> <value>", "Set configuration option"),
            ("/stream on|off", "Toggle streaming mode"),
            ("/tools on|off", "Toggle tool call display"),
            ("/format md|plain|json", "Set output format"),
            ("/file <path>", "Send file contents as message"),
            ("/export [path]", "Export conversation to file"),
            ("/status", "Check server health"),
            ("/compact", "Compact view (hide metadata)"),
            ("/verbose", "Verbose view (show all metadata)"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        console.print(table)

    def _format_response(self, content: str) -> None:
        """Format and print agent response."""
        if self.config.output_format == "markdown":
            console.print(Markdown(content))
        elif self.config.output_format == "json":
            try:
                data = json.loads(content)
                console.print_json(data=data)
            except json.JSONDecodeError:
                console.print(content)
        else:
            console.print(content)

    def _format_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Format and print a tool call."""
        if not self.config.show_tools:
            return

        console.print()
        console.print(f"[tool]⚙ Tool Call:[/tool] {name}")
        if args:
            console.print(
                Syntax(
                    json.dumps(args, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
            )

    def _format_tool_result(self, _name: str, result: Any) -> None:
        """Format and print a tool result."""
        if not self.config.show_tools:
            return

        console.print("[tool_result]↳ Result:[/tool_result]")
        if isinstance(result, (dict, list)):
            console.print(
                Syntax(
                    json.dumps(result, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
            )
        else:
            console.print(f"  {result}")
        console.print()

    def _format_thinking(self, content: str) -> None:
        """Format and print thinking/reasoning."""
        if not self.config.show_thinking:
            return

        console.print(f"[thinking]💭 {content}[/thinking]")

    def _format_usage(self, usage: dict[str, Any]) -> None:
        """Format and print usage stats."""
        if not self.config.show_usage:
            return

        tokens = usage.get("total_tokens", 0)
        cost = usage.get("cost", 0)
        latency = usage.get("latency_ms", 0)

        console.print()
        console.print(
            f"[info]📊 Tokens: {tokens} | Cost: ${cost:.4f} | Latency: {latency}ms[/info]"
        )

    async def _handle_command(self, command: str) -> bool:
        """Handle a slash command. Returns True if should continue REPL."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[info]Goodbye![/info]")
            return False

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/clear":
            self.conversation = Conversation(id=f"conv-{int(time.time())}")
            console.print("[success]Conversation cleared[/success]")

        elif cmd == "/save":
            if self.conversation:
                name = arg or f"conversation-{int(time.time())}"
                path = self.config.history_dir / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.conversation.save(path)
                console.print(f"[success]Saved to {path}[/success]")
            else:
                console.print("[error]No conversation to save[/error]")

        elif cmd == "/load":
            if arg:
                path = self.config.history_dir / f"{arg}.json"
                if path.exists():
                    self.conversation = Conversation.load(path)
                    console.print(
                        f"[success]Loaded {len(self.conversation.messages)} messages[/success]"
                    )
                else:
                    console.print(f"[error]File not found: {path}[/error]")
            else:
                # List available conversations
                if self.config.history_dir.exists():
                    files = list(self.config.history_dir.glob("*.json"))
                    if files:
                        console.print("[bold]Available conversations:[/bold]")
                        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                            console.print(f"  {f.stem}")
                    else:
                        console.print("[info]No saved conversations[/info]")
                else:
                    console.print("[info]No saved conversations[/info]")

        elif cmd == "/history":
            if self.conversation and self.conversation.messages:
                for msg in self.conversation.messages:
                    role_style = "user" if msg.role == "user" else "agent"
                    console.print(
                        f"[{role_style}]{msg.role}:[/{role_style}] {msg.content[:100]}..."
                    )
            else:
                console.print("[info]No messages in conversation[/info]")

        elif cmd == "/endpoints":
            console.print("[info]Fetching endpoints...[/info]")
            assert self.client is not None
            endpoints = await self.client.list_endpoints()
            if endpoints:
                table = Table(title="Available Endpoints")
                table.add_column("Path", style="cyan")
                table.add_column("Method")
                table.add_column("Description")
                for ep in endpoints:
                    table.add_row(ep["path"], ep["method"], ep.get("summary", ""))
                console.print(table)
            else:
                console.print("[error]Could not fetch endpoints[/error]")

        elif cmd == "/use":
            if arg:
                self.config.endpoint = arg if arg.startswith("/") else f"/{arg}"
                console.print(f"[success]Now using endpoint: {self.config.endpoint}[/success]")
            else:
                console.print("[error]Usage: /use <endpoint>[/error]")

        elif cmd == "/config":
            table = Table(title="Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value")
            table.add_row("base_url", self.config.base_url)
            table.add_row("endpoint", self.config.endpoint)
            table.add_row("api_key", "***" if self.config.api_key else "None")
            table.add_row("stream", str(self.config.stream))
            table.add_row("show_tools", str(self.config.show_tools))
            table.add_row("show_thinking", str(self.config.show_thinking))
            table.add_row("output_format", self.config.output_format)
            console.print(table)

        elif cmd == "/set":
            parts = arg.split(maxsplit=1)
            if len(parts) == 2:
                key, value_str = parts
                if hasattr(self.config, key):
                    # Convert value to appropriate type
                    current = getattr(self.config, key)
                    new_value: Any
                    if isinstance(current, bool):
                        new_value = value_str.lower() in ("true", "1", "yes", "on")
                    elif isinstance(current, (int, float)):
                        new_value = type(current)(value_str)
                    else:
                        new_value = value_str
                    setattr(self.config, key, new_value)
                    console.print(f"[success]Set {key} = {new_value}[/success]")
                else:
                    console.print(f"[error]Unknown setting: {key}[/error]")
            else:
                console.print("[error]Usage: /set <key> <value>[/error]")

        elif cmd == "/stream":
            if arg.lower() in ("on", "true", "1"):
                self.config.stream = True
            elif arg.lower() in ("off", "false", "0"):
                self.config.stream = False
            console.print(f"[success]Streaming: {self.config.stream}[/success]")

        elif cmd == "/tools":
            if arg.lower() in ("on", "true", "1"):
                self.config.show_tools = True
            elif arg.lower() in ("off", "false", "0"):
                self.config.show_tools = False
            console.print(f"[success]Show tools: {self.config.show_tools}[/success]")

        elif cmd == "/format":
            if arg in ("md", "markdown"):
                self.config.output_format = "markdown"
            elif arg == "plain":
                self.config.output_format = "plain"
            elif arg == "json":
                self.config.output_format = "json"
            console.print(f"[success]Output format: {self.config.output_format}[/success]")

        elif cmd == "/file":
            if arg:
                path = Path(arg).expanduser()
                if path.exists():
                    content = path.read_text()
                    console.print(f"[info]Sending file: {path} ({len(content)} chars)[/info]")
                    await self._send_message(content)
                else:
                    console.print(f"[error]File not found: {path}[/error]")
            else:
                console.print("[error]Usage: /file <path>[/error]")

        elif cmd == "/export":
            if self.conversation:
                path = (
                    Path(arg).expanduser() if arg else Path(f"conversation-{int(time.time())}.md")
                )

                lines = [f"# Conversation {self.conversation.id}\n"]
                for msg in self.conversation.messages:
                    lines.append(f"## {msg.role.title()}\n")
                    lines.append(f"{msg.content}\n")

                path.write_text("\n".join(lines))
                console.print(f"[success]Exported to {path}[/success]")
            else:
                console.print("[error]No conversation to export[/error]")

        elif cmd == "/status":
            console.print("[info]Checking server status...[/info]")
            assert self.client is not None
            healthy = await self.client.health_check()
            if healthy:
                console.print("[success]✓ Server is healthy[/success]")
            else:
                console.print("[error]✗ Server is not responding[/error]")

        elif cmd == "/compact":
            self.config.show_tools = False
            self.config.show_thinking = False
            self.config.show_usage = False
            console.print("[success]Compact mode enabled[/success]")

        elif cmd == "/verbose":
            self.config.show_tools = True
            self.config.show_thinking = True
            self.config.show_usage = True
            console.print("[success]Verbose mode enabled[/success]")

        else:
            console.print(f"[error]Unknown command: {cmd}[/error]")
            console.print("[info]Type /help for available commands[/info]")

        return True

    async def _send_message(self, message: str) -> None:
        """Send a message to the agent and display response."""
        if not self.conversation:
            self.conversation = Conversation(id=f"conv-{int(time.time())}")

        # Add user message
        self.conversation.add_message("user", message)

        # Get conversation history
        history = self.conversation.get_history()[:-1]  # Exclude current message

        console.print()

        try:
            if self.config.stream:
                await self._stream_response(message, history)
            else:
                await self._invoke_response(message, history)
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")

    async def _stream_response(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> None:
        """Stream response from agent."""
        full_response = ""
        usage = {}

        assert self.client is not None
        with Live(console=console, refresh_per_second=10) as live:
            buffer = ""

            async for event in self.client.stream(message, history):
                event_type = event.get("type", "")
                data = event.get("data", {})

                if event_type == "token":
                    content = data.get("content", "")
                    buffer += content
                    full_response += content

                    # Update display
                    if self.config.output_format == "markdown":
                        live.update(Markdown(buffer))
                    else:
                        live.update(Text(buffer))

                elif event_type == "tool_call":
                    live.stop()
                    self._format_tool_call(data.get("name", ""), data.get("input", {}))
                    live.start()

                elif event_type == "tool_result":
                    live.stop()
                    self._format_tool_result(data.get("name", ""), data.get("output"))
                    live.start()

                elif event_type == "thinking":
                    live.stop()
                    self._format_thinking(data.get("content", ""))
                    live.start()

                elif event_type == "usage":
                    usage = data

                elif event_type == "done":
                    if "result" in data:
                        full_response = str(data["result"])

        # Add assistant message
        if full_response and self.conversation:
            self.conversation.add_message("assistant", full_response)

        # Show usage
        if usage:
            self._format_usage(usage)

        console.print()

    async def _invoke_response(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> None:
        """Get non-streaming response from agent."""
        assert self.client is not None
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            response = await self.client.invoke(message, history)

        # Extract response
        content = response.get("response", response.get("output", str(response)))

        # Format and display
        self._format_response(content)

        # Add to conversation
        if self.conversation:
            self.conversation.add_message("assistant", content)

        # Show usage
        if "usage" in response:
            self._format_usage(response["usage"])

        console.print()

    async def run(self) -> None:
        """Run the interactive REPL."""
        self._running = True
        self.conversation = Conversation(id=f"conv-{int(time.time())}")

        async with AgentClient(self.config) as client:
            self.client = client

            # Check server health
            healthy = await client.health_check()
            if not healthy:
                console.print("[error]Warning: Server is not responding[/error]")
                console.print(
                    f"[info]Make sure the server is running at {self.config.base_url}[/info]"
                )
                console.print()

            self._print_welcome()

            while self._running:
                try:
                    # Get user input
                    user_input = Prompt.ask("[user]>[/user]")

                    if not user_input.strip():
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        self._running = await self._handle_command(user_input)
                        continue

                    # Send message
                    await self._send_message(user_input)

                except KeyboardInterrupt:
                    console.print("\n[info]Use /quit to exit[/info]")
                except EOFError:
                    break
                except Exception as e:
                    console.print(f"[error]Error: {e}[/error]")


async def run_single_query(
    message: str,
    config: AgentConfig,
    output_file: Path | None = None,
) -> None:
    """Run a single query against the agent."""
    async with AgentClient(config) as client:
        if config.stream:
            full_response = ""

            async for event in client.stream(message):
                event_type = event.get("type", "")
                data = event.get("data", {})

                if event_type == "token":
                    content = data.get("content", "")
                    full_response += content
                    if not output_file:
                        print(content, end="", flush=True)

                elif event_type == "tool_call" and config.show_tools:
                    console.print(f"\n[tool]⚙ {data.get('name')}[/tool]")

                elif event_type == "done":
                    if "result" in data:
                        full_response = str(data["result"])

            print()  # Newline after streaming

            if output_file:
                output_file.write_text(full_response)
        else:
            response = await client.invoke(message)
            content = response.get("response", response.get("output", str(response)))

            if output_file:
                output_file.write_text(content)
            else:
                if config.output_format == "markdown":
                    console.print(Markdown(content))
                elif config.output_format == "json":
                    console.print_json(data=response)
                else:
                    print(content)
