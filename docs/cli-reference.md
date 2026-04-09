# CLI Reference

The FastAgentic CLI streamlines project scaffolding, local development, and testing. This reference documents each command and its key options.

## Global Options

| Option | Description |
| ------ | ----------- |
| `--version, -v` | Show version and exit |
| `--help` | Show help message |

## `fastagentic new`

Scaffold a new application from a template.

```bash
fastagentic new my-service --template pydanticai
```

| Option | Description |
| ------ | ----------- |
| `--template, -t` | Template to use (`pydanticai`, `langgraph`, `crewai`, `langchain`) |
| `--directory, -d` | Directory to create project in |

Generated structure:

- `app.py` with an `App` instance
- `models.py` for Pydantic schemas
- `agent.py` for the agent/workflow
- `tests/` for tests
- `config/` for configuration

## `fastagentic run`

Launches the ASGI application server. Supports both development (Uvicorn) and production (Gunicorn) modes.

```bash
# Development with auto-reload
fastagentic run --reload

# Production with Gunicorn
fastagentic run --server gunicorn --workers 4

# Production with concurrency limits
fastagentic run --server gunicorn --workers 4 --max-concurrent 100
```

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `app_path` | `app:app` | Path to the app module (e.g., `app:app` or `main:application`) |
| `--host` | `127.0.0.1` | Bind address for the server |
| `--port` | `8000` | Port for HTTP traffic |
| `--server, -s` | `uvicorn` | Server type: `uvicorn` or `gunicorn` |
| `--workers` | `1` | Number of worker processes |
| `--reload` | `false` | Enable auto-reload (dev only, forces 1 worker) |

### Scalability Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-concurrent` | unlimited | Maximum concurrent requests per worker |
| `--instance-id` | auto | Instance ID for cluster metrics |

### Connection Pool Options

| Option | Default | Description |
|--------|---------|-------------|
| `--redis-pool-size` | `10` | Redis connection pool size per worker |
| `--db-pool-size` | `5` | Database connection pool size per worker |
| `--db-max-overflow` | `10` | Database pool overflow connections |

### Examples

```bash
# Development
fastagentic run --reload

# Staging
fastagentic run --server gunicorn --workers 2 --max-concurrent 50

# Production
fastagentic run \
  --server gunicorn \
  --workers 4 \
  --max-concurrent 100 \
  --redis-pool-size 20 \
  --instance-id worker-1
```

See the [Scaling Guide](scaling.md) for detailed production deployment documentation.

## `fastagentic info`

Display information about the current FastAgentic application.

```bash
fastagentic info
```

## `fastagentic inspect`

Inspect registered decorators, schemas, and configuration.

```bash
# Show summary of all registered items
fastagentic inspect

# List specific item types
fastagentic inspect --list tools
fastagentic inspect --list resources
fastagentic inspect --list prompts
fastagentic inspect --list agents

# Show schema for a specific item
fastagentic inspect --schema get_time

# Show configuration
fastagentic inspect --config
```

| Option | Description |
|--------|-------------|
| `--list, -l` | List items: `tools`, `resources`, `prompts`, `agents` |
| `--schema, -s` | Show schema for a specific item by name |
| `--config, -c` | Show current configuration |
| `--json, -j` | Output as JSON |

## `fastagentic templates`

Manage project templates.

```bash
# List all available templates
fastagentic templates list

# List by category
fastagentic templates list --category official
fastagentic templates list --category community

# Search for templates
fastagentic templates search "agent"

# Show template details
fastagentic templates info pydanticai

# Refresh template cache
fastagentic templates refresh
```

| Subcommand | Description |
|------------|-------------|
| `list [--category]` | List available templates |
| `search <query>` | Search templates by name or description |
| `info <name>` | Show details about a template |
| `refresh` | Refresh template cache from remote |

## `fastagentic config`

Manage FastAgentic configuration.

```bash
# Show current configuration
fastagentic config show

# Show with secrets (masked)
fastagentic config show --secrets

# Validate a configuration file
fastagentic config validate config/settings.yaml

# Generate default configuration
fastagentic config init -o settings.yaml
```

| Subcommand | Description |
|------------|-------------|
| `show [--secrets]` | Display resolved configuration |
| `validate <path>` | Validate a configuration file |
| `init -o <path>` | Generate default configuration file |

## `fastagentic test contract`

Validates that REST and MCP schemas remain in sync.

```bash
fastagentic test contract
```

| Option | Description |
| ------ | ----------- |
| `app_path` | Path to the app module (default: `app:app`) |

The command compares:

- OpenAPI operation IDs vs. MCP tool names
- Security scopes across both surfaces
- Pydantic model serialization for inputs/outputs

## `fastagentic mcp`

MCP protocol operations.

### `mcp serve`

Run the app as an MCP server via stdio.

```bash
fastagentic mcp serve app:app
```

This enables the app to be used with MCP clients like Claude Desktop, VS Code extensions, etc.

In `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "my-agent": {
      "command": "fastagentic",
      "args": ["mcp", "serve", "app:app"]
    }
  }
}
```

### `mcp validate`

Validate MCP schema against specification.

```bash
fastagentic mcp validate
```

### `mcp schema`

Print the MCP schema.

```bash
fastagentic mcp schema
```

### `mcp export`

Export MCP manifest to a file.

```bash
# Export as JSON
fastagentic mcp export -o mcp_manifest.json

# Export as YAML
fastagentic mcp export -o mcp_manifest.yaml --format yaml
```

| Option | Description |
|--------|-------------|
| `--output, -o` | Output file path (default: `mcp_manifest.json`) |
| `--format, -f` | Output format (`json`, `yaml`) |

### `mcp call`

Call an MCP tool directly for testing.

```bash
# Call with JSON input
fastagentic mcp call get_time --input '{"timezone": "UTC"}'

# Call with file input
fastagentic mcp call analyze --file input.json
```

| Option | Description |
|--------|-------------|
| `tool_name` | Name of the tool to call |
| `--input, -i` | Input JSON string |
| `--file, -f` | Input JSON file |

### `mcp stdio`

Run interactive stdio session for MCP (alias for `mcp serve`).

```bash
fastagentic mcp stdio
```

## `fastagentic a2a`

A2A protocol operations.

### `a2a validate`

Validate A2A Agent Card compliance.

```bash
fastagentic a2a validate
```

### `a2a card`

Display the A2A Agent Card.

```bash
fastagentic a2a card
```

### `a2a list`

List registered A2A skills.

```bash
fastagentic a2a list
```

Shows a table of all registered skills with their paths, descriptions, and streaming capability.

### `a2a export`

Export A2A Agent Card to a file.

```bash
# Export as JSON
fastagentic a2a export -o agent_card.json

# Export as YAML
fastagentic a2a export -o agent_card.yaml --format yaml
```

| Option | Description |
|--------|-------------|
| `--output, -o` | Output file path (default: `agent_card.json`) |
| `--format, -f` | Output format (`json`, `yaml`) |

### `a2a ping`

Check connectivity to an external A2A agent.

```bash
fastagentic a2a ping https://agent.example.com
```

Pings an external agent's `.well-known/agent.json` endpoint and displays agent information.

### `a2a invoke`

Invoke a local A2A skill (shows how to call it).

```bash
fastagentic a2a invoke chat --input '{"message": "hello"}'
```

Note: For local skills, this shows the HTTP request format since invocation requires a running server.

## `fastagentic agent`

Interactive agent testing and development CLI. Provides a Claude Code / Gemini CLI-like experience for working with agents.

### `agent chat`

Start an interactive chat session.

```bash
fastagentic agent chat
fastagentic agent chat --url http://localhost:8000 --endpoint /chat
```

| Option | Description |
|--------|-------------|
| `--url, -u` | Agent server URL (default: `http://localhost:8000`) |
| `--endpoint, -e` | Agent endpoint path (default: `/chat`) |
| `--api-key, -k` | API key for authentication |
| `--stream/--no-stream` | Enable/disable streaming (default: enabled) |
| `--verbose, -v` | Show tool calls and metadata |

**REPL Commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/quit`, `/exit`, `/q` | Exit the CLI |
| `/clear` | Clear conversation history |
| `/save [name]` | Save conversation to file |
| `/load [name]` | Load conversation from file |
| `/history` | Show conversation history |
| `/endpoints` | List available endpoints |
| `/use <endpoint>` | Switch endpoint |
| `/config` | Show configuration |
| `/set <key> <value>` | Set configuration |
| `/stream on\|off` | Toggle streaming |
| `/tools on\|off` | Toggle tool display |
| `/format md\|plain\|json` | Set output format |
| `/file <path>` | Send file contents |
| `/export [path]` | Export to markdown |
| `/status` | Check server health |
| `/compact` | Hide metadata |
| `/verbose` | Show all metadata |

### `agent query`

Send a single message to an agent.

```bash
fastagentic agent query "Hello, how are you?"
echo "Summarize this" | fastagentic agent query -
fastagentic agent query "Generate code" -o output.txt
```

| Option | Description |
|--------|-------------|
| `MESSAGE` | Message to send (use `-` for stdin) |
| `--url, -u` | Agent server URL |
| `--endpoint, -e` | Agent endpoint path |
| `--api-key, -k` | API key |
| `--stream/--no-stream` | Enable/disable streaming |
| `--output, -o` | Output file path |
| `--format, -f` | Output format (`plain`, `markdown`, `json`) |

### `agent config`

View or modify CLI configuration.

```bash
fastagentic agent config --show
fastagentic agent config --url http://production:8000
fastagentic agent config --api-key sk-xxx
```

| Option | Description |
|--------|-------------|
| `--show, -s` | Show current configuration |
| `--url, -u` | Set default server URL |
| `--endpoint, -e` | Set default endpoint |
| `--api-key, -k` | Set API key |

Configuration is stored in `~/.fastagentic/config.json`.

### `agent history`

Manage conversation history.

```bash
fastagentic agent history --list
fastagentic agent history --load conv-123
fastagentic agent history --delete conv-123
fastagentic agent history --clear
```

| Option | Description |
|--------|-------------|
| `--list, -l` | List saved conversations |
| `--load NAME` | Load and display a conversation |
| `--delete, -d NAME` | Delete a conversation |
| `--clear` | Clear all conversation history |

Conversations are stored in `~/.fastagentic/history/`.

See the [Agent CLI Guide](cli-agent.md) for detailed documentation.

## Environment Variables

### General

| Variable | Description |
|----------|-------------|
| `FASTAGENTIC_ENV` | Environment (dev, staging, prod) |
| `FASTAGENTIC_CONFIG` | Config file path |
| `FASTAGENTIC_LOG_LEVEL` | Log level (DEBUG, INFO, WARN, ERROR) |
| `FASTAGENTIC_LOG_FORMAT` | Log format (text, json) |

### Server Configuration

| Variable | Description |
|----------|-------------|
| `FASTAGENTIC_SERVER` | Server type: `uvicorn` or `gunicorn` |
| `FASTAGENTIC_HOST` | Bind address |
| `FASTAGENTIC_PORT` | Port number |
| `FASTAGENTIC_WORKERS` | Number of worker processes |
| `FASTAGENTIC_MAX_CONCURRENT` | Max concurrent requests per worker |
| `FASTAGENTIC_INSTANCE_ID` | Instance identifier for metrics |
| `FASTAGENTIC_TIMEOUT_KEEP_ALIVE` | Keep-alive timeout (seconds) |
| `FASTAGENTIC_TIMEOUT_GRACEFUL_SHUTDOWN` | Graceful shutdown timeout (seconds) |

### Connection Pools

| Variable | Description |
|----------|-------------|
| `FASTAGENTIC_REDIS_POOL_SIZE` | Redis pool size per worker |
| `FASTAGENTIC_REDIS_POOL_TIMEOUT` | Redis pool timeout (seconds) |
| `FASTAGENTIC_REDIS_SOCKET_TIMEOUT` | Redis socket timeout (seconds) |
| `FASTAGENTIC_DB_POOL_SIZE` | Database pool size per worker |
| `FASTAGENTIC_DB_MAX_OVERFLOW` | Database pool max overflow |
| `FASTAGENTIC_DB_POOL_TIMEOUT` | Database pool timeout (seconds) |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Configuration error |
| `3` | Validation error |
| `4` | Connection error |

## Shell Completion

```bash
# Bash
fastagentic --install-completion bash

# Zsh
fastagentic --install-completion zsh

# Fish
fastagentic --install-completion fish
```

---

## Planned CLI Commands (Roadmap)

The following commands are planned for future releases:

| Command | Description |
|---------|-------------|
| `fastagentic add endpoint` | Create decorator stubs within an existing project |
| `fastagentic tail` | Stream logs, events, and telemetry in real time |

See the [Roadmap](roadmap.md) for more details.
