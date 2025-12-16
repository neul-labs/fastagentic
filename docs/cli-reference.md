# CLI Reference

The FastAgentic CLI streamlines project scaffolding, local development, introspection, and contract testing. This reference documents each command and its key options.

## Global Options

| Option          | Description                              |
| --------------- | ---------------------------------------- |
| `--config PATH` | Use an alternate configuration file      |
| `--env NAME`    | Load environment-specific settings       |
| `--quiet`       | Reduce logging verbosity                 |
| `--json`        | Emit machine-readable output when available |

## `fastagentic new`

Scaffold a new application.

```bash
fastagentic new my-service --adapter langgraph --auth oidc
```

| Option             | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `--adapter`        | Preconfigure sample workflow (`langchain`, `langgraph`, `crewai`) |
| `--auth`           | Include auth boilerplate (`oidc`, `none`)                |
| `--telemetry`      | Enable telemetry configuration (`otel`, `none`)          |
| `--force`          | Overwrite existing directory                             |

Generated structure:

- `app.py` with an `App` instance
- `models/` for Pydantic schemas
- `endpoints/` with sample decorators
- `config/settings.yaml` and `.env`
- `tests/test_contracts.py` for schema parity

## `fastagentic add endpoint`

Create decorator stubs within an existing project.

```bash
fastagentic add endpoint support triage --type agent --stream --durable
```

| Option        | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| `--type`      | `tool`, `resource`, `prompt`, or `agent`                          |
| `--path`      | REST path to register                                             |
| `--stream`    | Enable streaming boilerplate                                      |
| `--durable`   | Include checkpointing helpers                                     |
| `--scopes`    | Comma-delimited list of required scopes                           |

The generator creates files under `endpoints/` or `prompts/` with Pydantic model placeholders.

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
| `--server, -s` | uvicorn | Server type: `uvicorn` or `gunicorn` |
| `--host` | 127.0.0.1 | Bind address for the server |
| `--port` | 8000 | Port for HTTP traffic |
| `--workers` | 1 | Number of worker processes |
| `--reload` | false | Enable auto-reload (dev only, forces 1 worker) |

### Scalability Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-concurrent` | unlimited | Maximum concurrent requests per worker |
| `--instance-id` | auto | Instance ID for cluster metrics |
| `--timeout-graceful` | 30 | Graceful shutdown timeout (seconds) |

### Connection Pool Options

| Option | Default | Description |
|--------|---------|-------------|
| `--redis-pool-size` | 10 | Redis connection pool size per worker |
| `--db-pool-size` | 5 | Database connection pool size per worker |
| `--db-max-overflow` | 10 | Database pool overflow connections |

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

## `fastagentic tail`

Streams logs, events, and telemetry in real time.

```bash
fastagentic tail --runs 10 --events token,node_start
```

| Option          | Description                                              |
| --------------- | -------------------------------------------------------- |
| `--runs`        | Limit to the most recent N runs                          |
| `--events`      | Comma-delimited list of event types to display           |
| `--follow`      | Continue streaming until interrupted                     |
| `--json`        | Emit structured JSON rather than formatted text          |

`tail` can connect to local or remote deployments by reading configuration from environment variables or the specified config file.

## `fastagentic test contract`

Validates that REST and MCP schemas remain in sync.

```bash
fastagentic test contract --fail-on-drift
```

| Option             | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `--fail-on-drift`  | Exit with non-zero status when discrepancies detected |
| `--output PATH`    | Write diff report to file                            |
| `--json`           | Emit diff as JSON                                    |

The command compares:

- OpenAPI operation IDs vs. MCP tool names
- Security scopes across both surfaces
- Pydantic model serialization for inputs/outputs
- Prompt metadata and example alignments

## `fastagentic inspect`

Interactively explore registered decorators, schemas, and runtime configuration.

```bash
fastagentic inspect --schema tool:summaries
```

| Option          | Description                                             |
| --------------- | ------------------------------------------------------- |
| `--schema`      | Print the schema for a specific tool/resource/prompt    |
| `--list`        | List registered assets (`tools`, `resources`, `prompts`, `agents`) |
| `--runs`        | Show active runs and their statuses                     |
| `--config`      | Display resolved configuration values                   |

## `fastagentic templates`

Manage project templates.

### `templates list`

```bash
fastagentic templates list [--category official|community] [--verbose]
```

| Option | Description |
|--------|-------------|
| `--category` | Filter by `official`, `community`, or `all` |
| `--verbose` | Show detailed template info |

### `templates search`

```bash
fastagentic templates search "multi-agent"
```

### `templates info`

```bash
fastagentic templates info pydanticai
```

### `templates refresh`

Refresh template cache from remote repository.

```bash
fastagentic templates refresh
```

## `fastagentic mcp`

MCP protocol operations.

### `mcp validate`

Validate MCP schema against specification.

```bash
fastagentic mcp validate [--spec-version 2025-11-25]
```

### `mcp manifest`

Generate MCP manifest file.

```bash
fastagentic mcp manifest > mcp.json
fastagentic mcp manifest -o mcp.yaml --format yaml
```

### `mcp call`

Test tool invocation.

```bash
fastagentic mcp call summarize_text --input '{"text": "..."}'
fastagentic mcp call analyze_data --file input.json
```

### `mcp stdio`

Start interactive stdio session.

```bash
fastagentic mcp stdio --interactive
```

## `fastagentic a2a`

A2A protocol operations.

### `a2a validate`

Validate A2A Agent Card.

```bash
fastagentic a2a validate [--spec-version 0.3]
```

### `a2a card`

Display or export Agent Card.

```bash
fastagentic a2a card
fastagentic a2a card -o agent.json --extended
```

### `a2a list`

List registered agents.

```bash
fastagentic a2a list [--internal] [--external]
```

### `a2a invoke`

Invoke an agent skill.

```bash
fastagentic a2a invoke support-triage --input '{"title": "..."}'
fastagentic a2a invoke deep-research --stream --input '{"query": "..."}'
```

### `a2a ping`

Check connectivity to external agent.

```bash
fastagentic a2a ping https://external-agent.example.com
```

## `fastagentic agent`

Interactive agent testing and development CLI. Provides a Claude Code / Gemini CLI-like experience for working with agents.

### `agent chat`

Start an interactive chat session.

```bash
fastagentic agent chat
fastagentic agent chat --url http://localhost:8000 --endpoint /chat
fastagentic agent chat --verbose
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

## `fastagentic config`

Configuration management.

### `config show`

Display resolved configuration.

```bash
fastagentic config show [--env production] [--secrets]
```

### `config validate`

Validate configuration file.

```bash
fastagentic config validate config/settings.yaml
```

### `config init`

Generate default configuration.

```bash
fastagentic config init -o config/settings.yaml
fastagentic config init -o .env --format env
```

## Environments and Configuration

CLI commands read configuration in this order:

1. Command-line flags
2. Environment variables (`FASTAGENTIC_*`)
3. Project `config/settings.yaml`
4. Local `.env` file

This hierarchy keeps local development flexible while preserving predictable deployment behavior.

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

