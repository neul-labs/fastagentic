# Agent CLI

The Agent CLI provides a Claude Code / Gemini CLI-like experience for testing and developing agents. It offers an interactive REPL for conversations, single-query execution for scripting, and comprehensive configuration management.

## Quick Start

```bash
# Start interactive chat
fastagentic agent chat

# Send a single query
fastagentic agent query "What is 2 + 2?"

# Configure the CLI
fastagentic agent config --url http://localhost:8000
```

## Commands

### Interactive Chat

Start an interactive chat session with an agent:

```bash
fastagentic agent chat [OPTIONS]
```

**Options:**
- `--url, -u` - Agent server URL (default: `http://localhost:8000`)
- `--endpoint, -e` - Agent endpoint path (default: `/chat`)
- `--api-key, -k` - API key for authentication
- `--stream/--no-stream` - Enable/disable streaming (default: enabled)
- `--verbose, -v` - Show tool calls and metadata

**Example:**

```bash
fastagentic agent chat --url http://localhost:8000 --endpoint /chat --verbose
```

### REPL Commands

Once in the interactive chat, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/quit`, `/exit`, `/q` | Exit the CLI |
| `/clear` | Clear conversation history |
| `/save [name]` | Save conversation to file |
| `/load [name]` | Load conversation from file |
| `/history` | Show conversation history |
| `/endpoints` | List available agent endpoints |
| `/use <endpoint>` | Switch to a different endpoint |
| `/config` | Show current configuration |
| `/set <key> <value>` | Set configuration option |
| `/stream on\|off` | Toggle streaming mode |
| `/tools on\|off` | Toggle tool call display |
| `/format md\|plain\|json` | Set output format |
| `/file <path>` | Send file contents as message |
| `/export [path]` | Export conversation to markdown |
| `/status` | Check server health |
| `/compact` | Hide metadata (compact view) |
| `/verbose` | Show all metadata (verbose view) |

### Single Query

Send a single message to an agent:

```bash
fastagentic agent query MESSAGE [OPTIONS]
```

**Arguments:**
- `MESSAGE` - Message to send (use `-` for stdin)

**Options:**
- `--url, -u` - Agent server URL
- `--endpoint, -e` - Agent endpoint path
- `--api-key, -k` - API key for authentication
- `--stream/--no-stream` - Enable/disable streaming
- `--output, -o` - Output file path
- `--format, -f` - Output format (`plain`, `markdown`, `json`)

**Examples:**

```bash
# Simple query
fastagentic agent query "Summarize this document"

# Read from stdin
cat document.txt | fastagentic agent query -

# Save output to file
fastagentic agent query "Generate code" -o output.py

# JSON output
fastagentic agent query "List items" --format json
```

### Configuration

View or modify CLI configuration:

```bash
fastagentic agent config [OPTIONS]
```

**Options:**
- `--show, -s` - Show current configuration
- `--url, -u` - Set default server URL
- `--endpoint, -e` - Set default endpoint
- `--api-key, -k` - Set API key

**Examples:**

```bash
# Show current config
fastagentic agent config --show

# Set server URL
fastagentic agent config --url http://production:8000

# Set API key
fastagentic agent config --api-key sk-xxx
```

Configuration is stored in `~/.fastagentic/config.json`.

### History Management

Manage conversation history:

```bash
fastagentic agent history [OPTIONS]
```

**Options:**
- `--list, -l` - List saved conversations
- `--load NAME` - Load and display a conversation
- `--delete, -d NAME` - Delete a conversation
- `--clear` - Clear all conversation history

**Examples:**

```bash
# List saved conversations
fastagentic agent history --list

# Load a conversation
fastagentic agent history --load conv-1234567

# Delete a conversation
fastagentic agent history --delete conv-1234567

# Clear all history
fastagentic agent history --clear
```

## Configuration

### Environment Variables

The CLI respects these environment variables:

- `FASTAGENTIC_URL` - Default server URL
- `FASTAGENTIC_API_KEY` - Default API key
- `FASTAGENTIC_ENDPOINT` - Default endpoint

### Config File

Configuration is stored in `~/.fastagentic/config.json`:

```json
{
  "base_url": "http://localhost:8000",
  "endpoint": "/chat",
  "api_key": null,
  "timeout": 300.0,
  "stream": true,
  "show_tools": true,
  "show_thinking": true,
  "show_usage": true,
  "output_format": "markdown",
  "max_history": 100
}
```

## Features

### Streaming Responses

By default, responses are streamed with real-time updates. Use `--no-stream` to wait for complete responses.

### Tool Call Visualization

When `--verbose` is enabled, tool calls are displayed with:
- Tool name
- Input arguments (JSON formatted)
- Output results

### Conversation History

Conversations are automatically tracked and can be:
- Saved to `~/.fastagentic/history/`
- Loaded for continuation
- Exported to markdown

### Output Formats

Three output formats are supported:
- `markdown` - Rendered markdown (default in REPL)
- `plain` - Plain text (default for queries)
- `json` - Raw JSON response

## Integration with Agents

The CLI works with any FastAgentic agent endpoint. The endpoint should accept:

**Request:**
```json
{
  "message": "User message",
  "stream": true,
  "history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Streaming Response (SSE):**
```
data: {"type": "token", "data": {"content": "Hello"}}
data: {"type": "tool_call", "data": {"name": "search", "input": {}}}
data: {"type": "tool_result", "data": {"name": "search", "output": []}}
data: {"type": "usage", "data": {"total_tokens": 100}}
data: {"type": "done", "data": {"result": "Complete response"}}
data: [DONE]
```

**Non-Streaming Response:**
```json
{
  "response": "Complete response text",
  "usage": {
    "total_tokens": 100,
    "cost": 0.001,
    "latency_ms": 500
  }
}
```

## Examples

### Basic Chat Session

```bash
$ fastagentic agent chat

FastAgentic Agent CLI
Interactive agent testing and development

Server: http://localhost:8000
Endpoint: /chat

Commands:
  /help      - Show all commands
  /quit      - Exit the CLI
  /clear     - Clear conversation
  /save      - Save conversation

> What is the capital of France?

Paris is the capital of France.

> /save france-chat
Saved to ~/.fastagentic/history/france-chat.json

> /quit
Goodbye!
```

### Scripting with Pipes

```bash
# Process a file
cat report.txt | fastagentic agent query - --endpoint /summarize -o summary.txt

# Chain commands
fastagentic agent query "Generate test data" | fastagentic agent query - --endpoint /analyze
```

### Batch Processing

```bash
# Process multiple inputs
for file in docs/*.md; do
  fastagentic agent query "$(cat $file)" -o "summaries/$(basename $file)"
done
```
