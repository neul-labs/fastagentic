# LlamaIndex Document Q&A

A document question-answering agent built with LlamaIndex and deployed with FastAgentic.

## Features

- LlamaIndex VectorStoreIndex for semantic search
- Source node tracking in responses
- Streaming responses via SSE
- Checkpointing for durability
- REST, MCP, and A2A interfaces

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the server
uv run fastagentic run

# 4. Test with CLI
uv run fastagentic agent chat --endpoint /query
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query the document collection |
| `/query/stream` | POST | Query with streaming response |
| `/query/{run_id}` | GET | Get run status |
| `/query/{run_id}/resume` | POST | Resume from checkpoint |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/mcp/schema` | GET | MCP schema |

## Testing

```bash
# Simple query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAgentic?"}'

# Query about RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does RAG work?"}'

# Streaming response
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "What is LlamaIndex?"}'

# Using the Agent CLI
uv run fastagentic agent chat --endpoint /query
```

## Sample Documents

The `data/` folder contains sample documents about:
- FastAgentic framework
- LlamaIndex data framework
- RAG (Retrieval-Augmented Generation) overview

Add your own documents to `data/` and restart the server.

## Response Format

Responses include the answer and source nodes:

```json
{
  "response": "FastAgentic is a production deployment framework...",
  "source_nodes": [
    {
      "text": "FastAgentic - The Deployment Layer for AI Agents...",
      "score": 0.92,
      "metadata": {"file_name": "fastagentic.txt"}
    }
  ]
}
```

## Project Structure

```
llamaindex-docs/
├── app.py              # FastAgentic application with LlamaIndex
├── data/               # Document directory
│   ├── fastagentic.txt
│   ├── llamaindex.txt
│   └── rag-overview.txt
├── pyproject.toml      # Dependencies
├── .env.example        # Environment template
├── CLAUDE.md           # Claude Code instructions
└── README.md           # This file
```

## Adding Your Own Documents

1. Add documents to the `data/` folder (supports .txt, .pdf, .md, etc.)
2. Restart the server to reindex
3. Query your documents via the API

## Using Chat Engine

For conversational memory, switch to chat engine:

```python
chat_engine = index.as_chat_engine(chat_mode="context")
adapter = LlamaIndexAdapter(chat_engine=chat_engine)
```

## Learn More

- [FastAgentic Documentation](https://github.com/neul-labs/fastagentic)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex Adapter Guide](../../docs/adapters/llamaindex.md)
