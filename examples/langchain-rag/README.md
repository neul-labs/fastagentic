# LangChain RAG Agent

A retrieval-augmented generation (RAG) agent built with LangChain LCEL chains and deployed with FastAgentic.

## Features

- LangChain LCEL chain with prompt | llm | parser pipeline
- Simple in-memory knowledge base retrieval
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
uv run fastagentic agent chat --endpoint /ask
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question (RAG) |
| `/ask/stream` | POST | Ask with streaming response |
| `/ask/{run_id}` | GET | Get run status |
| `/ask/{run_id}/resume` | POST | Resume from checkpoint |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/mcp/schema` | GET | MCP schema |

## Testing

```bash
# Simple question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAgentic?"}'

# Streaming response
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "What is RAG?"}'

# Using the Agent CLI
uv run fastagentic agent chat --endpoint /ask
```

## Knowledge Base

The example includes a simple in-memory knowledge base with topics:
- FastAgentic
- LangChain
- RAG
- LCEL

You can extend the `KNOWLEDGE_BASE` dictionary in `app.py` or replace it with a real vector store.

## Extending with Vector Store

Replace the simple retrieval with a real vector store:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_texts(texts, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## Project Structure

```
langchain-rag/
├── app.py              # FastAgentic application with LangChain chain
├── pyproject.toml      # Dependencies
├── .env.example        # Environment template
├── CLAUDE.md           # Claude Code instructions
└── README.md           # This file
```

## Learn More

- [FastAgentic Documentation](https://github.com/neul-labs/fastagentic)
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Adapter Guide](../../docs/adapters/langchain.md)
