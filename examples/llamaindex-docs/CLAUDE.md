# LlamaIndex Document Q&A - Claude Code Guide

This is a FastAgentic example using LlamaIndex for document question-answering.

## Project Structure

```
llamaindex-docs/
├── CLAUDE.md           # This file - instructions for Claude Code
├── app.py              # Main FastAgentic application with LlamaIndex
├── data/               # Document directory
│   ├── fastagentic.txt
│   ├── llamaindex.txt
│   └── rag-overview.txt
├── pyproject.toml      # Dependencies
├── .env.example        # Environment variables template
└── README.md           # User documentation
```

## Key Commands

```bash
# Install dependencies
uv sync

# Run the server
uv run fastagentic run

# Test with CLI
uv run fastagentic agent chat --endpoint /query

# Run tests
uv run pytest tests/ -v
```

## Architecture

- **FastAgentic App** (`app.py`): Wraps LlamaIndex query engine with REST/MCP interfaces
- **LlamaIndex Index**: VectorStoreIndex for semantic document search
- **Query Engine**: Streaming-enabled query engine with top-k retrieval
- **Documents**: Sample text files in `data/` directory

## Key Components

### LlamaIndex Setup

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)
```

### LlamaIndexAdapter

```python
@agent_endpoint(
    path="/query",
    runnable=LlamaIndexAdapter(query_engine=query_engine),
    stream=True,
    durable=True,
)
```

## When Modifying

1. **Add documents**: Add files to `data/` folder and restart
2. **Change embedding model**: Modify `Settings.embed_model`
3. **Change LLM**: Modify `Settings.llm`
4. **Adjust retrieval**: Change `similarity_top_k` in query engine

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models/embeddings
- `FASTAGENTIC_ENV`: Environment (dev/staging/prod)
- `REDIS_URL`: Optional durable store

## Testing

```bash
# Run tests
uv run pytest

# Test endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Common Tasks

### Use chat engine for memory

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(chat_mode="context", memory=memory)
adapter = LlamaIndexAdapter(chat_engine=chat_engine)
```

### Add metadata filtering

```python
query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[MetadataFilter(key="category", value="technical")]
    )
)
```

### Use different vector store

```python
from llama_index.vector_stores.chroma import ChromaVectorStore

vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)
```
