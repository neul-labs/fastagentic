# LangChain RAG Agent - Claude Code Guide

This is a FastAgentic example using LangChain for retrieval-augmented generation.

## Project Structure

```
langchain-rag/
├── CLAUDE.md           # This file - instructions for Claude Code
├── app.py              # Main FastAgentic application with LangChain chain
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
uv run fastagentic agent chat --endpoint /ask

# Run tests
uv run pytest tests/ -v
```

## Architecture

- **FastAgentic App** (`app.py`): Wraps the LangChain chain with REST/MCP interfaces
- **LCEL Chain**: `prompt | llm | parser` pipeline with retrieval
- **Knowledge Base**: Simple in-memory dictionary (replace with vector store for production)

## Key Components

### LangChain LCEL Chain

```python
chain = (
    {"context": lambda x: retrieve_context(x["question"]), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### LangChainAdapter

```python
@agent_endpoint(
    path="/ask",
    runnable=LangChainAdapter(chain),
    stream=True,
    durable=True,
)
```

## When Modifying

1. **Change retrieval**: Replace `retrieve_context()` with vector store retriever
2. **Change prompt**: Edit the `ChatPromptTemplate.from_template()` call
3. **Change model**: Modify `ChatOpenAI(model="gpt-4o-mini")`
4. **Add tools**: Use LangChain tools with agent executor

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI models
- `FASTAGENTIC_ENV`: Environment (dev/staging/prod)
- `REDIS_URL`: Optional durable store

## Testing

```bash
# Run tests
uv run pytest

# Test endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangChain?"}'
```

## Common Tasks

### Add vector store retrieval

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Enable checkpointing

```python
@agent_endpoint(
    path="/ask",
    runnable=LangChainAdapter(chain, checkpoint_events=["on_chain_end"]),
    durable=True,
)
```
