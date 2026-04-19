# LlamaIndex Adapter

The LlamaIndex adapter wraps [LlamaIndex](https://www.llamaindex.ai/) query engines, chat engines, and agents for deployment through FastAgentic. Deploy RAG applications with full checkpointing and source tracking.

## TL;DR

Wrap any LlamaIndex `QueryEngine`, `ChatEngine`, or `Agent` and get REST + MCP + streaming + durability + source events.

## Why LlamaIndex + FastAgentic?

LlamaIndex excels at RAG and document processing. FastAgentic adds production deployment:

| Capability | LlamaIndex | FastAgentic |
|------------|------------|-------------|
| Document indexing | Built-in | Inherited |
| Query engines | Built-in | Inherited |
| Chat engines | Built-in | Inherited |
| Agents | Built-in | Inherited |
| REST API | Manual | Automatic |
| MCP Protocol | No | Yes |
| Checkpoints | No | Built-in |
| Source tracking | Callbacks | Automatic events |
| Auth & Policy | Application code | Middleware |

**Index with LlamaIndex. Deploy with FastAgentic.**

## Before FastAgentic

```python
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

app = FastAPI()

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

@app.post("/query")
async def query(question: str):
    response = query_engine.query(question)
    return {"answer": str(response), "sources": [n.text for n in response.source_nodes]}

# No streaming, no checkpoints, no auth, no durability...
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.llamaindex import LlamaIndexAdapter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)

app = App(
    title="RAG Service",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost",
)

@agent_endpoint(
    path="/query",
    runnable=LlamaIndexAdapter(query_engine=query_engine),
    stream=True,
    durable=True,
)
async def query(question: str) -> str:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /query` | Run query synchronously |
| `POST /query/stream` | Run with token streaming |
| `GET /query/{run_id}` | Get run status and result |
| `POST /query/{run_id}/resume` | Resume from checkpoint |

### LlamaIndex-Specific Features

**Query Engine Support**

```python
query_engine = index.as_query_engine(streaming=True)
adapter = LlamaIndexAdapter(query_engine=query_engine)
```

**Chat Engine Support**

```python
chat_engine = index.as_chat_engine()
adapter = LlamaIndexAdapter(chat_engine=chat_engine)

# Automatically tracks chat_history in checkpoints
```

**Agent Support**

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(tools, llm=llm)
adapter = LlamaIndexAdapter(agent=agent)

# Tool events extracted from agent execution
```

**Source Node Tracking**

Automatically emits source events:

```python
# Events emitted:
# node_start → token → source → source → checkpoint → done
```

## Configuration Options

### LlamaIndexAdapter Constructor

```python
LlamaIndexAdapter(
    query_engine: QueryEngine | None = None,
    chat_engine: ChatEngine | None = None,
    agent: AgentRunner | None = None,
    checkpoint_queries: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_engine` | `QueryEngine` | `None` | LlamaIndex query engine |
| `chat_engine` | `ChatEngine` | `None` | LlamaIndex chat engine |
| `agent` | `AgentRunner` | `None` | LlamaIndex agent |
| `checkpoint_queries` | `bool` | `True` | Create checkpoints after queries |

**Note:** Provide exactly one of `query_engine`, `chat_engine`, or `agent`.

### Builder Methods

```python
# Configure checkpointing
adapter = LlamaIndexAdapter(query_engine=qe).with_checkpoints(enabled=True)

# Switch engine type
adapter = LlamaIndexAdapter(query_engine=qe).with_query_engine(new_qe)
adapter = LlamaIndexAdapter(query_engine=qe).with_chat_engine(ce)
adapter = LlamaIndexAdapter(query_engine=qe).with_agent(agent)
```

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `node_start` | Query/retrieval begins | `{name}` |
| `node_end` | Query completes | `{name}` |
| `token` | Output token | `{content}` |
| `source` | Source node retrieved | `{text, metadata, score}` |
| `tool_call` | Tool invoked (agents) | `{name, input}` |
| `tool_result` | Tool returns (agents) | `{name, output}` |
| `checkpoint` | State saved | `{step}` |
| `done` | Query complete | `{result}` |

## Checkpoint State

The adapter automatically creates checkpoints containing:

```python
{
    "state": {
        "query": str,       # Original query
        "response": str,    # Response text
        "completed": bool,
    },
    "messages": [           # Chat history (chat engine only)
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
    "context": {
        "sources": [        # Retrieved source nodes
            {"text": "...", "metadata": {...}, "score": 0.95},
        ],
    },
}
```

**Checkpoint Triggers:**
- After retrieval completes
- After generation completes
- On query completion

**Resume Behavior:**
When resuming, the adapter:
1. Restores `chat_history` from checkpoint (for chat engines)
2. Sets `_is_resumed = True` on the run context
3. Emits a `NODE_START` event with `name: "__resume__"`

## Common Patterns

### RAG Query Engine

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=5,
)

@agent_endpoint(
    path="/search",
    runnable=LlamaIndexAdapter(query_engine=query_engine),
    stream=True,
)
async def search(query: str) -> str:
    pass
```

### Chat with Memory

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
)

@agent_endpoint(
    path="/chat",
    runnable=LlamaIndexAdapter(chat_engine=chat_engine),
    stream=True,
    durable=True,  # Persists chat history
)
async def chat(message: str) -> str:
    pass
```

### ReAct Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="search",
    description="Search the knowledge base",
)

agent = ReActAgent.from_tools([query_tool], llm=llm, verbose=True)

@agent_endpoint(
    path="/agent",
    runnable=LlamaIndexAdapter(agent=agent),
    stream=True,
    durable=True,
)
async def agent_query(question: str) -> str:
    pass
```

### Sub-Question Query

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2],
)

@agent_endpoint(
    path="/complex-query",
    runnable=LlamaIndexAdapter(query_engine=sub_question_engine),
    stream=True,
)
async def complex_query(question: str) -> str:
    pass
```

## Source Event Details

Source nodes are automatically emitted as events:

```python
{
    "type": "source",
    "data": {
        "text": "Document excerpt...",
        "metadata": {
            "file_name": "doc.pdf",
            "page": 5,
        },
        "score": 0.92,
    }
}
```

This enables:
- Real-time source attribution
- Citation generation
- Source quality monitoring

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [LangChain Adapter](langchain.md) - For chain deployment
- [DSPy Adapter](dspy.md) - For prompt optimization
