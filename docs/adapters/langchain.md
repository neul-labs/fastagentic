# LangChain Adapter

The LangChain adapter wraps [LangChain](https://python.langchain.com/) runnables for deployment through FastAgentic. Deploy chains, agents, and LCEL pipelines with full governance.

## TL;DR

Wrap any LangChain `Runnable` and get REST + MCP + streaming + durability.

## Before FastAgentic

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

app = FastAPI()

prompt = ChatPromptTemplate.from_template("Summarize: {text}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | StrOutputParser()

add_routes(app, chain, path="/summarize")

# LangServe provides REST but:
# - No MCP protocol
# - No built-in auth
# - No durable checkpoints
# - Limited streaming control
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.langchain import LangChainAdapter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Summarize: {text}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | StrOutputParser()

app = App(
    title="Summarizer",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost",
)

@agent_endpoint(
    path="/summarize",
    runnable=LangChainAdapter(chain),
    stream=True,
    durable=True,
    scopes=["summarize:run"],
)
async def summarize(text: str) -> str:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /summarize` | Run chain synchronously |
| `POST /summarize/stream` | Run with token streaming |
| `GET /summarize/{run_id}` | Get run status and result |
| `POST /summarize/{run_id}/resume` | Resume from checkpoint |

### LangChain-Specific Features

**LCEL Support**

Full support for LangChain Expression Language:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | parser
)

@agent_endpoint(path="/qa", runnable=LangChainAdapter(chain))
```

**Agent Executor**

Wrap LangChain agents:

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

@agent_endpoint(
    path="/agent",
    runnable=LangChainAdapter(executor),
    stream=True,
)
async def run_agent(query: str) -> str:
    pass
```

**Callback Integration**

LangChain callbacks work with FastAgentic events:

```python
LangChainAdapter(
    chain,
    include_callbacks=True,  # Forward callback events
)
```

## Configuration Options

### LangChainAdapter Constructor

```python
LangChainAdapter(
    runnable: Runnable,
    include_callbacks: bool = True,
    checkpoint_on_tool_call: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runnable` | `Runnable` | required | LangChain runnable |
| `include_callbacks` | `bool` | `True` | Forward callback events |
| `checkpoint_on_tool_call` | `bool` | `True` | Checkpoint before tools |

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `token` | LLM output token | `{content}` |
| `chain_start` | Chain begins | `{name, inputs}` |
| `chain_end` | Chain completes | `{name, outputs}` |
| `tool_call` | Tool invoked | `{tool, args}` |
| `tool_result` | Tool returns | `{tool, output}` |
| `retriever_start` | Retrieval begins | `{query}` |
| `retriever_end` | Retrieval complete | `{documents}` |

## Migration from LangServe

### Step 1: Remove LangServe

```python
# Before
from langserve import add_routes
add_routes(app, chain, path="/summarize")

# After - remove langserve import and add_routes call
```

### Step 2: Add FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.langchain import LangChainAdapter

app = App(...)

@agent_endpoint(path="/summarize", runnable=LangChainAdapter(chain))
async def summarize(text: str) -> str:
    pass
```

### What You Gain

| Feature | LangServe | FastAgentic |
|---------|-----------|-------------|
| REST endpoints | Yes | Yes |
| MCP protocol | No | Yes |
| OAuth2 auth | No | Yes |
| Durable checkpoints | No | Yes |
| Cost tracking | No | Yes |
| Multi-framework | No | Yes |

## Common Patterns

### RAG Pipeline

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

retriever = Chroma(embedding_function=OpenAIEmbeddings()).as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

@agent_endpoint(path="/qa", runnable=LangChainAdapter(chain), stream=True)
async def qa(question: str) -> str:
    pass
```

### Tool-Using Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun

tools = [DuckDuckGoSearchRun()]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

@agent_endpoint(
    path="/search",
    runnable=LangChainAdapter(executor),
    stream=True,
    durable=True,
)
async def search(query: str) -> str:
    pass
```

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [LangGraph Adapter](langgraph.md) - For stateful workflows
- [PydanticAI Adapter](pydanticai.md) - For type-safe agents
