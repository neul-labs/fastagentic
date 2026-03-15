# LangChain Template

Production-ready starter for chains and agents using [LangChain](https://www.langchain.com/).

## Create Project

```bash
fastagentic new my-agent --template langchain
cd my-agent
```

## Project Structure

```
my-agent/
├── app.py                    # FastAgentic entry point
├── chains/
│   ├── __init__.py
│   └── qa.py                 # RAG chain
├── agents/
│   ├── __init__.py
│   └── assistant.py          # Tool-using agent
├── tools/
│   ├── __init__.py
│   └── custom.py             # Custom tools
├── models/
│   ├── inputs.py
│   └── outputs.py
├── config/
│   └── settings.yaml
├── tests/
│   ├── test_chains.py
│   └── test_contracts.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── k8s/
│   └── *.yaml
└── pyproject.toml
```

## Core Files

### app.py

```python
"""FastAgentic application entry point."""
import os
from fastagentic import App
from fastagentic.protocols import enable_mcp, enable_a2a
from fastagentic.adapters.langchain import LangChainAdapter

from chains.qa import qa_chain
from agents.assistant import assistant_agent
from models.inputs import QuestionInput, AssistantInput
from models.outputs import AnswerOutput, AssistantOutput

app = App(
    title="Knowledge Assistant",
    version="1.0.0",
    description="RAG-powered Q&A and tool-using assistant",
    durable_store=os.getenv("DURABLE_STORE", "redis://localhost:6379"),
)

enable_mcp(app, tasks_enabled=True)
enable_a2a(app)


# RAG chain endpoint
@app.agent_endpoint(
    path="/qa",
    runnable=LangChainAdapter(qa_chain),
    input_model=QuestionInput,
    output_model=AnswerOutput,
    stream=True,
    mcp_tool="ask_question",
    a2a_skill="knowledge-qa",
)
async def question_answer(question: QuestionInput) -> AnswerOutput:
    """Answer questions using RAG."""
    ...


# Agent endpoint
@app.agent_endpoint(
    path="/assist",
    runnable=LangChainAdapter(
        assistant_agent,
        stream_tokens=True,
        include_tool_calls=True,
    ),
    input_model=AssistantInput,
    output_model=AssistantOutput,
    stream=True,
    durable=True,
    mcp_tool="assistant",
    a2a_skill="general-assistant",
)
async def assistant(request: AssistantInput) -> AssistantOutput:
    """General-purpose assistant with tools."""
    ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### chains/qa.py

```python
"""RAG question-answering chain using LCEL."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Initialize components
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings()

# Vector store (configure path via env)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on
    the provided context. If the context doesn't contain relevant information,
    say so clearly. Always cite your sources.

    Context:
    {context}"""),
    ("human", "{question}"),
])


def format_docs(docs):
    """Format retrieved documents."""
    return "\n\n".join([
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    ])


# Build LCEL chain
qa_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

### agents/assistant.py

```python
"""Tool-using assistant agent."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor

from tools.custom import calculator, web_search, code_executor


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True)

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to various tools.
    Use tools when needed to provide accurate and helpful responses.
    Always explain your reasoning."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Available tools
tools = [calculator, web_search, code_executor]

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create executor
assistant_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    return_intermediate_steps=True,
)
```

### tools/custom.py

```python
"""Custom tools for the assistant."""
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3")
    """
    try:
        # Safe evaluation
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def code_executor(code: str, language: str = "python") -> str:
    """Execute code in a sandboxed environment.

    Args:
        code: The code to execute
        language: Programming language (python, javascript)
    """
    # Use sandboxed execution (e.g., Docker, E2B)
    from sandbox import execute
    return execute(code, language)


# Tavily search
web_search = TavilySearchResults(max_results=5)
```

### models/inputs.py

```python
"""Input models."""
from pydantic import BaseModel, Field


class QuestionInput(BaseModel):
    """Question for RAG Q&A."""

    question: str = Field(description="The question to answer")
    filters: dict | None = Field(
        default=None,
        description="Optional metadata filters for retrieval",
    )


class AssistantInput(BaseModel):
    """Input for general assistant."""

    input: str = Field(description="User message or request")
    chat_history: list[dict] | None = Field(
        default=None,
        description="Previous conversation messages",
    )
```

### models/outputs.py

```python
"""Output models."""
from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    """RAG answer output."""

    answer: str = Field(description="The answer to the question")
    sources: list[str] = Field(
        default_factory=list,
        description="Sources used for the answer",
    )


class AssistantOutput(BaseModel):
    """Assistant response output."""

    output: str = Field(description="Assistant's response")
    intermediate_steps: list[dict] | None = Field(
        default=None,
        description="Tool calls and results",
    )
```

### tests/test_chains.py

```python
"""Chain tests."""
import pytest
from unittest.mock import patch, MagicMock

from chains.qa import qa_chain
from agents.assistant import assistant_agent


@pytest.mark.asyncio
async def test_qa_chain_structure():
    """Test QA chain has correct structure."""
    # Chain should have retriever, prompt, llm, parser
    assert qa_chain.first is not None
    assert qa_chain.last is not None


@pytest.mark.asyncio
@patch("chains.qa.retriever")
async def test_qa_chain_execution(mock_retriever):
    """Test QA chain execution."""
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="Test content", metadata={"source": "test"})
    ]

    result = await qa_chain.ainvoke({"question": "What is X?"})
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_assistant_has_tools():
    """Test assistant has correct tools."""
    assert len(assistant_agent.tools) == 3
    tool_names = [t.name for t in assistant_agent.tools]
    assert "calculator" in tool_names
    assert "web_search" in tool_names
```

## Configuration

### .env.example

```bash
# LLM Provider
OPENAI_API_KEY=sk-...

# Search API
TAVILY_API_KEY=tvly-...

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db

# Durable Store
DURABLE_STORE=redis://localhost:6379

# Application
LOG_LEVEL=INFO
```

### pyproject.toml

```toml
[project]
name = "my-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastagentic[langchain]>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "chromadb>=0.5.0",
    "tavily-python>=0.5.0",
]
```

## Running

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env

# Index documents (for RAG)
python scripts/index_docs.py

# Run
fastagentic run --reload
```

## API Usage

### RAG Q&A

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

### Stream Q&A

```bash
curl -N http://localhost:8000/qa/stream \
  -H "Accept: text/event-stream" \
  -d '{"question": "Explain the architecture"}'

# Streams tokens:
# event: token
# data: {"content": "The", "delta": "The"}
# event: token
# data: {"content": "The architecture", "delta": " architecture"}
```

### Assistant with Tools

```bash
curl -X POST http://localhost:8000/assist \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Calculate 15% tip on $85.50, then search for tipping etiquette"
  }'

# Response includes tool calls:
{
  "output": "The 15% tip would be $12.83...",
  "intermediate_steps": [
    {"tool": "calculator", "input": "85.50 * 0.15", "output": "12.825"},
    {"tool": "web_search", "input": "tipping etiquette", "output": "..."}
  ]
}
```

## Migrating from LangServe

If you have existing LangServe deployments, FastAgentic provides an easy migration path:

### Before (LangServe)

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()
add_routes(app, qa_chain, path="/qa")
```

### After (FastAgentic)

```python
from fastagentic import App
from fastagentic.adapters.langchain import LangChainAdapter

app = App(title="My App")

@app.agent_endpoint(
    path="/qa",
    runnable=LangChainAdapter(qa_chain),
    stream=True,
    mcp_tool="qa",           # Now also exposed via MCP
    a2a_skill="qa-agent",    # And A2A
)
async def qa(input: Input) -> Output:
    ...
```

### Migration Benefits

| Feature | LangServe | FastAgentic |
|---------|-----------|-------------|
| REST API | Yes | Yes |
| Streaming | SSE only | SSE + WebSocket + MCP |
| MCP Protocol | No | Yes |
| A2A Protocol | No | Yes |
| Authentication | Manual | Built-in OAuth2/OIDC |
| Checkpointing | No | Yes |
| Cost Tracking | No | Yes |
| Observability | Manual | Built-in OTEL |

## LCEL Composition

FastAgentic preserves full LCEL functionality:

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Complex chain with parallel execution
complex_chain = (
    RunnableParallel(
        summary=summary_chain,
        keywords=keyword_chain,
        sentiment=sentiment_chain,
    )
    | RunnableLambda(combine_results)
    | output_chain
)

# Deploy with FastAgentic
@app.agent_endpoint(
    path="/analyze",
    runnable=LangChainAdapter(complex_chain),
)
async def analyze(input: AnalysisInput) -> AnalysisOutput:
    ...
```

## Next Steps

- [LangChain Adapter](../adapters/langchain.md) - Full adapter documentation
- [Protocol Support](../protocols/index.md) - MCP and A2A details
- [Operations Guide](../operations/index.md) - Production deployment
