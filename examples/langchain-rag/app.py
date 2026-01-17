"""FastAgentic RAG Application with LangChain.

A retrieval-augmented generation agent using LangChain LCEL chains,
deployed with FastAgentic for REST, MCP, and streaming interfaces.
"""

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.langchain import LangChainAdapter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Create the FastAgentic app
app = App(
    title="LangChain RAG Agent",
    version="1.0.0",
    description="A RAG agent powered by LangChain LCEL chains",
    # Uncomment for production:
    # oidc_issuer="https://auth.example.com",
    # durable_store="redis://localhost:6379",
)

# Simple in-memory knowledge base
KNOWLEDGE_BASE = {
    "fastagentic": "FastAgentic is a deployment framework for AI agents. It provides REST, MCP, and A2A interfaces with built-in auth, streaming, and durability.",
    "langchain": "LangChain is a framework for developing applications powered by large language models. It provides tools for chains, agents, and retrieval.",
    "rag": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to produce more accurate and grounded responses.",
    "lcel": "LCEL (LangChain Expression Language) is a declarative way to compose chains using the pipe operator.",
}


def retrieve_context(question: str) -> str:
    """Simple keyword-based retrieval from knowledge base."""
    relevant = []
    question_lower = question.lower()
    for topic, content in KNOWLEDGE_BASE.items():
        if topic in question_lower or any(word in question_lower for word in topic.split()):
            relevant.append(f"**{topic.title()}**: {content}")

    if not relevant:
        return "No specific context found. Please provide a general answer."
    return "\n\n".join(relevant)


# Create the LangChain LCEL chain
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:""")

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

chain = (
    {"context": lambda x: retrieve_context(x["question"]), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Expose tools via MCP
@tool(
    name="search_knowledge",
    description="Search the knowledge base for information",
)
async def search_knowledge(query: str) -> str:
    """Search the knowledge base."""
    return retrieve_context(query)


# Expose resources via MCP
@resource(
    name="knowledge-topics",
    uri="topics",
    description="List available topics in the knowledge base",
)
async def get_topics() -> dict:
    """Return available topics."""
    return {
        "topics": list(KNOWLEDGE_BASE.keys()),
        "count": len(KNOWLEDGE_BASE),
    }


# Main RAG endpoint
@agent_endpoint(
    path="/ask",
    runnable=LangChainAdapter(chain),
    stream=True,
    durable=True,
    mcp_tool="ask",
    description="Ask a question using RAG",
)
async def ask(question: str) -> str:
    """Answer a question using retrieval-augmented generation."""
    pass  # Handler is provided by the adapter


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "langchain-rag"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
