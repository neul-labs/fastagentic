"""FastAgentic Document Q&A Application with LlamaIndex.

A document question-answering agent using LlamaIndex query engines,
deployed with FastAgentic for REST, MCP, and streaming interfaces.
"""

import os
from pathlib import Path

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.llamaindex import LlamaIndexAdapter

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure LlamaIndex settings
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# Load documents from the data directory
data_dir = Path(__file__).parent / "data"
if data_dir.exists():
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents)
else:
    # Create empty index if no data directory
    index = VectorStoreIndex([])

# Create query engine with streaming
query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

# Create the FastAgentic app
app = App(
    title="LlamaIndex Document Q&A",
    version="1.0.0",
    description="A document Q&A agent powered by LlamaIndex",
    # Uncomment for production:
    # oidc_issuer="https://auth.example.com",
    # durable_store="redis://localhost:6379",
)


# Expose tools via MCP
@tool(
    name="search_documents",
    description="Search through the document collection",
)
async def search_documents(query: str) -> str:
    """Search documents and return relevant excerpts."""
    response = query_engine.query(query)
    sources = []
    for node in response.source_nodes:
        sources.append(f"- {node.text[:200]}...")
    return f"Answer: {response}\n\nSources:\n" + "\n".join(sources)


# Expose resources via MCP
@resource(
    name="document-stats",
    uri="stats",
    description="Get statistics about the document collection",
)
async def get_document_stats() -> dict:
    """Return document collection statistics."""
    return {
        "document_count": len(documents) if data_dir.exists() else 0,
        "index_type": "VectorStoreIndex",
        "similarity_top_k": 3,
    }


@resource(
    name="agent-info",
    uri="info",
    description="Get information about this agent",
)
async def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "name": "LlamaIndex Document Q&A",
        "version": "1.0.0",
        "capabilities": ["document-qa", "source-tracking", "streaming"],
        "model": "gpt-4o-mini",
    }


# Main query endpoint
@agent_endpoint(
    path="/query",
    runnable=LlamaIndexAdapter(query_engine=query_engine),
    stream=True,
    durable=True,
    mcp_tool="query",
    description="Query the document collection",
)
async def query(question: str) -> str:
    """Query documents and get an answer with sources."""
    pass  # Handler is provided by the adapter


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    doc_count = len(documents) if data_dir.exists() else 0
    return {
        "status": "healthy",
        "service": "llamaindex-docs",
        "documents": doc_count,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
