"""FastAgentic Reasoning Application with DSPy.

A chain-of-thought reasoning agent using DSPy modules,
deployed with FastAgentic for REST, MCP, and streaming interfaces.
"""

import os

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.dspy import DSPyAdapter

import dspy

# Configure DSPy with the language model
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)


# Define the QA signature with reasoning
class ReasoningQA(dspy.Signature):
    """Answer questions with step-by-step reasoning."""

    question: str = dspy.InputField(desc="The question to answer")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
    answer: str = dspy.OutputField(desc="The final answer")


# Create the DSPy module with Chain of Thought
reasoning_module = dspy.ChainOfThought(ReasoningQA)


# Create the FastAgentic app
app = App(
    title="DSPy Reasoning Agent",
    version="1.0.0",
    description="A reasoning agent powered by DSPy Chain of Thought",
    # Uncomment for production:
    # oidc_issuer="https://auth.example.com",
    # durable_store="redis://localhost:6379",
)


# Expose tools via MCP
@tool(
    name="explain_concept",
    description="Explain a concept using reasoning",
)
async def explain_concept(concept: str) -> str:
    """Explain a concept with reasoning."""
    result = reasoning_module(question=f"Explain the concept of {concept}")
    return f"Reasoning: {result.reasoning}\n\nAnswer: {result.answer}"


# Expose resources via MCP
@resource(
    name="agent-info",
    uri="info",
    description="Get information about the reasoning agent",
)
async def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "name": "DSPy Reasoning Agent",
        "version": "1.0.0",
        "capabilities": ["reasoning", "chain-of-thought", "streaming"],
        "model": "gpt-4o-mini",
        "module": "ChainOfThought",
    }


# Main reasoning endpoint
@agent_endpoint(
    path="/ask",
    runnable=DSPyAdapter(reasoning_module, stream_fields=["reasoning", "answer"]),
    stream=True,
    durable=True,
    mcp_tool="reason",
    description="Answer a question with step-by-step reasoning",
)
async def ask(question: str) -> str:
    """Answer a question using chain-of-thought reasoning."""
    pass  # Handler is provided by the adapter


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "dspy-reasoning"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
