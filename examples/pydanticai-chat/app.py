"""FastAgentic Chat Application with PydanticAI.

A simple chat agent deployed with FastAgentic, exposing REST, MCP, and A2A interfaces.
"""

from fastagentic import App, agent_endpoint, tool, resource, prompt
from fastagentic.adapters.pydanticai import PydanticAIAdapter

from agent import chat_agent
from models import ChatRequest, ChatResponse

# Create the FastAgentic app
app = App(
    title="PydanticAI Chat Agent",
    version="1.0.0",
    description="A simple chat agent powered by PydanticAI",
    # Uncomment for production:
    # oidc_issuer="https://auth.example.com",
    # durable_store="redis://localhost:6379",
    # telemetry=True,
)


# Expose tools via MCP
@tool(
    name="get_current_time",
    description="Get the current date and time",
)
async def get_current_time() -> str:
    """Return current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()


@tool(
    name="calculate",
    description="Perform basic arithmetic calculations",
)
async def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    # Simple safe eval for basic math
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid expression"
    try:
        result = eval(expression)  # Safe due to character filtering
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Expose resources via MCP
@resource(
    name="agent-info",
    uri="info",
    description="Get information about this agent",
)
async def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "name": "PydanticAI Chat Agent",
        "version": "1.0.0",
        "capabilities": ["chat", "tools", "streaming"],
        "model": "gpt-4o-mini",
    }


# Expose prompts via MCP
@prompt(
    name="chat_system",
    description="System prompt for the chat agent",
)
def chat_system_prompt() -> str:
    """Return the system prompt."""
    return """You are a helpful AI assistant. You can:
- Answer questions
- Perform calculations using the calculate tool
- Tell the current time using the get_current_time tool

Be concise and helpful in your responses."""


# Main chat endpoint
@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(chat_agent),
    input_model=ChatRequest,
    output_model=ChatResponse,
    stream=True,
    mcp_tool="chat",
    a2a_skill="chat-assistant",
    description="Chat with the AI assistant",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message and return a response."""
    pass  # Handler is provided by the adapter


# Health check (automatically added, but can customize)
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pydanticai-chat"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
