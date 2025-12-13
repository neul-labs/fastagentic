"""Simple FastAgentic example application.

This example demonstrates the basic usage of FastAgentic with decorators.

Run with:
    cd examples
    fastagentic run simple_app:app --reload
"""

from fastagentic import App, agent_endpoint, prompt, resource, tool

# Create the application
app = App(
    title="Simple Agent",
    version="0.1.0",
    description="A simple FastAgentic example",
)


# Define a tool
@tool(name="greet", description="Greet someone by name")
async def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}! Welcome to FastAgentic."


@tool(name="add_numbers", description="Add two numbers together")
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Define a resource
@resource(
    name="status",
    uri="status",
    description="Get the current application status",
    cache_ttl=60,
)
async def get_status() -> dict:
    """Return the application status."""
    return {
        "status": "healthy",
        "version": app.config.version,
        "name": app.config.title,
    }


# Define a prompt
@prompt(
    name="assistant_prompt",
    description="System prompt for the assistant",
)
def assistant_prompt(context: str = "") -> str:
    """Generate the assistant system prompt."""
    base = "You are a helpful assistant powered by FastAgentic."
    if context:
        return f"{base}\n\nAdditional context: {context}"
    return base


# A simple endpoint (without an adapter)
@app.post("/echo")
async def echo(message: str) -> dict:
    """Echo the message back."""
    return {"echo": message}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app.fastapi, host="127.0.0.1", port=8000, reload=True)
