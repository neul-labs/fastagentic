"""FastAgentic Chat Application with Semantic Kernel.

A chat agent with plugins using Microsoft Semantic Kernel,
deployed with FastAgentic for REST, MCP, and streaming interfaces.
"""

import os
from datetime import datetime

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.semantic_kernel import SemanticKernelAdapter

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

# Create the Semantic Kernel
kernel = sk.Kernel()

# Add OpenAI chat completion service
kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
)


# Define a plugin with useful functions
class UtilityPlugin:
    """A plugin with utility functions for the assistant."""

    @kernel_function(name="get_time", description="Get the current date and time")
    def get_time(self) -> str:
        """Return the current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @kernel_function(name="calculate", description="Perform arithmetic calculations")
    def calculate(self, expression: str) -> str:
        """Safely evaluate a math expression."""
        import ast
        import operator

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def safe_eval(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            elif isinstance(node, ast.BinOp):
                op = operators.get(type(node.op))
                if op:
                    return op(safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                op = operators.get(type(node.op))
                if op:
                    return op(safe_eval(node.operand))
            elif isinstance(node, ast.Expression):
                return safe_eval(node.body)
            raise ValueError("Invalid expression")

        try:
            tree = ast.parse(expression, mode='eval')
            return str(safe_eval(tree))
        except Exception as e:
            return f"Error: {e}"

    @kernel_function(name="format_text", description="Format text in different styles")
    def format_text(self, text: str, style: str = "upper") -> str:
        """Format text in the specified style."""
        if style == "upper":
            return text.upper()
        elif style == "lower":
            return text.lower()
        elif style == "title":
            return text.title()
        elif style == "reverse":
            return text[::-1]
        return text


# Register the plugin with the kernel
kernel.add_plugin(UtilityPlugin(), "Utility")


# Create the FastAgentic app
app = App(
    title="Semantic Kernel Chat",
    version="1.0.0",
    description="A chat agent powered by Microsoft Semantic Kernel",
    # Uncomment for production:
    # oidc_issuer="https://auth.example.com",
    # durable_store="redis://localhost:6379",
)


# Expose tools via MCP
@tool(
    name="get_current_time",
    description="Get the current date and time",
)
async def get_current_time() -> str:
    """Return current timestamp."""
    return datetime.now().isoformat()


@tool(
    name="calculate_math",
    description="Perform arithmetic calculations",
)
async def calculate_math(expression: str) -> str:
    """Calculate a math expression."""
    plugin = UtilityPlugin()
    return plugin.calculate(expression)


# Expose resources via MCP
@resource(
    name="available-plugins",
    uri="plugins",
    description="List available Semantic Kernel plugins",
)
async def get_plugins() -> dict:
    """Return available plugins."""
    plugins = []
    for name, plugin in kernel.plugins.items():
        functions = [f.name for f in plugin.functions.values()]
        plugins.append({"name": name, "functions": functions})
    return {"plugins": plugins}


@resource(
    name="agent-info",
    uri="info",
    description="Get information about this agent",
)
async def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "name": "Semantic Kernel Chat",
        "version": "1.0.0",
        "capabilities": ["chat", "plugins", "streaming"],
        "model": "gpt-4o-mini",
        "plugins": list(kernel.plugins.keys()),
    }


# Main chat endpoint
@agent_endpoint(
    path="/chat",
    runnable=SemanticKernelAdapter(kernel),
    stream=True,
    durable=True,
    mcp_tool="chat",
    description="Chat with the AI assistant",
)
async def chat(message: str) -> str:
    """Chat with the Semantic Kernel assistant."""
    pass  # Handler is provided by the adapter


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "semantic-kernel-chat",
        "plugins": list(kernel.plugins.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
