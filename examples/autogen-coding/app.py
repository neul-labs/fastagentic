"""FastAgentic Coding Assistant with AutoGen.

A multi-agent coding assistant using Microsoft AutoGen,
deployed with FastAgentic for REST, MCP, and streaming interfaces.
"""

import os

from fastagentic import App, agent_endpoint, tool, resource
from fastagentic.adapters.autogen import AutoGenAdapter

from autogen import AssistantAgent, UserProxyAgent

# LLM configuration
llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

# Create the coding assistant agent
assistant = AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="""You are a helpful coding assistant. You can:
- Write and explain code in any language
- Debug and fix issues
- Suggest improvements and best practices
- Answer programming questions

When writing code, provide clear explanations and well-commented code.
If the user asks to execute code, generate the code but note that execution
is handled by the user proxy agent.
""",
)

# Create the user proxy agent (handles code execution)
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",  # No human input needed in API context
    max_consecutive_auto_reply=5,
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False,  # Set to True for isolated execution
    },
)


# Create the FastAgentic app
app = App(
    title="AutoGen Coding Assistant",
    version="1.0.0",
    description="A coding assistant powered by Microsoft AutoGen",
    # Uncomment for production:
    # oidc_issuer="https://auth.example.com",
    # durable_store="redis://localhost:6379",
)


# Expose tools via MCP
@tool(
    name="explain_code",
    description="Get an explanation of code",
)
async def explain_code(code: str, language: str = "python") -> str:
    """Explain what a piece of code does."""
    from autogen import ChatResult

    result: ChatResult = await user_proxy.a_initiate_chat(
        assistant,
        message=f"Please explain this {language} code:\n```{language}\n{code}\n```",
        max_turns=2,
    )
    return result.chat_history[-1]["content"]


@tool(
    name="review_code",
    description="Get a code review with suggestions",
)
async def review_code(code: str, language: str = "python") -> str:
    """Review code and provide suggestions."""
    from autogen import ChatResult

    result: ChatResult = await user_proxy.a_initiate_chat(
        assistant,
        message=f"Please review this {language} code and suggest improvements:\n```{language}\n{code}\n```",
        max_turns=2,
    )
    return result.chat_history[-1]["content"]


# Expose resources via MCP
@resource(
    name="agent-info",
    uri="info",
    description="Get information about this agent",
)
async def get_agent_info() -> dict:
    """Return agent metadata."""
    return {
        "name": "AutoGen Coding Assistant",
        "version": "1.0.0",
        "capabilities": ["coding", "code-execution", "debugging", "streaming"],
        "model": "gpt-4o-mini",
        "agents": ["coder", "user"],
    }


@resource(
    name="supported-languages",
    uri="languages",
    description="List supported programming languages",
)
async def get_languages() -> dict:
    """Return supported languages."""
    return {
        "languages": [
            "python",
            "javascript",
            "typescript",
            "rust",
            "go",
            "java",
            "c++",
            "shell",
        ]
    }


# Main coding endpoint
@agent_endpoint(
    path="/code",
    runnable=AutoGenAdapter(
        initiator=user_proxy,
        recipient=assistant,
        max_turns=5,
        checkpoint_turns=True,
    ),
    stream=True,
    durable=True,
    mcp_tool="code",
    description="Get coding help from the AI assistant",
)
async def code(task: str) -> str:
    """Get help with a coding task."""
    pass  # Handler is provided by the adapter


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "autogen-coding",
        "agents": ["coder", "user"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app.fastapi", host="0.0.0.0", port=8000, reload=True)
