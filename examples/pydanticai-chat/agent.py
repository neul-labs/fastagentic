"""PydanticAI Agent Definition.

This module defines the PydanticAI agent with its system prompt and tools.
"""

from datetime import datetime

from pydantic_ai import Agent

# System prompt for the chat agent
SYSTEM_PROMPT = """You are a helpful AI assistant powered by FastAgentic.

You have access to the following tools:
- get_current_time: Get the current date and time
- calculate: Perform basic arithmetic calculations

Guidelines:
- Be concise and helpful
- Use tools when appropriate
- If you don't know something, say so
- Format responses clearly
"""

# Create the PydanticAI agent
chat_agent = Agent(
    model="openai:gpt-4o-mini",  # Or "anthropic:claude-3-haiku-20240307"
    system_prompt=SYSTEM_PROMPT,
)


# Register tools with the agent
@chat_agent.tool
async def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current timestamp in ISO format.
    """
    return datetime.now().isoformat()


@chat_agent.tool
async def calculate(expression: str) -> str:
    """Perform basic arithmetic calculations.

    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"

    Returns:
        The result of the calculation.
    """
    # Simple safe eval for basic math
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid expression. Only numbers and basic operators allowed."
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
