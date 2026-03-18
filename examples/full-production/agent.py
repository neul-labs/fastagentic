"""Production agent definition."""

from pydantic_ai import Agent

SYSTEM_PROMPT = """You are a production support assistant.

Guidelines:
- Be helpful, accurate, and professional
- Use available tools when appropriate
- Never share sensitive information
- Escalate complex issues to human support
"""

production_agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
)


@production_agent.tool
async def get_current_time() -> str:
    """Get current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()
