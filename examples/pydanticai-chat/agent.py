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
    import ast
    import operator

    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def safe_eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants allowed")
        elif isinstance(node, ast.BinOp):
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(safe_eval(node.operand))
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
