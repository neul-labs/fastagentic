"""Workflow nodes for the research agent.

Each node is a function that takes state and returns updated state.
"""

from langchain_openai import ChatOpenAI

from state import ResearchState

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def plan_node(state: ResearchState) -> dict:
    """Create a research plan based on the query.

    This node analyzes the query and creates a structured plan
    for how to research the topic.
    """
    prompt = f"""You are a research planner. Given this query, create a brief research plan.

Query: {state.query}

Create a plan with 2-3 research steps. Be concise."""

    response = await llm.ainvoke(prompt)
    return {"plan": response.content}


async def research_node(state: ResearchState) -> dict:
    """Conduct research based on the plan.

    This node simulates research by generating findings.
    In production, this would call external APIs, databases, etc.
    """
    iteration = state.iteration + 1

    prompt = f"""You are a researcher. Based on the plan, generate one key finding.

Query: {state.query}
Plan: {state.plan}
Previous findings: {state.findings}
Iteration: {iteration}

Generate ONE new finding (2-3 sentences). Don't repeat previous findings."""

    response = await llm.ainvoke(prompt)
    return {
        "findings": [response.content],
        "iteration": iteration,
    }


async def analyze_node(state: ResearchState) -> dict:
    """Analyze the collected research findings.

    This node synthesizes all findings into a coherent analysis.
    """
    findings_text = "\n".join(f"- {f}" for f in state.findings)

    prompt = f"""You are an analyst. Analyze these research findings.

Query: {state.query}
Findings:
{findings_text}

Provide a brief analysis (3-4 sentences) synthesizing the key insights."""

    response = await llm.ainvoke(prompt)
    return {"analysis": response.content}


async def summarize_node(state: ResearchState) -> dict:
    """Create a final summary of the research.

    This node produces the final output for the user.
    """
    prompt = f"""You are a writer. Create a final summary of this research.

Query: {state.query}
Analysis: {state.analysis}

Write a concise summary (2-3 sentences) answering the original query."""

    response = await llm.ainvoke(prompt)
    return {"summary": response.content}


def should_continue_research(state: ResearchState) -> str:
    """Decide whether to continue researching or move to analysis.

    Returns:
        "research" to continue, "analyze" to move forward
    """
    if state.iteration < state.max_iterations and len(state.findings) < 3:
        return "research"
    return "analyze"
