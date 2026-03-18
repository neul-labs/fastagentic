"""CrewAI task definitions."""

from crewai import Task
from agents import researcher, analyst, writer


def create_research_task(topic: str) -> Task:
    """Create a research task for the given topic."""
    return Task(
        description=f"""Research the following topic thoroughly: {topic}

        Gather key facts, recent developments, and important context.
        Focus on reliable sources and verified information.""",
        expected_output="A comprehensive research summary with key facts and sources",
        agent=researcher,
    )


def create_analysis_task(topic: str) -> Task:
    """Create an analysis task."""
    return Task(
        description=f"""Analyze the research findings about: {topic}

        Identify patterns, insights, and implications.
        Highlight the most important takeaways.""",
        expected_output="An analysis with key insights and patterns identified",
        agent=analyst,
    )


def create_writing_task(topic: str) -> Task:
    """Create a writing task."""
    return Task(
        description=f"""Write a clear, engaging summary about: {topic}

        Use the research and analysis to create content that's
        informative and easy to understand.""",
        expected_output="A well-written summary suitable for a general audience",
        agent=writer,
    )
