"""CrewAI crew definition."""

from crewai import Crew, Process
from agents import researcher, analyst, writer
from tasks import create_research_task, create_analysis_task, create_writing_task


def create_analysis_crew(topic: str) -> Crew:
    """Create a content analysis crew for the given topic.

    Args:
        topic: The topic to research and analyze

    Returns:
        A configured CrewAI crew
    """
    # Create tasks for this topic
    research_task = create_research_task(topic)
    analysis_task = create_analysis_task(topic)
    writing_task = create_writing_task(topic)

    # Create the crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # Tasks run in order
        verbose=True,
    )

    return crew


# Default crew for direct usage
analysis_crew = create_analysis_crew("general topic")
