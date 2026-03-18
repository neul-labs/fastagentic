"""LangGraph workflow definition.

This module defines the research workflow as a LangGraph StateGraph.
"""

from langgraph.graph import StateGraph, END

from state import ResearchState
from nodes import (
    plan_node,
    research_node,
    analyze_node,
    summarize_node,
    should_continue_research,
)

# Create the graph
graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("plan", plan_node)
graph.add_node("research", research_node)
graph.add_node("analyze", analyze_node)
graph.add_node("summarize", summarize_node)

# Add edges
graph.set_entry_point("plan")
graph.add_edge("plan", "research")

# Conditional edge: continue research or analyze
graph.add_conditional_edges(
    "research",
    should_continue_research,
    {
        "research": "research",
        "analyze": "analyze",
    },
)

graph.add_edge("analyze", "summarize")
graph.add_edge("summarize", END)

# Compile the graph
research_workflow = graph.compile()


# For visualization
if __name__ == "__main__":
    # Print ASCII representation
    print("Research Workflow Graph:")
    print(research_workflow.get_graph().draw_ascii())
