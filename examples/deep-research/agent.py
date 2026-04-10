"""Deep Research Agent Examples.

This module demonstrates TWO patterns for using FastAgentic:

1. OPAQUE WRAPPING (Zero code changes)
   - Wrap any existing agent function
   - Entire execution is one checkpoint
   - Resume skips if already complete

2. STEP-AWARE (Fine-grained checkpointing)
   - Agent uses StepTracker
   - Each step is checkpointed
   - Resume continues from exact step

Choose based on your needs:
- Opaque: Quick wins, existing agents, simple use cases
- Step-aware: Long workflows, need mid-run resume, observability
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# PATTERN 1: OPAQUE WRAPPING (Zero code changes)
# =============================================================================

# Import existing agent - UNCHANGED
try:
    from local_deep_research import quick_summary, generate_report
    HAS_LOCAL_DEEP_RESEARCH = True
except ImportError:
    HAS_LOCAL_DEEP_RESEARCH = False

    # Mock functions for when local-deep-research is not installed
    def quick_summary(query: str, **kwargs) -> dict:
        """Mock quick_summary for demo."""
        return {"summary": f"Mock summary for: {query}", "sources": []}

    def generate_report(query: str, **kwargs) -> dict:
        """Mock generate_report for demo."""
        return {"report": f"Mock report for: {query}", "sources": []}


async def research_opaque(topic: str, run_id: str | None = None) -> dict:
    """Research using opaque wrapping - zero code changes to the agent.

    The entire local_deep_research execution is wrapped as a single unit.
    If the run completes, the result is cached. Resume returns cached result.

    Example:
        result = await research_opaque("quantum computing")
        # If this exact run_id was already completed, returns cached result instantly
        result = await research_opaque("quantum computing", run_id="research-001")
    """
    from fastagentic import run_opaque, FileCheckpointStore

    return await run_opaque(
        quick_summary,
        query=topic,
        run_id=run_id,
        store=FileCheckpointStore(".checkpoints"),
    )


# =============================================================================
# PATTERN 2: STEP-AWARE (Fine-grained checkpointing)
# =============================================================================

from pydantic_ai import Agent

from fastagentic.runner import StepTracker

from models import Analysis, ResearchResult, SearchFindings, SearchResult

# System prompts for each phase
SEARCH_PROMPT = """You are a research assistant focused on gathering information.
Given a topic, generate relevant search queries and synthesize findings.
Be thorough but concise."""

ANALYZE_PROMPT = """You are an analytical research assistant.
Given research findings, identify key themes, main points, and gaps.
Provide structured analysis with clear reasoning."""

SYNTHESIZE_PROMPT = """You are a research synthesis expert.
Given research findings and analysis, create a comprehensive summary.
Write in a clear, authoritative style suitable for a research report."""


class DeepResearchAgent:
    """Multi-step research agent with fine-grained checkpointing.

    This pattern gives you:
    - Per-step checkpointing (resume from exact step)
    - Token tracking per phase
    - Step-level observability

    Use this when:
    - Your workflow has distinct phases
    - You need to resume from mid-execution
    - You want detailed cost/time tracking per step
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        """Initialize the research agent."""
        self.model = model
        self._search_agent = Agent(model, system_prompt=SEARCH_PROMPT)
        self._analyze_agent = Agent(model, system_prompt=ANALYZE_PROMPT)
        self._synthesize_agent = Agent(model, system_prompt=SYNTHESIZE_PROMPT)

    async def run(self, topic: str, tracker: StepTracker) -> ResearchResult:
        """Execute the full research pipeline with step tracking.

        Each phase is checkpointed. If interrupted, resume continues
        from the last completed step.
        """
        # Phase 1: Search
        async with tracker.step("search") as step:
            findings = await self._search(topic)
            step.set_output(findings.model_dump())
            step.add_tokens(findings.tokens_used)

        # Phase 2: Analyze
        async with tracker.step("analyze") as step:
            prev_findings = step.get_previous_output("search")
            if prev_findings:
                findings = SearchFindings.model_validate(prev_findings)

            analysis = await self._analyze(findings)
            step.set_output(analysis.model_dump())
            step.add_tokens(analysis.tokens_used)

        # Phase 3: Synthesize
        async with tracker.step("synthesize") as step:
            prev_analysis = step.get_previous_output("analyze")
            if prev_analysis:
                analysis = Analysis.model_validate(prev_analysis)

            synthesis = await self._synthesize(topic, findings, analysis)
            step.set_output(synthesis)

        return ResearchResult(
            topic=topic,
            findings=findings,
            analysis=analysis,
            synthesis=synthesis,
            total_tokens=tracker.graph.total_tokens,
            total_duration_ms=tracker.graph.total_duration_ms,
        )

    async def _search(self, topic: str) -> SearchFindings:
        """Phase 1: Search for information on the topic."""
        prompt = f"Research the following topic thoroughly: {topic}"
        result = await self._search_agent.run(prompt)
        tokens = result.usage().total_tokens if result.usage() else 0

        return SearchFindings(
            query=topic,
            results=[
                SearchResult(
                    title=f"Research on {topic}",
                    snippet=str(result.data)[:500],
                    source="LLM synthesis",
                ),
            ],
            tokens_used=tokens,
        )

    async def _analyze(self, findings: SearchFindings) -> Analysis:
        """Phase 2: Analyze the search findings."""
        prompt = f"Analyze these research findings:\n{findings.model_dump_json(indent=2)}"
        result = await self._analyze_agent.run(prompt)
        tokens = result.usage().total_tokens if result.usage() else 0

        return Analysis(
            key_themes=["Theme 1", "Theme 2", "Theme 3"],
            main_points=["Point 1", "Point 2", "Point 3"],
            gaps=["Gap 1", "Gap 2"],
            tokens_used=tokens,
        )

    async def _synthesize(
        self,
        topic: str,
        findings: SearchFindings,
        analysis: Analysis,
    ) -> str:
        """Phase 3: Synthesize final research summary."""
        prompt = f"Create a research summary on: {topic}\nBased on: {analysis.model_dump_json()}"
        result = await self._synthesize_agent.run(prompt)
        return str(result.data)


# =============================================================================
# COMPARISON: When to use each pattern
# =============================================================================

"""
OPAQUE WRAPPING (run_opaque)
============================
Pros:
- Zero code changes to existing agents
- Drop-in replacement for any function
- Simple mental model

Cons:
- Resume restarts entire agent (can't resume mid-execution)
- No per-step observability
- All-or-nothing caching

Use when:
- Wrapping existing agents you don't control
- Simple, fast operations
- You just need "did it complete?" semantics


STEP-AWARE (StepTracker)
========================
Pros:
- Resume from exact step
- Per-step token/time tracking
- Fine-grained observability

Cons:
- Requires StepTracker integration
- More code to write
- Agent must be designed with steps

Use when:
- Building new multi-step workflows
- Long-running operations (30+ minutes)
- Need detailed cost analysis
- Want true crash recovery
"""

# Create default agent instances
research_agent = DeepResearchAgent()
