# Deep Research Demo

**Wrap any existing agent with zero code changes.**

This demo shows FastAgentic's core value proposition:

> Your agent, unchanged. Now production-safe.

## The Demo

```bash
python demo.py "quantum computing"
```

```
┌─────────────────────────────────────────────────────────────┐
│              FastAgentic Demo: Wrap Any Agent               │
├─────────────────────────────────────────────────────────────┤
│ Agent:  local_deep_research.quick_summary                   │
│ Query:  quantum computing                                   │
│ Run ID: research-7f3a2b                                     │
│                                                             │
│ This is an UNMODIFIED existing agent.                       │
│ FastAgentic wraps it with zero code changes.                │
└─────────────────────────────────────────────────────────────┘

● Running quick_summary...
✓ Complete! Cached for resume. Duration: 45.2s
```

## The Magic: Resume

Run again with the same run ID:

```bash
python demo.py "quantum computing" --run-id research-7f3a2b
```

```
✓ Found cached result for research-7f3a2b - skipping execution
```

**Instant.** No re-execution. The result was cached.

## How It Works

```python
from local_deep_research import quick_summary  # Existing agent
from fastagentic import run_opaque

# One line wraps ANY function
result = await run_opaque(
    quick_summary,           # The existing agent - unchanged
    query="quantum computing",
    run_id="research-001",   # For resume capability
)
```

That's it. Zero changes to `local_deep_research`.

## Quick Start

```bash
# Run with mock agent (no API key needed)
python demo.py

# For real LLM research, install local-deep-research
pip install local-deep-research
export OPENAI_API_KEY=sk-...
python demo.py
```

## Two Patterns

This demo shows two patterns:

### Pattern 1: Opaque Wrapping (Zero Code Changes)

```python
from fastagentic import run_opaque

# Wrap any existing agent
result = await run_opaque(existing_agent, query="...")
```

- Zero code changes to the agent
- Entire execution is one checkpoint
- Resume returns cached result if complete

### Pattern 2: Step-Aware (Fine-Grained Checkpointing)

```python
from fastagentic import run, StepTracker

async def my_agent(input, tracker: StepTracker):
    async with tracker.step("search"):
        results = await search(input)
    async with tracker.step("analyze"):
        analysis = await analyze(results)
    return analysis

result = await run(my_agent, "quantum computing")
```

- Per-step checkpointing
- Resume from exact step
- Detailed token/time tracking

## Files

| File | Description |
|------|-------------|
| `demo.py` | Main demo - opaque wrapping |
| `agent.py` | Both patterns side-by-side |
| `models.py` | Pydantic models |
| `app.py` | FastAgentic HTTP server |

## Why This Matters

| Before FastAgentic | After FastAgentic |
|-------------------|-------------------|
| Agent crashes → restart from beginning | Agent crashes → resume from checkpoint |
| No visibility into progress | See step-by-step execution |
| Pay twice for failed runs | Cached results skip re-execution |
| Agent code tightly coupled to infra | Agent code unchanged |

## The One-Liner

```python
from fastagentic import run_opaque

# Any function. Any agent. Zero changes.
result = await run_opaque(your_agent, your_input)
```
