# Braintrust Integration

[Braintrust](https://braintrust.dev) provides experiment tracking, scoring, and evaluation for AI applications. FastAgentic integrates via the `BraintrustHook` for production evaluation.

## Installation

```bash
pip install fastagentic[braintrust]
```

## Quick Start

```python
from fastagentic import agent_endpoint
from fastagentic.integrations.braintrust import BraintrustHook

@agent_endpoint(
    path="/triage",
    runnable=...,
    eval_hooks=[
        BraintrustHook(
            api_key="...",
            project="support-triage",
        ),
    ],
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

## Configuration

### Environment Variables

```bash
export BRAINTRUST_API_KEY="..."
```

```python
BraintrustHook(project="my-project")  # Auto-reads from env
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `api_key` | Braintrust API key | `$BRAINTRUST_API_KEY` |
| `project` | Project name | Required |
| `experiment` | Experiment name | Auto-generated |
| `scores` | Score functions to run | None |
| `async_mode` | Run evaluation async | `true` |
| `sample_rate` | Fraction of requests to evaluate | `1.0` |

## Evaluation Modes

### Async (Default)

Evaluation runs in background, doesn't delay response:

```python
BraintrustHook(
    project="triage",
    async_mode=True,  # Default
)
```

### Sync

Wait for evaluation before returning response:

```python
BraintrustHook(
    project="triage",
    async_mode=False,  # Adds latency
)
```

## Built-in Scorers

```python
from fastagentic.integrations.braintrust import (
    BraintrustHook,
    Factuality,
    Relevance,
    Coherence,
    Safety,
)

BraintrustHook(
    project="triage",
    scores=[
        Factuality(),      # Check factual accuracy
        Relevance(),       # Response relevance to input
        Coherence(),       # Logical consistency
        Safety(),          # Content safety
    ],
)
```

## Custom Scorers

```python
from fastagentic.integrations.braintrust import Scorer

class TriageAccuracy(Scorer):
    name = "triage_accuracy"

    async def score(self, input: dict, output: dict, expected: dict = None) -> float:
        # Custom scoring logic
        if expected and output.get("priority") == expected.get("priority"):
            return 1.0
        return 0.0

BraintrustHook(
    project="triage",
    scores=[TriageAccuracy()],
)
```

## LLM-as-Judge

Use an LLM to evaluate responses:

```python
from fastagentic.integrations.braintrust import LLMJudge

BraintrustHook(
    project="triage",
    scores=[
        LLMJudge(
            model="gpt-4o-mini",
            criteria="Rate the response quality from 0-1",
            rubric="""
            1.0: Excellent - Correctly prioritized with clear reasoning
            0.7: Good - Correct priority, minimal reasoning
            0.3: Poor - Wrong priority or unclear
            0.0: Unacceptable - Completely wrong or harmful
            """,
        ),
    ],
)
```

## Expected Values

Provide expected outputs for comparison:

```python
@agent_endpoint(
    path="/triage",
    runnable=...,
    eval_hooks=[BraintrustHook(project="triage")],
)
async def triage(ticket: TicketIn, ctx: AgentContext) -> TicketOut:
    # Set expected value for evaluation
    ctx.eval_expected = {
        "priority": "high",
        "category": "billing",
    }
    return await run_triage(ticket)
```

## Experiment Tracking

Track experiments across deployments:

```python
import os

BraintrustHook(
    project="triage",
    experiment=f"v{os.getenv('VERSION')}-{os.getenv('ENV')}",
    metadata={
        "model": "gpt-4o",
        "prompt_version": "v2.1",
        "git_sha": os.getenv("GIT_SHA"),
    },
)
```

## Datasets

Use Braintrust datasets for evaluation:

```python
from fastagentic.integrations.braintrust import BraintrustEval

# Run evaluation against dataset
async def run_eval():
    evaluator = BraintrustEval(
        project="triage",
        dataset="triage-golden-set",
        endpoint="http://localhost:8000/triage",
    )

    results = await evaluator.run()
    print(f"Accuracy: {results.accuracy:.2%}")
    print(f"Avg latency: {results.avg_latency_ms}ms")
```

## Sampling

For high-volume endpoints, sample evaluations:

```python
BraintrustHook(
    project="triage",
    sample_rate=0.1,  # Evaluate 10% of requests
)
```

## Feedback Loop

Send user feedback to Braintrust:

```python
from fastagentic import App
from fastagentic.integrations.braintrust import BraintrustClient

braintrust = BraintrustClient(api_key="...")

@app.post("/feedback/{run_id}")
async def submit_feedback(run_id: str, score: float, comment: str = None):
    await braintrust.log_feedback(
        run_id=run_id,
        score=score,
        comment=comment,
    )
    return {"status": "recorded"}
```

## Integration with Other Hooks

```python
from fastagentic import App
from fastagentic.integrations.langfuse import LangfuseHook
from fastagentic.integrations.braintrust import BraintrustHook

app = App(
    hooks=[
        # Langfuse for tracing
        LangfuseHook(...),
    ],
)

@agent_endpoint(
    path="/triage",
    runnable=...,
    eval_hooks=[
        # Braintrust for evaluation (async, separate from tracing)
        BraintrustHook(project="triage"),
    ],
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

## Soft Failure Thresholds

Don't hard-fail on borderline scores:

```python
BraintrustHook(
    project="triage",
    scores=[Relevance()],
    thresholds={
        "relevance": {
            "fail": 0.3,   # Reject below 0.3
            "warn": 0.5,   # Log warning below 0.5
            "pass": 0.7,   # Consider good above 0.7
        },
    },
    on_fail="log",  # or "reject"
)
```

## Metrics

```
fastagentic_eval_runs_total{project="triage"} 5234
fastagentic_eval_score{project="triage", scorer="relevance", quantile="p50"} 0.82
fastagentic_eval_score{project="triage", scorer="relevance", quantile="p10"} 0.45
fastagentic_eval_latency_ms{project="triage", quantile="p99"} 234
```

## Alternative: LangSmith

```python
from fastagentic.integrations.langsmith import LangSmithHook

@agent_endpoint(
    path="/triage",
    runnable=...,
    eval_hooks=[
        LangSmithHook(
            project="triage",
            evaluators=["relevance", "coherence"],
        ),
    ],
)
```

## Troubleshooting

### Scores not appearing

- Check API key permissions
- Verify project name exists
- Enable debug logging

### High latency

- Use `async_mode=True` (default)
- Reduce scorer complexity
- Sample high-volume endpoints

## Next Steps

- [Braintrust Docs](https://www.braintrust.dev/docs)
- [LangSmith Integration](langsmith.md)
- [Hooks Architecture](../hooks.md)
