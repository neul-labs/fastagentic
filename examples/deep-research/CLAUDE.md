# Deep Research Demo - Claude Code Guide

FastAgentic demo showcasing checkpoint and resume capabilities.

## Project Structure

```
deep-research/
├── demo.py          # Main CLI demo script
├── agent.py         # DeepResearchAgent with 3 phases
├── app.py           # FastAgentic HTTP server
├── models.py        # Pydantic models
├── pyproject.toml   # Dependencies
└── README.md        # User documentation
```

## Quick Commands

```bash
# Run the demo
python demo.py

# Resume a crashed run
python demo.py --resume run-abc123

# Custom topic
python demo.py "artificial intelligence"

# Start HTTP server
fastagentic run
```

## Key Components

### DeepResearchAgent (agent.py)

Three-phase research agent:
1. `search` - Gather information on topic
2. `analyze` - Identify themes and patterns
3. `synthesize` - Create final summary

Uses PydanticAI with `gpt-4o-mini`.

### StepTracker Integration

```python
async with tracker.step("search") as step:
    result = await self._search(topic)
    step.set_output(result)
    step.add_tokens(tokens_used)
```

### Checkpoint Storage

Checkpoints stored in `.checkpoints/` directory:
```
.checkpoints/
└── run-abc123/
    ├── ckpt-001.json
    └── ckpt-002.json
```

## Environment Variables

- `OPENAI_API_KEY` - Required for LLM calls

## Testing

```bash
# Run demo manually
python demo.py

# Test crash/resume
python demo.py &
# Wait for "analyze" phase
kill -INT $!
# Resume
python demo.py --resume <run_id>
```
