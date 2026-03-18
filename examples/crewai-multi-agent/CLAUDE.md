# CrewAI Multi-Agent System - Claude Code Guide

This example demonstrates a multi-agent collaboration system using CrewAI.

## Project Structure

```
crewai-multi-agent/
├── CLAUDE.md          # This file
├── app.py             # FastAgentic application
├── crew.py            # CrewAI crew definition
├── agents.py          # Individual agent definitions
├── tasks.py           # Task definitions
├── models.py          # API models
├── pyproject.toml     # Dependencies
└── README.md          # Documentation
```

## Key Commands

```bash
uv sync
uv run fastagentic run
uv run fastagentic agent chat --endpoint /analyze
```

## Architecture

This example implements a **content analysis crew**:

- **Researcher Agent**: Gathers information
- **Analyst Agent**: Analyzes data
- **Writer Agent**: Creates summaries

## When Modifying

1. **Add new agent**: Create in `agents.py`, add to crew in `crew.py`
2. **Add new task**: Define in `tasks.py`, assign to agent
3. **Change tools**: Update agent's `tools` list
4. **Sequential vs Hierarchical**: Change `process` in crew definition
