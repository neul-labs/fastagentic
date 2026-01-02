# Protocol Support

FastAgentic implements two complementary protocols for agentic interoperability:

| Protocol | Purpose | FastAgentic Role |
|----------|---------|------------------|
| **MCP** (Model Context Protocol) | Tool and resource access | Expose agent capabilities to LLM hosts |
| **A2A** (Agent-to-Agent) | Agent collaboration | Enable agents to discover and delegate to each other |

## Why Two Protocols?

**MCP** answers: "What tools and resources can this agent use?"
- Connects LLM applications to external capabilities
- Tools, resources, and prompts exposed to AI models
- Single agent ↔ host relationship

**A2A** answers: "What other agents can this agent collaborate with?"
- Connects agents to other agents
- Task delegation and multi-agent workflows
- Many-to-many agent relationships

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Host (Claude, etc.)                │
└─────────────────────────────────────────────────────────────┘
                              │ MCP
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAgentic Runtime                       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Agent Registry (A2A)                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ Agent A  │  │ Agent B  │  │ Agent C  │   ...    │   │
│  │  │(Pydantic)│  │(LangGraph│  │ (CrewAI) │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ▲  │  ▲                             │
│                    A2A  │  │  │  A2A                        │
│                         └──┴──┘                             │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         External         External       External
         A2A Agent        A2A Agent      A2A Agent
```

## Protocol Versions

FastAgentic tracks the latest stable protocol specifications:

| Protocol | Version | Release Date | Key Features |
|----------|---------|--------------|--------------|
| MCP | 2025-11-25 | November 2025 | Tasks, Extensions, OAuth, Parallel Tools |
| A2A | 0.3 | August 2025 | gRPC, Signed Cards, Extended SDK |

## Unified Benefits

By implementing both protocols, FastAgentic provides:

1. **Single Definition, Dual Exposure**
   - Define your agent once with decorators
   - Automatically exposed via MCP (for LLM hosts) and A2A (for other agents)

2. **Internal Agent Registry**
   - Agents deployed in FastAgentic can discover each other
   - A2A Agent Cards generated from adapter metadata
   - Task delegation without external orchestration

3. **External Interoperability**
   - Connect to any A2A-compliant agent (LangChain, Vertex AI, Azure AI Foundry)
   - Expose capabilities to any MCP-compliant host (Claude Desktop, IDEs)

4. **Unified Authentication**
   - Single OAuth2/OIDC configuration
   - MCP authorization and A2A security schemes aligned
   - Consistent identity across protocol boundaries

## Quick Start

```python
from fastagentic import App
from fastagentic.adapters.pydanticai import PydanticAIAdapter
from fastagentic.protocols import enable_a2a, enable_mcp

app = App(
    title="Support Triage",
    version="1.0.0",
)

# Enable both protocols
enable_mcp(app)      # /mcp/schema, /mcp/tools, stdio transport
enable_a2a(app)      # /.well-known/agent.json, /a2a/tasks

# Register an agent - automatically exposed via both protocols
@app.agent_endpoint(
    path="/triage",
    runnable=PydanticAIAdapter(triage_agent),
    a2a_skill="support-triage",  # A2A skill name
    mcp_tool="triage_ticket",    # MCP tool name
)
async def triage(ticket: TicketIn) -> TicketOut:
    ...
```

## Learn More

- [MCP Implementation](mcp.md) - Model Context Protocol details
- [A2A Integration](a2a.md) - Agent-to-Agent protocol details
- [Templates](../templates/index.md) - Starter projects for each framework

## Sources

Protocol specifications:
- [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25)
- [A2A Specification (v0.3)](https://a2a-protocol.org/latest/specification/)
- [A2A GitHub](https://github.com/google-a2a/A2A)
