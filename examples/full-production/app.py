"""Full Production FastAgentic Application.

This example demonstrates all production features:
- Authentication (OIDC)
- Authorization (RBAC, scopes)
- Durability (Redis checkpoints)
- Observability (OTEL, Langfuse)
- Policy (rate limits, budgets)
- Compliance (PII detection)
- HITL (approval workflows)
"""

import os

from fastagentic import (
    App,
    agent_endpoint,
    tool,
    resource,
    prompt,
    RetryPolicy,
    RateLimit,
    CircuitBreaker,
)
from fastagentic.policy import RBACPolicy, Role, Permission, BudgetPolicy, Budget
from fastagentic.compliance import PIIConfig, PIIDetectionHook
from fastagentic.hitl import ApprovalPolicy, require_confirmation
from fastagentic.adapters.pydanticai import PydanticAIAdapter

from config import settings
from agent import production_agent
from models import ChatRequest, ChatResponse
from hooks import observability_hook, audit_hook
from policies import admin_role, user_role, budget_policy

# =============================================================================
# Application Configuration
# =============================================================================

app = App(
    title="Production Agent Service",
    version="1.0.0",
    description="A production-ready agent deployment",

    # Authentication
    oidc_issuer=settings.oidc_issuer,
    oidc_audience=settings.oidc_audience,

    # Durability
    durable_store=settings.redis_url,

    # Observability
    telemetry=True,

    # Reliability
    retry_policy=RetryPolicy(
        max_attempts=3,
        backoff="exponential",
        retry_on=["rate_limit", "timeout", "server_error"],
    ),
    rate_limit=RateLimit(
        rpm=settings.rate_limit_rpm,
        tpm=settings.rate_limit_tpm,
        by="user",
    ),

    # Integrations (uncomment when configured)
    # integrations=[
    #     LangfuseIntegration(
    #         public_key=settings.langfuse_public_key,
    #         secret_key=settings.langfuse_secret_key,
    #     ),
    #     LakeraIntegration(
    #         api_key=settings.lakera_api_key,
    #         block_on_detect=True,
    #     ),
    # ],
)

# =============================================================================
# Policy Configuration
# =============================================================================

# Register policies
app.register_policy(RBACPolicy(roles=[admin_role, user_role]))
app.register_policy(budget_policy)

# PII Detection
pii_config = PIIConfig(
    detect_types=["email", "phone", "ssn", "credit_card"],
    mask_in_logs=True,
    block_pii_in_output=False,
)
app.add_hook(PIIDetectionHook(pii_config))

# Custom hooks
app.add_hook(observability_hook)
app.add_hook(audit_hook)

# =============================================================================
# Tools
# =============================================================================

@tool(
    name="search_knowledge_base",
    description="Search the knowledge base for information",
    scopes=["kb:read"],
)
async def search_kb(query: str, limit: int = 5) -> list[dict]:
    """Search knowledge base."""
    # Production: Connect to vector DB
    return [{"title": f"Result for {query}", "score": 0.9}]


@tool(
    name="create_ticket",
    description="Create a support ticket",
    scopes=["tickets:write"],
)
@require_confirmation(message="Create support ticket?")
async def create_ticket(title: str, description: str, priority: str = "medium") -> dict:
    """Create a support ticket with approval."""
    return {"ticket_id": "TKT-12345", "status": "created"}


# =============================================================================
# Resources
# =============================================================================

@resource(name="service-health", uri="health", cache_ttl=30)
async def health_check() -> dict:
    """Health check with dependency status."""
    return {
        "status": "healthy",
        "redis": "connected",
        "version": "1.0.0",
    }


@resource(name="user-profile", uri="users/{user_id}", scopes=["users:read"])
async def get_user(user_id: str) -> dict:
    """Get user profile."""
    return {"user_id": user_id, "name": "User"}


# =============================================================================
# Prompts
# =============================================================================

@prompt(name="system", description="Production system prompt")
def system_prompt() -> str:
    """System prompt with safety guidelines."""
    return """You are a helpful assistant for our production service.

Guidelines:
- Be helpful and accurate
- Never share sensitive information
- Escalate complex issues to human support
- Follow company policies
"""


# =============================================================================
# Agent Endpoints
# =============================================================================

@agent_endpoint(
    path="/chat",
    runnable=PydanticAIAdapter(production_agent),
    input_model=ChatRequest,
    output_model=ChatResponse,
    stream=True,
    durable=True,
    scopes=["chat:read", "chat:write"],
    mcp_tool="chat",
    a2a_skill="support-assistant",
    description="Chat with the production agent",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Production chat endpoint with full observability."""
    pass


@agent_endpoint(
    path="/admin/analyze",
    runnable=PydanticAIAdapter(production_agent),
    input_model=ChatRequest,
    output_model=ChatResponse,
    scopes=["admin:read"],  # Admin only
    description="Admin analysis endpoint",
)
async def admin_analyze(request: ChatRequest) -> ChatResponse:
    """Admin-only endpoint for analysis."""
    pass


# =============================================================================
# Custom Routes
# =============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastagentic.dashboard import PrometheusExporter
    exporter = PrometheusExporter()
    return exporter.export()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app.fastapi",
        host="0.0.0.0",
        port=8000,
        reload=settings.env == "dev",
        workers=1 if settings.env == "dev" else 4,
    )
