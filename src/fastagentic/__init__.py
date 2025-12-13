"""FastAgentic - The deployment layer for agentic applications.

Build agents with anything. Ship them with FastAgentic.
"""

from fastagentic.app import App
from fastagentic.decorators import agent_endpoint, prompt, resource, tool
from fastagentic.context import AgentContext, RunContext
from fastagentic.reliability import (
    RetryPolicy,
    Timeout,
    CircuitBreaker,
    FallbackChain,
    RateLimit,
)
from fastagentic.policy import (
    Policy,
    PolicyContext,
    PolicyResult,
    PolicyAction,
    PolicyEngine,
    Role,
    Permission,
    RBACPolicy,
    ScopePolicy,
    Scope,
    BudgetPolicy,
    Budget,
    BudgetPeriod,
)
from fastagentic.cost import CostTracker, CostRecord, ModelPricing
from fastagentic.audit import AuditLogger, AuditEvent, AuditEventType, AuditSeverity
from fastagentic.prompts import (
    PromptTemplate,
    PromptVariable,
    PromptRegistry,
    PromptVersion,
    ABTest,
    PromptVariant,
)
from fastagentic.hitl import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalPolicy,
    ApprovalStatus,
    EscalationManager,
    EscalationTrigger,
    EscalationLevel,
    require_confirmation,
)
from fastagentic.cluster import (
    Worker,
    WorkerStatus,
    WorkerConfig,
    WorkerPool,
    Task,
    TaskStatus,
    TaskResult,
    TaskQueue,
    Coordinator,
    CoordinatorConfig,
)
from fastagentic.checkpoint import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointStore,
    CheckpointConfig,
    CheckpointManager,
    InMemoryCheckpointStore,
    FileCheckpointStore,
)
from fastagentic.sdk import (
    FastAgenticClient,
    AsyncFastAgenticClient,
    ClientConfig,
    RunRequest,
    RunResponse,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    FastAgenticError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    TimeoutError as SDKTimeoutError,
    ServerError,
)
from fastagentic.compliance import (
    PIIDetector,
    PIIType,
    PIIMatch,
    PIIMasker,
    PIIConfig,
    PIIDetectionHook,
    PIIMaskingHook,
)
from fastagentic.dashboard import (
    StatsCollector,
    RunStats,
    EndpointStats,
    SystemStats,
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    PrometheusExporter,
    DashboardAPI,
    DashboardConfig,
)
from fastagentic.ops import (
    ReadinessChecker,
    ReadinessCheck,
    CheckResult,
    CheckStatus,
    ReadinessReport,
)

__version__ = "1.2.0"

__all__ = [
    # Core
    "App",
    # Decorators
    "tool",
    "resource",
    "prompt",
    "agent_endpoint",
    # Context
    "AgentContext",
    "RunContext",
    # Reliability
    "RetryPolicy",
    "Timeout",
    "CircuitBreaker",
    "FallbackChain",
    "RateLimit",
    # Policy
    "Policy",
    "PolicyContext",
    "PolicyResult",
    "PolicyAction",
    "PolicyEngine",
    "Role",
    "Permission",
    "RBACPolicy",
    "ScopePolicy",
    "Scope",
    "BudgetPolicy",
    "Budget",
    "BudgetPeriod",
    # Cost
    "CostTracker",
    "CostRecord",
    "ModelPricing",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    # Prompts
    "PromptTemplate",
    "PromptVariable",
    "PromptRegistry",
    "PromptVersion",
    "ABTest",
    "PromptVariant",
    # HITL
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalPolicy",
    "ApprovalStatus",
    "EscalationManager",
    "EscalationTrigger",
    "EscalationLevel",
    "require_confirmation",
    # Cluster
    "Worker",
    "WorkerStatus",
    "WorkerConfig",
    "WorkerPool",
    "Task",
    "TaskStatus",
    "TaskResult",
    "TaskQueue",
    "Coordinator",
    "CoordinatorConfig",
    # Checkpoint
    "Checkpoint",
    "CheckpointMetadata",
    "CheckpointStore",
    "CheckpointConfig",
    "CheckpointManager",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    # SDK
    "FastAgenticClient",
    "AsyncFastAgenticClient",
    "ClientConfig",
    "RunRequest",
    "RunResponse",
    "RunStatus",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "ToolResult",
    "FastAgenticError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "SDKTimeoutError",
    "ServerError",
    # Compliance
    "PIIDetector",
    "PIIType",
    "PIIMatch",
    "PIIMasker",
    "PIIConfig",
    "PIIDetectionHook",
    "PIIMaskingHook",
    # Dashboard
    "StatsCollector",
    "RunStats",
    "EndpointStats",
    "SystemStats",
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "PrometheusExporter",
    "DashboardAPI",
    "DashboardConfig",
    # Ops
    "ReadinessChecker",
    "ReadinessCheck",
    "CheckResult",
    "CheckStatus",
    "ReadinessReport",
    # Version
    "__version__",
]
