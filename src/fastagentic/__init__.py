"""FastAgentic - The deployment layer for agentic applications.

Build agents with anything. Ship them with FastAgentic.
"""

from fastagentic.app import App
from fastagentic.audit import AuditEvent, AuditEventType, AuditLogger, AuditSeverity
from fastagentic.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointManager,
    CheckpointMetadata,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from fastagentic.cluster import (
    Coordinator,
    CoordinatorConfig,
    Task,
    TaskQueue,
    TaskResult,
    TaskStatus,
    Worker,
    WorkerConfig,
    WorkerPool,
    WorkerStatus,
)
from fastagentic.compliance import (
    PIIConfig,
    PIIDetectionHook,
    PIIDetector,
    PIIMasker,
    PIIMaskingHook,
    PIIMatch,
    PIIType,
)
from fastagentic.context import AgentContext, RunContext
from fastagentic.cost import CostRecord, CostTracker, ModelPricing
from fastagentic.dashboard import (
    Counter,
    DashboardAPI,
    DashboardConfig,
    EndpointStats,
    Gauge,
    Histogram,
    MetricsRegistry,
    PrometheusExporter,
    RunStats,
    StatsCollector,
    SystemStats,
)
from fastagentic.decorators import agent_endpoint, prompt, resource, tool
from fastagentic.hitl import (
    ApprovalManager,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalStatus,
    EscalationLevel,
    EscalationManager,
    EscalationTrigger,
    require_confirmation,
)
from fastagentic.ops import (
    CheckResult,
    CheckStatus,
    ReadinessCheck,
    ReadinessChecker,
    ReadinessReport,
)
from fastagentic.policy import (
    Budget,
    BudgetPeriod,
    BudgetPolicy,
    Permission,
    Policy,
    PolicyAction,
    PolicyContext,
    PolicyEngine,
    PolicyResult,
    RBACPolicy,
    Role,
    Scope,
    ScopePolicy,
)
from fastagentic.prompts import (
    ABTest,
    PromptRegistry,
    PromptTemplate,
    PromptVariable,
    PromptVariant,
    PromptVersion,
)
from fastagentic.reliability import CircuitBreaker, FallbackChain, RateLimit, RetryPolicy, Timeout
from fastagentic.sdk import (
    AsyncFastAgenticClient,
    AuthenticationError,
    ClientConfig,
    FastAgenticClient,
    FastAgenticError,
    RateLimitError,
    RunRequest,
    RunResponse,
    RunStatus,
    ServerError,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    ValidationError,
)
from fastagentic.sdk import TimeoutError as SDKTimeoutError

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
