"""Policy engine for FastAgentic.

Provides role-based access control (RBAC), scope-based permissions,
budget enforcement, and quota management.
"""

from fastagentic.policy.base import Policy, PolicyAction, PolicyContext, PolicyResult
from fastagentic.policy.budget import Budget, BudgetPeriod, BudgetPolicy
from fastagentic.policy.engine import PolicyEngine
from fastagentic.policy.rbac import Permission, RBACPolicy, Role
from fastagentic.policy.scopes import Scope, ScopePolicy

__all__ = [
    # Base
    "Policy",
    "PolicyContext",
    "PolicyResult",
    "PolicyAction",
    # RBAC
    "Role",
    "Permission",
    "RBACPolicy",
    # Scopes
    "ScopePolicy",
    "Scope",
    # Budget
    "BudgetPolicy",
    "Budget",
    "BudgetPeriod",
    # Engine
    "PolicyEngine",
]
