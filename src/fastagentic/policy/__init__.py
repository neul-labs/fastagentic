"""Policy engine for FastAgentic.

Provides role-based access control (RBAC), scope-based permissions,
budget enforcement, and quota management.
"""

from fastagentic.policy.base import Policy, PolicyContext, PolicyResult, PolicyAction
from fastagentic.policy.rbac import Role, Permission, RBACPolicy
from fastagentic.policy.scopes import ScopePolicy, Scope
from fastagentic.policy.budget import BudgetPolicy, Budget, BudgetPeriod
from fastagentic.policy.engine import PolicyEngine

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
