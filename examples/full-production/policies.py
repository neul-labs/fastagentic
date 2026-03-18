"""Policy definitions for RBAC and budgets."""

from fastagentic.policy import Role, Permission, BudgetPolicy, Budget, BudgetPeriod

# Define roles
admin_role = Role(
    name="admin",
    permissions=[
        Permission(resource="*", actions=["*"]),
    ],
)

user_role = Role(
    name="user",
    permissions=[
        Permission(resource="chat", actions=["read", "write"]),
        Permission(resource="kb", actions=["read"]),
    ],
)

# Budget policy
budget_policy = BudgetPolicy(
    budgets=[
        Budget(
            name="daily_limit",
            amount=100.0,
            period=BudgetPeriod.DAILY,
            scope="user",
        ),
    ],
)
