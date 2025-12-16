"""Conftest for mutation testing.

These tests are designed to catch specific mutations in critical code paths.
Run with: mutmut run --paths-to-mutate=src/fastagentic/policy
"""

import pytest

from fastagentic.context import UserInfo
from fastagentic.policy.base import PolicyContext


@pytest.fixture
def policy_context_factory():
    """Factory for creating PolicyContext objects."""

    def create(
        user_id: str | None = "test-user",
        tenant_id: str | None = "test-tenant",
        endpoint: str | None = "/test",
        resource: str | None = "tools/search",
        action: str | None = "invoke",
        roles: list[str] | None = None,
        required_scopes: list[str] | None = None,
        user_scopes: list[str] | None = None,
        estimated_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> PolicyContext:
        # Create UserInfo if user_id is provided
        user = UserInfo(id=user_id) if user_id else None

        return PolicyContext(
            user=user,
            tenant_id=tenant_id,
            endpoint=endpoint,
            resource=resource,
            action=action,
            required_scopes=required_scopes or [],
            user_scopes=user_scopes or [],
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
        )

    return create
