"""Policy engine for combining and evaluating multiple policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fastagentic.policy.base import Policy, PolicyAction, PolicyContext, PolicyResult

logger = logging.getLogger(__name__)


@dataclass
class PolicyEngineConfig:
    """Configuration for the policy engine.

    Attributes:
        fail_open: If True, allow on policy errors; if False, deny on errors
        log_decisions: Whether to log all policy decisions
        short_circuit_deny: Stop evaluating after first deny
        short_circuit_allow: Stop evaluating after first explicit allow
    """

    fail_open: bool = False
    log_decisions: bool = True
    short_circuit_deny: bool = True
    short_circuit_allow: bool = False


@dataclass
class PolicyDecision:
    """Result of policy engine evaluation.

    Attributes:
        action: Final action (allow/deny/warn/limit)
        reason: Combined reason from all policies
        policy_results: Individual results from each policy
        warnings: Any warnings from policies
        limits: Combined limits from all policies
    """

    action: PolicyAction
    reason: str | None = None
    policy_results: dict[str, PolicyResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        """Check if the decision allows the request."""
        return self.action in (PolicyAction.ALLOW, PolicyAction.WARN, PolicyAction.LIMIT)

    @property
    def is_denied(self) -> bool:
        """Check if the decision denies the request."""
        return self.action == PolicyAction.DENY


class PolicyEngine:
    """Engine for combining and evaluating multiple policies.

    Policies are evaluated in priority order (higher priority first).
    The engine combines results to make a final allow/deny decision.

    Example:
        engine = PolicyEngine()

        # Add policies (evaluated in priority order)
        engine.add_policy(rbac_policy)      # priority=100
        engine.add_policy(scope_policy)     # priority=90
        engine.add_policy(budget_policy)    # priority=50

        # Evaluate
        ctx = PolicyContext(
            user=user_info,
            endpoint="/triage",
            action="invoke",
        )
        decision = await engine.evaluate(ctx)

        if decision.is_denied:
            raise HTTPException(403, decision.reason)
    """

    def __init__(
        self,
        config: PolicyEngineConfig | None = None,
    ) -> None:
        """Initialize policy engine.

        Args:
            config: Engine configuration
        """
        self._config = config or PolicyEngineConfig()
        self._policies: list[Policy] = []

    @property
    def policies(self) -> list[Policy]:
        """Get list of registered policies (sorted by priority)."""
        return sorted(self._policies, key=lambda p: p.priority, reverse=True)

    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the engine.

        Args:
            policy: The policy to add
        """
        self._policies.append(policy)

    def remove_policy(self, policy_name: str) -> None:
        """Remove a policy by name.

        Args:
            policy_name: Name of the policy to remove
        """
        self._policies = [p for p in self._policies if p.name != policy_name]

    def get_policy(self, policy_name: str) -> Policy | None:
        """Get a policy by name.

        Args:
            policy_name: Name of the policy

        Returns:
            The policy or None if not found
        """
        for policy in self._policies:
            if policy.name == policy_name:
                return policy
        return None

    async def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        """Evaluate all policies and return combined decision.

        Args:
            ctx: Policy evaluation context

        Returns:
            PolicyDecision with final action and details
        """
        policy_results: dict[str, PolicyResult] = {}
        warnings: list[str] = []
        limits: dict[str, Any] = {}
        deny_reasons: list[str] = []

        # Evaluate policies in priority order
        for policy in self.policies:
            if not policy.enabled:
                continue

            try:
                result = await policy.evaluate(ctx)
                policy_results[policy.name] = result

                if self._config.log_decisions:
                    logger.debug(
                        f"Policy '{policy.name}' decision: {result.action.value}"
                        f"{f' - {result.reason}' if result.reason else ''}"
                    )

                # Handle deny
                if result.is_denied:
                    deny_reasons.append(f"{policy.name}: {result.reason}")
                    if self._config.short_circuit_deny:
                        return PolicyDecision(
                            action=PolicyAction.DENY,
                            reason=deny_reasons[0],
                            policy_results=policy_results,
                        )

                # Collect warnings
                if result.action == PolicyAction.WARN and result.reason:
                    warnings.append(f"{policy.name}: {result.reason}")

                # Collect limits
                if result.action == PolicyAction.LIMIT:
                    limits.update(result.limits)

                # Short-circuit on explicit allow
                if (
                    self._config.short_circuit_allow
                    and result.action == PolicyAction.ALLOW
                    and result.reason  # Explicit allow has a reason
                ):
                    return PolicyDecision(
                        action=PolicyAction.ALLOW,
                        reason=result.reason,
                        policy_results=policy_results,
                    )

            except Exception as e:
                logger.error(f"Policy '{policy.name}' error: {e}")
                if not self._config.fail_open:
                    return PolicyDecision(
                        action=PolicyAction.DENY,
                        reason=f"Policy evaluation error: {policy.name}",
                        policy_results=policy_results,
                    )

        # If any policy denied, return deny
        if deny_reasons:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="; ".join(deny_reasons),
                policy_results=policy_results,
            )

        # If there are limits, return limit action
        if limits:
            return PolicyDecision(
                action=PolicyAction.LIMIT,
                reason="Request allowed with limits",
                policy_results=policy_results,
                warnings=warnings,
                limits=limits,
            )

        # If there are warnings, return warn action
        if warnings:
            return PolicyDecision(
                action=PolicyAction.WARN,
                reason="; ".join(warnings),
                policy_results=policy_results,
                warnings=warnings,
            )

        # All policies allowed
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            policy_results=policy_results,
        )

    async def evaluate_with_context(
        self,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        endpoint: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        required_scopes: list[str] | None = None,
        user_scopes: list[str] | None = None,
        estimated_tokens: int = 0,
        estimated_cost: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Convenience method to evaluate with context parameters.

        Args:
            user_id: User identifier
            tenant_id: Tenant/organization identifier
            endpoint: Endpoint being accessed
            action: Action being performed
            resource: Resource being accessed
            required_scopes: Scopes required for this request
            user_scopes: Scopes the user has
            estimated_tokens: Estimated token usage
            estimated_cost: Estimated cost
            metadata: Additional metadata

        Returns:
            PolicyDecision with final action
        """
        from fastagentic.context import UserInfo

        user = UserInfo(id=user_id) if user_id else None

        ctx = PolicyContext(
            user=user,
            tenant_id=tenant_id,
            endpoint=endpoint,
            action=action,
            resource=resource,
            required_scopes=required_scopes or [],
            user_scopes=user_scopes or [],
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
            metadata=metadata or {},
        )

        return await self.evaluate(ctx)


def create_default_engine(
    *,
    rbac: bool = True,
    scopes: bool = True,
    budget: bool = False,
    fail_open: bool = False,
) -> PolicyEngine:
    """Create a policy engine with default policies.

    Args:
        rbac: Include RBAC policy
        scopes: Include scope-based policy
        budget: Include budget policy
        fail_open: Allow on policy errors

    Returns:
        Configured PolicyEngine
    """
    from fastagentic.policy.budget import BudgetPolicy
    from fastagentic.policy.rbac import RBACPolicy
    from fastagentic.policy.scopes import ScopePolicy

    engine = PolicyEngine(
        config=PolicyEngineConfig(fail_open=fail_open),
    )

    if rbac:
        engine.add_policy(RBACPolicy())

    if scopes:
        engine.add_policy(ScopePolicy())

    if budget:
        engine.add_policy(BudgetPolicy())

    return engine
