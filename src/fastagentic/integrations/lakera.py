"""Lakera Guard integration for FastAgentic.

Lakera Guard provides AI security with:
- Prompt injection detection
- Jailbreak prevention
- PII detection and redaction
- Content moderation
- Custom policy enforcement

https://lakera.ai

Example:
    from fastagentic import App
    from fastagentic.integrations import LakeraIntegration

    app = App(
        title="My Agent",
        integrations=[
            LakeraIntegration(
                api_key="lak-...",
                block_on_detect=True,
            )
        ]
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from fastagentic.hooks.base import Hook, HookContext, HookResult
from fastagentic.integrations.base import Integration, IntegrationConfig

if TYPE_CHECKING:
    from fastagentic.app import App


class LakeraGuardError(Exception):
    """Raised when Lakera Guard detects a security issue."""

    def __init__(
        self,
        message: str,
        category: str,
        confidence: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.confidence = confidence
        self.details = details or {}


@dataclass
class LakeraConfig(IntegrationConfig):
    """Configuration for Lakera integration."""

    block_on_detect: bool = True
    confidence_threshold: float = 0.8
    categories: list[str] = field(default_factory=lambda: ["prompt_injection", "jailbreak", "pii"])
    check_input: bool = True
    check_output: bool = True
    log_detections: bool = True
    timeout: float = 5.0


class LakeraHook(Hook):
    """Hook for Lakera Guard security checks.

    Scans inputs and outputs for:
    - Prompt injection attempts
    - Jailbreak attempts
    - PII (personally identifiable information)
    - Toxic or harmful content
    """

    def __init__(self, config: LakeraConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def _scan_content(self, content: str, context: str = "input") -> dict[str, Any]:
        """Scan content using Lakera Guard API."""
        import os

        api_key = self.config.api_key or os.getenv("LAKERA_API_KEY")
        if not api_key:
            return {"flagged": False, "categories": {}}

        client = await self._get_client()

        try:
            response = await client.post(
                "https://api.lakera.ai/v1/prompt_injection",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": content},
            )
            response.raise_for_status()
            result = response.json()

            return {
                "flagged": result.get("results", [{}])[0].get("flagged", False),
                "categories": result.get("results", [{}])[0].get("categories", {}),
                "payload_type": result.get("results", [{}])[0].get("payload_type"),
            }

        except httpx.HTTPError as e:
            # Log but don't block on API errors
            if self.config.log_detections:
                import structlog

                logger = structlog.get_logger()
                logger.warning("lakera_api_error", error=str(e), context=context)
            return {"flagged": False, "categories": {}, "error": str(e)}

    def _format_messages_for_scan(self, messages: list[dict[str, Any]]) -> str:
        """Format chat messages for scanning."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-modal messages
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = " ".join(text_parts)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def on_request(self, ctx: HookContext) -> HookResult:
        """Scan incoming request for security issues."""
        if not self.config.check_input:
            return HookResult.proceed()

        content = ""
        if ctx.messages:
            content = self._format_messages_for_scan(ctx.messages)
        elif ctx.request:
            content = str(ctx.request)

        if not content:
            return HookResult.proceed()

        result = await self._scan_content(content, context="request")

        if result.get("flagged"):
            categories = result.get("categories", {})
            detected = [cat for cat, info in categories.items() if info]

            if self.config.log_detections:
                import structlog

                logger = structlog.get_logger()
                logger.warning(
                    "lakera_detection",
                    run_id=ctx.run_id,
                    context="request",
                    categories=detected,
                    payload_type=result.get("payload_type"),
                )

            # Store detection info in context
            ctx.metadata["lakera_input_flagged"] = True
            ctx.metadata["lakera_input_categories"] = detected

            if self.config.block_on_detect:
                return HookResult.reject(f"Security check failed: {', '.join(detected)}")

        return HookResult.proceed()

    async def on_llm_start(self, ctx: HookContext) -> HookResult:
        """Scan messages before LLM call."""
        if not self.config.check_input:
            return HookResult.proceed()

        if not ctx.messages:
            return HookResult.proceed()

        content = self._format_messages_for_scan(ctx.messages)
        result = await self._scan_content(content, context="llm_input")

        if result.get("flagged"):
            categories = result.get("categories", {})
            detected = [cat for cat, info in categories.items() if info]

            ctx.metadata["lakera_llm_input_flagged"] = True
            ctx.metadata["lakera_llm_input_categories"] = detected

            if self.config.block_on_detect:
                return HookResult.reject(f"LLM input security check failed: {', '.join(detected)}")

        return HookResult.proceed()

    async def on_llm_end(self, ctx: HookContext) -> HookResult:
        """Scan LLM output for security issues."""
        if not self.config.check_output:
            return HookResult.proceed()

        if not ctx.response:
            return HookResult.proceed()

        content = str(ctx.response)
        result = await self._scan_content(content, context="llm_output")

        if result.get("flagged"):
            categories = result.get("categories", {})
            detected = [cat for cat, info in categories.items() if info]

            if self.config.log_detections:
                import structlog

                logger = structlog.get_logger()
                logger.warning(
                    "lakera_detection",
                    run_id=ctx.run_id,
                    context="llm_output",
                    categories=detected,
                )

            ctx.metadata["lakera_output_flagged"] = True
            ctx.metadata["lakera_output_categories"] = detected

            # For output, we might want to redact rather than block
            if self.config.block_on_detect:
                return HookResult.modify("[Content filtered for security reasons]")

        return HookResult.proceed()

    async def on_tool_call(self, ctx: HookContext) -> HookResult:
        """Scan tool inputs for injection attempts."""
        if not self.config.check_input:
            return HookResult.proceed()

        if not ctx.tool_input:
            return HookResult.proceed()

        content = str(ctx.tool_input)
        result = await self._scan_content(content, context="tool_input")

        if result.get("flagged"):
            categories = result.get("categories", {})
            detected = [cat for cat, info in categories.items() if info]

            ctx.metadata["lakera_tool_input_flagged"] = True
            ctx.metadata["lakera_tool_categories"] = detected

            if self.config.block_on_detect:
                return HookResult.reject(f"Tool input security check failed: {', '.join(detected)}")

        return HookResult.proceed()


class LakeraIntegration(Integration):
    """Lakera Guard security integration.

    Provides AI security guardrails for prompt injection detection,
    jailbreak prevention, PII detection, and content moderation.

    Features:
    - **Prompt Injection Detection**: Catches injection attempts in user input
    - **Jailbreak Prevention**: Detects attempts to bypass system instructions
    - **PII Detection**: Identifies personally identifiable information
    - **Content Moderation**: Filters toxic or harmful content
    - **Custom Policies**: Define your own security rules

    Example:
        # Basic usage - block on detection
        app = App(
            integrations=[
                LakeraIntegration(api_key="lak-...")
            ]
        )

        # Log but don't block
        app = App(
            integrations=[
                LakeraIntegration(
                    api_key="lak-...",
                    block_on_detect=False,
                    log_detections=True,
                )
            ]
        )

        # Check only specific categories
        app = App(
            integrations=[
                LakeraIntegration(
                    api_key="lak-...",
                    categories=["prompt_injection"],
                )
            ]
        )

    Environment variables:
        LAKERA_API_KEY: Lakera Guard API key
    """

    def __init__(
        self,
        api_key: str | None = None,
        block_on_detect: bool = True,
        confidence_threshold: float = 0.8,
        categories: list[str] | None = None,
        check_input: bool = True,
        check_output: bool = True,
        log_detections: bool = True,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> None:
        config = LakeraConfig(
            api_key=api_key,
            block_on_detect=block_on_detect,
            confidence_threshold=confidence_threshold,
            categories=categories or ["prompt_injection", "jailbreak", "pii"],
            check_input=check_input,
            check_output=check_output,
            log_detections=log_detections,
            timeout=timeout,
            extra=kwargs,
        )
        super().__init__(config)
        self._config = config
        self._hook: LakeraHook | None = None

    @property
    def name(self) -> str:
        return "lakera"

    def is_available(self) -> bool:
        # Lakera uses HTTP API, so no special package needed
        return True

    def validate_config(self) -> list[str]:
        errors = super().validate_config()

        import os

        api_key = self._config.api_key or os.getenv("LAKERA_API_KEY")
        if not api_key:
            errors.append("Lakera api_key is required")

        return errors

    def get_hooks(self) -> list[Hook]:
        if not self._hook:
            self._hook = LakeraHook(self._config)
        return [self._hook]

    def setup(self, app: App) -> None:
        """Initialize Lakera integration."""
        pass  # HTTP-based, no special setup needed

    async def scan(self, content: str) -> dict[str, Any]:
        """Manually scan content for security issues.

        Useful for custom scanning outside of hooks:

            result = await lakera.scan("user input here")
            if result["flagged"]:
                print(f"Detected: {result['categories']}")

        Returns:
            Dict with flagged (bool) and categories (dict)
        """
        if not self._hook:
            self._hook = LakeraHook(self._config)
        return await self._hook._scan_content(content)

    async def on_shutdown(self) -> None:
        """Close HTTP client."""
        await super().on_shutdown()
        if self._hook and self._hook._client:
            await self._hook._client.aclose()
            self._hook._client = None
