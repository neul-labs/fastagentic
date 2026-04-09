"""Compliance hooks for FastAgentic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastagentic.compliance.pii import PIIConfig, PIIDetector, PIIMasker, PIIMatch, PIIType


@dataclass
class PIIDetectionResult:
    """Result of PII detection.

    Attributes:
        has_pii: Whether PII was detected
        matches: List of PII matches
        types: Set of PII types found
        should_block: Whether request should be blocked
        message: Human-readable message
    """

    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    types: set[PIIType] = field(default_factory=set)
    should_block: bool = False
    message: str = ""


class PIIDetectionHook:
    """Hook for detecting PII in requests/responses.

    Example:
        hook = PIIDetectionHook(
            block_on_detect=True,
            blocked_types={PIIType.SSN, PIIType.CREDIT_CARD},
        )

        # Check request
        result = hook.check_request(request_data)
        if result.should_block:
            raise ValueError(result.message)

        # Check response
        result = hook.check_response(response_data)
        if result.has_pii:
            log_warning(f"PII in response: {result.types}")
    """

    def __init__(
        self,
        detector: PIIDetector | None = None,
        config: PIIConfig | None = None,
        block_on_detect: bool = False,
        blocked_types: set[PIIType] | None = None,
        warn_types: set[PIIType] | None = None,
        check_keys: bool = True,
        check_values: bool = True,
    ) -> None:
        """Initialize PII detection hook.

        Args:
            detector: PII detector to use
            config: Detection configuration
            block_on_detect: Block request if PII detected
            blocked_types: PII types that should block
            warn_types: PII types that should warn
            check_keys: Check dictionary keys
            check_values: Check dictionary values
        """
        self._detector = detector or PIIDetector(config=config)
        self.block_on_detect = block_on_detect
        self.blocked_types = blocked_types or {
            PIIType.SSN,
            PIIType.CREDIT_CARD,
            PIIType.PASSWORD,
            PIIType.API_KEY,
            PIIType.ACCESS_TOKEN,
        }
        self.warn_types = warn_types or set(PIIType)
        self.check_keys = check_keys
        self.check_values = check_values

    def check_text(self, text: str) -> PIIDetectionResult:
        """Check text for PII.

        Args:
            text: Text to check

        Returns:
            Detection result
        """
        matches = self._detector.detect(text)
        types = {m.type for m in matches}

        has_pii = len(matches) > 0
        blocked = types & self.blocked_types
        should_block = self.block_on_detect and len(blocked) > 0

        message = ""
        if should_block:
            message = f"Blocked PII types detected: {', '.join(t.value for t in blocked)}"
        elif has_pii:
            message = f"PII detected: {', '.join(t.value for t in types)}"

        return PIIDetectionResult(
            has_pii=has_pii,
            matches=matches,
            types=types,
            should_block=should_block,
            message=message,
        )

    def check_dict(self, data: dict[str, Any]) -> PIIDetectionResult:
        """Check dictionary for PII.

        Args:
            data: Dictionary to check

        Returns:
            Detection result
        """
        all_matches: list[PIIMatch] = []
        all_types: set[PIIType] = set()

        def check_value(value: Any) -> None:
            if isinstance(value, str):
                matches = self._detector.detect(value)
                all_matches.extend(matches)
                all_types.update(m.type for m in matches)
            elif isinstance(value, dict):
                check_dict_recursive(value)
            elif isinstance(value, list):
                for item in value:
                    check_value(item)

        def check_dict_recursive(d: dict[str, Any]) -> None:
            for key, value in d.items():
                if self.check_keys and isinstance(key, str):
                    matches = self._detector.detect(key)
                    all_matches.extend(matches)
                    all_types.update(m.type for m in matches)

                if self.check_values:
                    check_value(value)

        check_dict_recursive(data)

        has_pii = len(all_matches) > 0
        blocked = all_types & self.blocked_types
        should_block = self.block_on_detect and len(blocked) > 0

        message = ""
        if should_block:
            message = f"Blocked PII types detected: {', '.join(t.value for t in blocked)}"
        elif has_pii:
            message = f"PII detected: {', '.join(t.value for t in all_types)}"

        return PIIDetectionResult(
            has_pii=has_pii,
            matches=all_matches,
            types=all_types,
            should_block=should_block,
            message=message,
        )

    def check_request(self, request_data: dict[str, Any]) -> PIIDetectionResult:
        """Check request data for PII.

        Args:
            request_data: Request data

        Returns:
            Detection result
        """
        return self.check_dict(request_data)

    def check_response(self, response_data: Any) -> PIIDetectionResult:
        """Check response data for PII.

        Args:
            response_data: Response data

        Returns:
            Detection result
        """
        if isinstance(response_data, dict):
            return self.check_dict(response_data)
        elif isinstance(response_data, str):
            return self.check_text(response_data)
        elif isinstance(response_data, list):
            all_matches: list[PIIMatch] = []
            all_types: set[PIIType] = set()

            for item in response_data:
                if isinstance(item, dict):
                    result = self.check_dict(item)
                elif isinstance(item, str):
                    result = self.check_text(item)
                else:
                    continue

                all_matches.extend(result.matches)
                all_types.update(result.types)

            has_pii = len(all_matches) > 0
            blocked = all_types & self.blocked_types
            should_block = self.block_on_detect and len(blocked) > 0

            return PIIDetectionResult(
                has_pii=has_pii,
                matches=all_matches,
                types=all_types,
                should_block=should_block,
            )

        return PIIDetectionResult(has_pii=False)


class PIIMaskingHook:
    """Hook for masking PII in requests/responses.

    Example:
        hook = PIIMaskingHook(
            mask_requests=True,
            mask_responses=True,
        )

        # Mask request
        masked_request = hook.mask_request(request_data)

        # Mask response
        masked_response = hook.mask_response(response_data)
    """

    def __init__(
        self,
        masker: PIIMasker | None = None,
        config: PIIConfig | None = None,
        mask_requests: bool = True,
        mask_responses: bool = True,
        mask_logs: bool = True,
        mask_char: str = "*",
        show_type: bool = False,
        types_to_mask: set[PIIType] | None = None,
    ) -> None:
        """Initialize PII masking hook.

        Args:
            masker: PII masker to use
            config: Detection configuration
            mask_requests: Whether to mask requests
            mask_responses: Whether to mask responses
            mask_logs: Whether to mask log data
            mask_char: Character for masking
            show_type: Show PII type in masked output
            types_to_mask: Only mask these types
        """
        self._masker = masker or PIIMasker(config=config)
        self.mask_requests = mask_requests
        self.mask_responses = mask_responses
        self.mask_logs = mask_logs
        self.mask_char = mask_char
        self.show_type = show_type
        self.types_to_mask = types_to_mask

    def mask_text(self, text: str) -> str:
        """Mask PII in text.

        Args:
            text: Text to mask

        Returns:
            Masked text
        """
        return self._masker.mask(
            text,
            mask_char=self.mask_char,
            show_type=self.show_type,
            types=self.types_to_mask,
        )

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask PII in dictionary.

        Args:
            data: Dictionary to mask

        Returns:
            Masked dictionary
        """
        return self._masker.get_masked_dict(
            data,
            mask_char=self.mask_char,
            deep=True,
        )

    def mask_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Mask PII in request data.

        Args:
            request_data: Request data

        Returns:
            Masked request data
        """
        if not self.mask_requests:
            return request_data
        return self.mask_dict(request_data)

    def mask_response(self, response_data: Any) -> Any:
        """Mask PII in response data.

        Args:
            response_data: Response data

        Returns:
            Masked response data
        """
        if not self.mask_responses:
            return response_data

        if isinstance(response_data, dict):
            return self.mask_dict(response_data)
        elif isinstance(response_data, str):
            return self.mask_text(response_data)
        elif isinstance(response_data, list):
            return [self.mask_response(item) for item in response_data]

        return response_data

    def mask_log_data(self, log_data: dict[str, Any]) -> dict[str, Any]:
        """Mask PII in log data.

        Args:
            log_data: Log data

        Returns:
            Masked log data
        """
        if not self.mask_logs:
            return log_data
        return self.mask_dict(log_data)
