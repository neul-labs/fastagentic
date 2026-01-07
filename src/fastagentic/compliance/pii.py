"""PII detection and masking for FastAgentic."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    # Personal identifiers
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # Social Security Number
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"

    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"

    # Healthcare
    MEDICAL_RECORD = "medical_record"
    HEALTH_INSURANCE = "health_insurance"

    # Location
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    IP_ADDRESS = "ip_address"

    # Authentication
    PASSWORD = "password"
    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"

    # Personal data
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"

    # Custom
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII match.

    Attributes:
        type: Type of PII detected
        value: The matched value
        start: Start position in text
        end: End position in text
        confidence: Detection confidence (0-1)
        context: Surrounding context
    """

    type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""

    @property
    def masked_value(self) -> str:
        """Get a masked version of the value."""
        if len(self.value) <= 4:
            return "*" * len(self.value)
        return self.value[:2] + "*" * (len(self.value) - 4) + self.value[-2:]


@dataclass
class PIIPattern:
    """A pattern for detecting PII.

    Attributes:
        type: PII type this pattern detects
        pattern: Regular expression pattern
        confidence: Base confidence for matches
        validator: Optional validation function
    """

    type: PIIType
    pattern: str
    confidence: float = 1.0
    validator: Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def find_matches(self, text: str) -> list[PIIMatch]:
        """Find all matches in text."""
        matches = []
        for match in self._compiled.finditer(text):
            value = match.group()

            # Apply validator if present
            confidence = self.confidence
            if self.validator and not self.validator(value):
                confidence *= 0.5

            # Get context
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]

            matches.append(
                PIIMatch(
                    type=self.type,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    context=context,
                )
            )

        return matches


# Default PII patterns
DEFAULT_PATTERNS: list[PIIPattern] = [
    # Email
    PIIPattern(
        type=PIIType.EMAIL,
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        confidence=0.95,
    ),
    # Phone (various formats)
    PIIPattern(
        type=PIIType.PHONE,
        pattern=r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        confidence=0.85,
    ),
    # SSN
    PIIPattern(
        type=PIIType.SSN,
        pattern=r"\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b",
        confidence=0.9,
        validator=lambda x: len(re.sub(r"[-\s]", "", x)) == 9,
    ),
    # Credit Card (Luhn validation would improve this)
    PIIPattern(
        type=PIIType.CREDIT_CARD,
        pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        confidence=0.9,
    ),
    # Credit Card with spaces/dashes
    PIIPattern(
        type=PIIType.CREDIT_CARD,
        pattern=r"\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b",
        confidence=0.8,
    ),
    # IP Address
    PIIPattern(
        type=PIIType.IP_ADDRESS,
        pattern=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        confidence=0.95,
    ),
    # IPv6 (simplified)
    PIIPattern(
        type=PIIType.IP_ADDRESS,
        pattern=r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        confidence=0.95,
    ),
    # Date of Birth (various formats)
    PIIPattern(
        type=PIIType.DATE_OF_BIRTH,
        pattern=r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12][0-9]|3[01])[/\-](?:19|20)[0-9]{2}\b",
        confidence=0.7,
    ),
    # ZIP Code (US)
    PIIPattern(
        type=PIIType.ZIP_CODE,
        pattern=r"\b[0-9]{5}(?:-[0-9]{4})?\b",
        confidence=0.6,  # Lower confidence as many numbers match
    ),
    # API Key patterns
    PIIPattern(
        type=PIIType.API_KEY,
        pattern=r"\b(?:sk|pk|api|key)[-_]?[a-zA-Z0-9]{20,}\b",
        confidence=0.85,
    ),
    # Bearer tokens
    PIIPattern(
        type=PIIType.ACCESS_TOKEN,
        pattern=r"\bBearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b",
        confidence=0.95,
    ),
    # Password in common formats
    PIIPattern(
        type=PIIType.PASSWORD,
        pattern=r'(?:password|passwd|pwd)\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
        confidence=0.9,
    ),
    # IBAN
    PIIPattern(
        type=PIIType.IBAN,
        pattern=r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}(?:[A-Z0-9]?){0,16}\b",
        confidence=0.9,
    ),
]


@dataclass
class PIIConfig:
    """Configuration for PII detection.

    Attributes:
        enabled_types: Which PII types to detect (None = all)
        min_confidence: Minimum confidence threshold
        custom_patterns: Additional custom patterns
        blocklist: Exact strings to always flag
        allowlist: Exact strings to never flag
    """

    enabled_types: set[PIIType] | None = None
    min_confidence: float = 0.5
    custom_patterns: list[PIIPattern] = field(default_factory=list)
    blocklist: set[str] = field(default_factory=set)
    allowlist: set[str] = field(default_factory=set)


class PIIDetector:
    """Detect PII in text.

    Example:
        detector = PIIDetector()

        # Detect all PII
        matches = detector.detect("Contact me at john@example.com or 555-123-4567")
        for match in matches:
            print(f"{match.type}: {match.value}")

        # Check if text contains PII
        if detector.contains_pii(text):
            print("Warning: PII detected")

        # Configure detection
        detector = PIIDetector(config=PIIConfig(
            enabled_types={PIIType.EMAIL, PIIType.PHONE},
            min_confidence=0.8,
        ))
    """

    def __init__(
        self,
        config: PIIConfig | None = None,
        patterns: list[PIIPattern] | None = None,
    ) -> None:
        """Initialize PII detector.

        Args:
            config: Detection configuration
            patterns: Custom patterns (uses defaults if None)
        """
        self.config = config or PIIConfig()
        self._patterns = patterns or DEFAULT_PATTERNS.copy()

        # Add custom patterns from config
        self._patterns.extend(self.config.custom_patterns)

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            List of PII matches
        """
        matches: list[PIIMatch] = []

        # Check blocklist first
        for blocked in self.config.blocklist:
            start = 0
            while True:
                pos = text.find(blocked, start)
                if pos == -1:
                    break
                matches.append(
                    PIIMatch(
                        type=PIIType.CUSTOM,
                        value=blocked,
                        start=pos,
                        end=pos + len(blocked),
                        confidence=1.0,
                    )
                )
                start = pos + 1

        # Run pattern matching
        for pattern in self._patterns:
            # Skip disabled types
            if (
                self.config.enabled_types is not None
                and pattern.type not in self.config.enabled_types
            ):
                continue

            for match in pattern.find_matches(text):
                # Skip allowlisted values
                if match.value in self.config.allowlist:
                    continue

                # Skip low confidence matches
                if match.confidence < self.config.min_confidence:
                    continue

                matches.append(match)

        # Sort by position and deduplicate overlapping
        matches.sort(key=lambda m: (m.start, -m.confidence))
        return self._deduplicate_overlapping(matches)

    def _deduplicate_overlapping(
        self,
        matches: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []

        result = [matches[0]]
        for match in matches[1:]:
            last = result[-1]
            # Check for overlap
            if match.start < last.end:
                # Keep the one with higher confidence
                if match.confidence > last.confidence:
                    result[-1] = match
            else:
                result.append(match)

        return result

    def contains_pii(self, text: str) -> bool:
        """Check if text contains any PII.

        Args:
            text: Text to check

        Returns:
            True if PII is detected
        """
        return len(self.detect(text)) > 0

    def get_pii_types(self, text: str) -> set[PIIType]:
        """Get the types of PII found in text.

        Args:
            text: Text to analyze

        Returns:
            Set of PII types found
        """
        return {match.type for match in self.detect(text)}

    def add_pattern(self, pattern: PIIPattern) -> None:
        """Add a custom pattern.

        Args:
            pattern: Pattern to add
        """
        self._patterns.append(pattern)


class PIIMasker:
    """Mask PII in text.

    Example:
        masker = PIIMasker()

        # Mask all PII
        masked = masker.mask("Email: john@example.com")
        # "Email: jo**********om"

        # Custom masking
        masked = masker.mask(text, mask_char="X", show_type=True)
        # "Email: [EMAIL:XXXX]"

        # Mask specific types
        masked = masker.mask(text, types={PIIType.EMAIL})
    """

    def __init__(
        self,
        detector: PIIDetector | None = None,
        config: PIIConfig | None = None,
    ) -> None:
        """Initialize PII masker.

        Args:
            detector: PII detector to use
            config: Configuration for detection
        """
        self._detector = detector or PIIDetector(config=config)

    def mask(
        self,
        text: str,
        *,
        mask_char: str = "*",
        show_type: bool = False,
        types: set[PIIType] | None = None,
        replacement: str | None = None,
    ) -> str:
        """Mask PII in text.

        Args:
            text: Text to mask
            mask_char: Character to use for masking
            show_type: Include PII type in replacement
            types: Only mask these types (None = all)
            replacement: Fixed replacement string

        Returns:
            Masked text
        """
        matches = self._detector.detect(text)

        # Filter by type if specified
        if types:
            matches = [m for m in matches if m.type in types]

        # Sort by position descending to replace from end
        matches.sort(key=lambda m: m.start, reverse=True)

        result = text
        for match in matches:
            if replacement:
                masked = replacement
            elif show_type:
                masked = f"[{match.type.value.upper()}:{mask_char * 4}]"
            else:
                # Partial masking - show first and last 2 chars
                val = match.value
                if len(val) <= 4:
                    masked = mask_char * len(val)
                else:
                    masked = val[:2] + mask_char * (len(val) - 4) + val[-2:]

            result = result[: match.start] + masked + result[match.end :]

        return result

    def redact(
        self,
        text: str,
        types: set[PIIType] | None = None,
    ) -> str:
        """Fully redact PII from text.

        Args:
            text: Text to redact
            types: Only redact these types (None = all)

        Returns:
            Redacted text
        """
        return self.mask(text, replacement="[REDACTED]", types=types)

    def get_masked_dict(
        self,
        data: dict[str, Any],
        *,
        mask_char: str = "*",
        deep: bool = True,
    ) -> dict[str, Any]:
        """Mask PII in a dictionary.

        Args:
            data: Dictionary to mask
            mask_char: Character for masking
            deep: Recursively mask nested dicts

        Returns:
            Masked dictionary (copy)
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.mask(value, mask_char=mask_char)
            elif isinstance(value, dict) and deep:
                result[key] = self.get_masked_dict(value, mask_char=mask_char, deep=True)
            elif isinstance(value, list) and deep:
                result[key] = [
                    self.get_masked_dict(v, mask_char=mask_char, deep=True)
                    if isinstance(v, dict)
                    else self.mask(v, mask_char=mask_char)
                    if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result
