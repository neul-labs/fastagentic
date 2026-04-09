"""Tests for compliance module."""

from fastagentic.compliance import (
    PIIConfig,
    PIIDetectionHook,
    PIIDetector,
    PIIMasker,
    PIIMaskingHook,
    PIIMatch,
    PIIType,
)
from fastagentic.compliance.pii import PIIPattern

# ============================================================================
# PIIDetector Tests
# ============================================================================


class TestPIIDetector:
    """Tests for PIIDetector."""

    def test_detect_email(self):
        """Test email detection."""
        detector = PIIDetector()
        matches = detector.detect("Contact me at john@example.com")

        assert len(matches) == 1
        assert matches[0].type == PIIType.EMAIL
        assert matches[0].value == "john@example.com"

    def test_detect_phone(self):
        """Test phone number detection."""
        detector = PIIDetector()
        matches = detector.detect("Call me at 555-123-4567")

        assert len(matches) == 1
        assert matches[0].type == PIIType.PHONE

    def test_detect_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector()
        matches = detector.detect("SSN: 123-45-6789")

        assert len(matches) == 1
        assert matches[0].type == PIIType.SSN

    def test_detect_credit_card(self):
        """Test credit card detection."""
        detector = PIIDetector()
        matches = detector.detect("Card: 4111-1111-1111-1111")

        assert len(matches) == 1
        assert matches[0].type == PIIType.CREDIT_CARD

    def test_detect_ip_address(self):
        """Test IP address detection."""
        detector = PIIDetector()
        matches = detector.detect("Server IP: 192.168.1.100")

        assert len(matches) == 1
        assert matches[0].type == PIIType.IP_ADDRESS

    def test_detect_multiple_pii(self):
        """Test detecting multiple PII types."""
        detector = PIIDetector()
        text = "Email: test@example.com, Phone: 555-123-4567"
        matches = detector.detect(text)

        types = {m.type for m in matches}
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types

    def test_contains_pii(self):
        """Test contains_pii helper."""
        detector = PIIDetector()

        assert detector.contains_pii("Email: test@example.com")
        assert not detector.contains_pii("Hello world")

    def test_get_pii_types(self):
        """Test get_pii_types helper."""
        detector = PIIDetector()
        text = "Email: test@example.com"
        types = detector.get_pii_types(text)

        assert PIIType.EMAIL in types

    def test_enabled_types_filter(self):
        """Test filtering by enabled types."""
        config = PIIConfig(enabled_types={PIIType.EMAIL})
        detector = PIIDetector(config=config)

        text = "Email: test@example.com, Phone: 555-123-4567"
        matches = detector.detect(text)

        # Should only detect email
        assert len(matches) == 1
        assert matches[0].type == PIIType.EMAIL

    def test_min_confidence_filter(self):
        """Test filtering by minimum confidence."""
        config = PIIConfig(min_confidence=0.9)
        detector = PIIDetector(config=config)

        # High confidence matches should still be detected
        matches = detector.detect("test@example.com")
        assert len(matches) == 1

    def test_allowlist(self):
        """Test allowlist."""
        config = PIIConfig(allowlist={"test@example.com"})
        detector = PIIDetector(config=config)

        matches = detector.detect("Email: test@example.com")
        assert len(matches) == 0

    def test_blocklist(self):
        """Test blocklist."""
        config = PIIConfig(blocklist={"secret-value"})
        detector = PIIDetector(config=config)

        matches = detector.detect("The secret-value is here")
        assert len(matches) == 1
        assert matches[0].type == PIIType.CUSTOM

    def test_custom_pattern(self):
        """Test custom pattern."""
        custom = PIIPattern(
            type=PIIType.CUSTOM,
            pattern=r"CUST-[0-9]{6}",
            confidence=1.0,
        )
        config = PIIConfig(custom_patterns=[custom])
        detector = PIIDetector(config=config)

        matches = detector.detect("Customer ID: CUST-123456")
        custom_matches = [m for m in matches if m.type == PIIType.CUSTOM]
        assert len(custom_matches) == 1


class TestPIIMatch:
    """Tests for PIIMatch."""

    def test_masked_value(self):
        """Test masked value generation."""
        match = PIIMatch(
            type=PIIType.EMAIL,
            value="john@example.com",
            start=0,
            end=16,
        )
        masked = match.masked_value
        assert masked.startswith("jo")
        assert masked.endswith("om")
        assert "*" in masked


class TestPIIConfig:
    """Tests for PIIConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PIIConfig()
        assert config.enabled_types is None
        assert config.min_confidence == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = PIIConfig(
            enabled_types={PIIType.EMAIL, PIIType.PHONE},
            min_confidence=0.8,
        )
        assert PIIType.EMAIL in config.enabled_types
        assert config.min_confidence == 0.8


# ============================================================================
# PIIMasker Tests
# ============================================================================


class TestPIIMasker:
    """Tests for PIIMasker."""

    def test_mask_email(self):
        """Test masking email."""
        masker = PIIMasker()
        masked = masker.mask("Email: john@example.com")

        assert "john@example.com" not in masked
        assert "*" in masked

    def test_mask_with_type(self):
        """Test masking with type shown."""
        masker = PIIMasker()
        masked = masker.mask("Email: john@example.com", show_type=True)

        assert "[EMAIL:" in masked

    def test_redact(self):
        """Test redaction."""
        masker = PIIMasker()
        redacted = masker.redact("Email: john@example.com")

        assert "[REDACTED]" in redacted
        assert "john@example.com" not in redacted

    def test_mask_specific_types(self):
        """Test masking specific types only."""
        masker = PIIMasker()
        text = "Email: test@example.com, Phone: 555-123-4567"
        masked = masker.mask(text, types={PIIType.EMAIL})

        # Email should be masked, phone should not
        assert "555-123-4567" in masked
        assert "test@example.com" not in masked

    def test_mask_dict(self):
        """Test masking dictionary."""
        masker = PIIMasker()
        data = {
            "email": "test@example.com",
            "name": "John Doe",
        }
        masked = masker.get_masked_dict(data)

        assert "test@example.com" not in masked["email"]
        assert masked["name"] == "John Doe"  # No PII

    def test_mask_nested_dict(self):
        """Test masking nested dictionary."""
        masker = PIIMasker()
        data = {
            "user": {
                "email": "test@example.com",
            }
        }
        masked = masker.get_masked_dict(data)

        assert "test@example.com" not in masked["user"]["email"]


# ============================================================================
# Hook Tests
# ============================================================================


class TestPIIDetectionHook:
    """Tests for PIIDetectionHook."""

    def test_check_text(self):
        """Test checking text."""
        hook = PIIDetectionHook()
        result = hook.check_text("Email: test@example.com")

        assert result.has_pii
        assert PIIType.EMAIL in result.types

    def test_check_dict(self):
        """Test checking dictionary."""
        hook = PIIDetectionHook()
        result = hook.check_dict({"email": "test@example.com"})

        assert result.has_pii

    def test_block_on_detect(self):
        """Test blocking on detection."""
        hook = PIIDetectionHook(
            block_on_detect=True,
            blocked_types={PIIType.SSN},
        )

        # SSN should block
        result = hook.check_text("SSN: 123-45-6789")
        assert result.should_block

        # Email should not block
        result = hook.check_text("Email: test@example.com")
        assert not result.should_block

    def test_check_request(self):
        """Test checking request data."""
        hook = PIIDetectionHook()
        result = hook.check_request({"input": "Email: test@example.com"})

        assert result.has_pii

    def test_check_response(self):
        """Test checking response data."""
        hook = PIIDetectionHook()

        # Dict response
        result = hook.check_response({"email": "test@example.com"})
        assert result.has_pii

        # String response
        result = hook.check_response("Email: test@example.com")
        assert result.has_pii


class TestPIIMaskingHook:
    """Tests for PIIMaskingHook."""

    def test_mask_text(self):
        """Test masking text."""
        hook = PIIMaskingHook()
        masked = hook.mask_text("Email: test@example.com")

        assert "test@example.com" not in masked

    def test_mask_dict(self):
        """Test masking dictionary."""
        hook = PIIMaskingHook()
        masked = hook.mask_dict({"email": "test@example.com"})

        assert "test@example.com" not in masked["email"]

    def test_mask_request(self):
        """Test masking request."""
        hook = PIIMaskingHook()
        request = {"input": "Email: test@example.com"}
        masked = hook.mask_request(request)

        assert "test@example.com" not in masked["input"]

    def test_mask_response(self):
        """Test masking response."""
        hook = PIIMaskingHook()

        # Dict response
        masked = hook.mask_response({"email": "test@example.com"})
        assert "test@example.com" not in masked["email"]

        # String response
        masked = hook.mask_response("Email: test@example.com")
        assert "test@example.com" not in masked

    def test_disabled_masking(self):
        """Test disabled masking."""
        hook = PIIMaskingHook(mask_requests=False)
        request = {"email": "test@example.com"}
        masked = hook.mask_request(request)

        assert masked == request  # Should be unchanged
