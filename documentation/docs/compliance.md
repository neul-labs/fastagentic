# Compliance & PII Detection

FastAgentic includes compliance tools for detecting and handling Personally Identifiable Information (PII).

## Quick Start

```python
from fastagentic import PIIDetector, PIIMasker

# Detect PII
detector = PIIDetector()
matches = detector.detect("Email me at john@example.com")

for match in matches:
    print(f"{match.type}: {match.value}")

# Mask PII
masker = PIIMasker()
safe_text = masker.mask("Call me at 555-123-4567")
print(safe_text)  # "Call me at 555-***-****"
```

## PII Detection

### Supported PII Types

```python
from fastagentic import PIIType

# Built-in PII types
PIIType.EMAIL           # Email addresses
PIIType.PHONE           # Phone numbers
PIIType.SSN             # Social Security Numbers
PIIType.CREDIT_CARD     # Credit card numbers
PIIType.IP_ADDRESS      # IP addresses
PIIType.DATE_OF_BIRTH   # Birth dates
PIIType.ADDRESS         # Physical addresses
PIIType.NAME            # Person names
PIIType.CUSTOM          # Custom patterns
```

### Basic Detection

```python
from fastagentic import PIIDetector

detector = PIIDetector()

# Detect all PII
text = "Contact: john@example.com, Phone: 555-123-4567"
matches = detector.detect(text)

for match in matches:
    print(f"Type: {match.type}")
    print(f"Value: {match.value}")
    print(f"Position: {match.start}-{match.end}")
    print(f"Confidence: {match.confidence}")

# Quick checks
has_pii = detector.contains_pii(text)
pii_types = detector.get_pii_types(text)
```

### Configuration

```python
from fastagentic import PIIDetector, PIIConfig, PIIType

config = PIIConfig(
    # Only detect specific types
    enabled_types={PIIType.EMAIL, PIIType.PHONE},

    # Minimum confidence threshold
    min_confidence=0.8,

    # Values to ignore
    allowlist={"support@company.com", "1-800-COMPANY"},

    # Additional values to flag
    blocklist={"internal-secret", "api-key-xyz"},
)

detector = PIIDetector(config=config)
```

### Custom Patterns

```python
from fastagentic.compliance.pii import PIIPattern

# Define custom pattern
custom = PIIPattern(
    type=PIIType.CUSTOM,
    pattern=r"CUST-[0-9]{6}",  # Customer ID format
    confidence=1.0,
    description="Customer ID",
)

config = PIIConfig(custom_patterns=[custom])
detector = PIIDetector(config=config)

matches = detector.detect("Customer ID: CUST-123456")
```

## PII Masking

### Basic Masking

```python
from fastagentic import PIIMasker

masker = PIIMasker()

# Mask PII with asterisks
text = "Email: john@example.com"
masked = masker.mask(text)
# "Email: jo**@*****le.com"

# Show PII type in mask
masked = masker.mask(text, show_type=True)
# "Email: [EMAIL:jo**@*****le.com]"

# Full redaction
redacted = masker.redact(text)
# "Email: [REDACTED]"
```

### Selective Masking

```python
from fastagentic import PIIMasker, PIIType

masker = PIIMasker()

text = "Email: test@example.com, SSN: 123-45-6789"

# Only mask specific types
masked = masker.mask(text, types={PIIType.SSN})
# "Email: test@example.com, SSN: ***-**-****"
```

### Masking Dictionaries

```python
masker = PIIMasker()

data = {
    "email": "john@example.com",
    "phone": "555-123-4567",
    "message": "Hello world",
}

masked_data = masker.get_masked_dict(data)
# {
#     "email": "jo**@*****le.com",
#     "phone": "555-***-****",
#     "message": "Hello world",
# }
```

### Nested Data

```python
data = {
    "user": {
        "contact": {
            "email": "john@example.com",
        }
    }
}

masked = masker.get_masked_dict(data)
# Handles nested structures automatically
```

## Compliance Hooks

### Detection Hook

```python
from fastagentic import PIIDetectionHook, PIIType

hook = PIIDetectionHook(
    block_on_detect=True,
    blocked_types={PIIType.SSN, PIIType.CREDIT_CARD},
)

# Check text
result = hook.check_text("SSN: 123-45-6789")
if result.should_block:
    raise ValueError("Sensitive PII detected")

# Check request data
result = hook.check_request({"input": "My SSN is 123-45-6789"})
if result.has_pii:
    print(f"PII types found: {result.types}")

# Check response data
result = hook.check_response({"output": "Contact: john@example.com"})
```

### Masking Hook

```python
from fastagentic import PIIMaskingHook

hook = PIIMaskingHook(
    mask_requests=True,
    mask_responses=True,
)

# Mask request before processing
request = {"input": "Email: john@example.com"}
safe_request = hook.mask_request(request)

# Mask response before returning
response = {"output": "Call 555-123-4567"}
safe_response = hook.mask_response(response)
```

### Selective Masking

```python
# Only mask responses, not requests
hook = PIIMaskingHook(
    mask_requests=False,
    mask_responses=True,
)

# Disabled masking passes through unchanged
request = {"email": "test@example.com"}
result = hook.mask_request(request)
# Returns original request unchanged
```

## Integration with App

```python
from fastagentic import App
from fastagentic import PIIDetectionHook, PIIMaskingHook

app = App()

# Add PII hooks
detection_hook = PIIDetectionHook(block_on_detect=True)
masking_hook = PIIMaskingHook()

@app.agent_endpoint("/chat")
async def chat(message: str):
    # Check input for sensitive PII
    result = detection_hook.check_text(message)
    if result.should_block:
        return {"error": "Cannot process sensitive information"}

    # Process message...
    response = await process_message(message)

    # Mask any PII in response
    return masking_hook.mask_response(response)
```

## Audit Logging

```python
from fastagentic import PIIDetectionHook, AuditLogger

audit = AuditLogger()
hook = PIIDetectionHook()

async def check_and_log(text: str, user_id: str):
    result = hook.check_text(text)

    if result.has_pii:
        await audit.log(
            event_type="pii_detected",
            severity="warning",
            actor_id=user_id,
            details={
                "pii_types": [t.value for t in result.types],
                "blocked": result.should_block,
            },
        )

    return result
```

## Best Practices

1. **Always mask before logging**: Never log raw PII
2. **Block sensitive types**: SSN, credit cards should block by default
3. **Allowlist known values**: Company emails, support numbers
4. **Audit all detections**: Log when PII is found for compliance
5. **Test with realistic data**: Ensure patterns catch real-world formats
