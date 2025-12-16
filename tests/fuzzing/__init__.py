"""Fuzzy testing package using Hypothesis.

This package contains property-based tests for:
- PII detection (test_pii_fuzzing.py)
- Cost calculations (test_cost_fuzzing.py)
- Reliability patterns (test_reliability_fuzzing.py)

Run with:
    pytest tests/fuzzing/ -v

Set HYPOTHESIS_PROFILE environment variable:
    HYPOTHESIS_PROFILE=ci pytest tests/fuzzing/  # More examples
    HYPOTHESIS_PROFILE=dev pytest tests/fuzzing/  # Default
    HYPOTHESIS_PROFILE=debug pytest tests/fuzzing/  # Verbose
"""
