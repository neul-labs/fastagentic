"""Conftest for fuzzy testing with Hypothesis."""

import pytest
from hypothesis import HealthCheck, Verbosity, settings

# Configure Hypothesis settings for CI vs local
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=1000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def pytest_configure(config):
    """Load appropriate Hypothesis profile based on environment."""
    import os

    profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile)


@pytest.fixture
def pathological_strings():
    """Generate strings known to cause ReDoS in naive regex patterns."""
    return [
        "a" * 10000,
        "a" * 1000 + "@" + "b" * 1000,
        "." * 1000,
        "\x00" * 100,
        "a@" * 500,
        "test@" + "a" * 500 + ".com",
        "1234567890" * 100,
        "1-2-3-4-5-6-7-8-9-0" * 50,
    ]


@pytest.fixture
def unicode_test_strings():
    """Unicode strings for testing international character handling."""
    return [
        "用户@例子.com",
        "пользователь@пример.ру",
        "مستخدم@مثال.com",
        "ユーザー@例え.jp",
        "😀@emoji.com",
        "test\u200b@zero-width.com",  # Zero-width space
        "test\ufeff@bom.com",  # BOM character
    ]
