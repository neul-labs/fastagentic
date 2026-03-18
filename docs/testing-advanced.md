# Advanced Testing Guide

Fuzzy testing and mutation testing for FastAgentic.

## Overview

FastAgentic includes support for two advanced testing approaches:

1. **Fuzzy Testing (Hypothesis)** - Property-based testing that generates random inputs to find edge cases
2. **Mutation Testing (mutmut)** - Validates test quality by introducing deliberate code mutations

## Installation

```bash
# Install testing dependencies
uv sync --extra testing
```

## Fuzzy Testing with Hypothesis

Hypothesis generates random test inputs to find edge cases your regular tests might miss.

### Running Fuzzy Tests

```bash
# Run all fuzzy tests
pytest tests/fuzzing/ -v

# Run with CI profile (more examples)
HYPOTHESIS_PROFILE=ci pytest tests/fuzzing/

# Run with debug profile (verbose output)
HYPOTHESIS_PROFILE=debug pytest tests/fuzzing/
```

### Writing Fuzzy Tests

```python
from hypothesis import given, strategies as st

@given(
    input_tokens=st.integers(min_value=0, max_value=1_000_000),
    output_tokens=st.integers(min_value=0, max_value=1_000_000),
)
def test_cost_calculation_non_negative(input_tokens: int, output_tokens: int):
    """Cost should always be non-negative."""
    pricing = ModelPricing(
        model="test",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    )

    cost = pricing.calculate_cost(input_tokens, output_tokens)

    assert cost >= 0
    assert not math.isnan(cost)
```

### Hypothesis Profiles

Configure in `pyproject.toml`:

```toml
[tool.hypothesis]
deadline = 1000
max_examples = 100
suppress_health_check = ["too_slow"]
```

Or register profiles in conftest.py:

```python
from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=200)
settings.register_profile("dev", max_examples=50)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
```

### Common Strategies

```python
from hypothesis import strategies as st

# Integers
st.integers(min_value=0, max_value=100)

# Floats (excluding NaN/Inf)
st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Text
st.text(min_size=0, max_size=100)

# Emails
st.emails()

# Lists
st.lists(st.integers(), min_size=1, max_size=10)

# Composite strategies
st.one_of(st.none(), st.integers(), st.text())
```

## Mutation Testing with mutmut

Mutation testing verifies your tests actually catch bugs by introducing small code changes (mutations) and checking if tests fail.

### Running Mutation Tests

```bash
# Run mutation testing on specific module
mutmut run --paths-to-mutate=src/fastagentic/policy/engine.py

# View results
mutmut results

# Show specific mutation
mutmut show 42

# Generate HTML report
mutmut html
```

### Configuration

Configure in `pyproject.toml`:

```toml
[tool.mutmut]
paths_to_mutate = "src/fastagentic/"
tests_dir = "tests/"
runner = "python -m pytest -x --tb=no -q"
```

### Priority Modules for Mutation Testing

Focus mutation testing on high-risk modules:

```bash
# Policy engine (auth bypass risk)
mutmut run --paths-to-mutate=src/fastagentic/policy/engine.py

# RBAC (permission bypass risk)
mutmut run --paths-to-mutate=src/fastagentic/policy/rbac.py

# PII detection (data leak risk)
mutmut run --paths-to-mutate=src/fastagentic/compliance/pii.py

# Cost calculations (financial risk)
mutmut run --paths-to-mutate=src/fastagentic/cost/tracker.py
```

### Common Mutations

mutmut introduces these code changes:

| Mutation Type | Example |
|--------------|---------|
| Boundary conditions | `>=` → `>`, `<` → `<=` |
| Boolean logic | `and` → `or`, `not` removal |
| Arithmetic | `+` → `-`, `*` → `/` |
| Return values | `return True` → `return False` |
| Constants | `0` → `1`, `""` → "mutant"` |

### Writing Mutation-Resistant Tests

```python
def test_boundary_condition():
    """Test exactly at boundary to catch >= vs > mutations."""
    # Test exactly at threshold
    assert check_limit(100) == True   # At limit
    assert check_limit(101) == False  # Just over
    assert check_limit(99) == True    # Just under

def test_both_branches():
    """Test both true and false paths."""
    assert validate(valid_input) == True
    assert validate(invalid_input) == False  # Don't forget negative cases!
```

## CI Integration

### GitHub Actions

```yaml
name: Advanced Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  fuzzing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --extra testing

      - name: Run fuzzy tests
        env:
          HYPOTHESIS_PROFILE: ci
        run: uv run pytest tests/fuzzing/ -v

  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --extra testing

      - name: Run mutation tests
        run: |
          uv run mutmut run --paths-to-mutate=src/fastagentic/policy/engine.py
          uv run mutmut results
```

## Test Organization

```
tests/
├── fuzzing/
│   ├── __init__.py
│   ├── conftest.py           # Hypothesis configuration
│   ├── test_pii_fuzzing.py   # PII detection tests
│   ├── test_cost_fuzzing.py  # Cost calculation tests
│   └── test_reliability_fuzzing.py  # Rate limit, circuit breaker
├── mutation/
│   ├── conftest.py           # Shared fixtures
│   ├── test_policy_mutations.py
│   ├── test_rbac_mutations.py
│   └── test_budget_mutations.py
└── ...
```

## Best Practices

### Fuzzy Testing

1. **Start simple** - Begin with basic strategies, add constraints as needed
2. **Use `assume()`** - Filter out invalid combinations
3. **Set reasonable bounds** - Don't test with billion-element lists
4. **Check for crashes first** - Then check correctness

### Mutation Testing

1. **Focus on critical paths** - Auth, security, financial calculations
2. **Aim for 80%+ mutation score** - Some mutations are false positives
3. **Review surviving mutations** - They reveal test gaps
4. **Run incrementally** - Full mutation testing is slow

## Debugging

### Hypothesis

```bash
# Print generated examples
HYPOTHESIS_PROFILE=debug pytest tests/fuzzing/test_cost.py -v -s

# Reproduce specific failure
@given(st.integers())
@settings(database=None)  # Disable example database
def test_something(x):
    ...
```

### mutmut

```bash
# Show all surviving mutations
mutmut results

# Show specific mutation
mutmut show 42

# Run tests for specific mutation
mutmut run --mutation-id 42
```

## Reference

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [mutmut Documentation](https://mutmut.readthedocs.io/)
- [Property-Based Testing Guide](https://hypothesis.works/articles/what-is-property-based-testing/)
