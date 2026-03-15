# Prompt Management

FastAgentic provides built-in prompt management with templates, versioning, and A/B testing.

## Overview

- **Prompt Templates** - Variable substitution with conditionals and loops
- **Prompt Registry** - Centralized prompt storage with versioning
- **A/B Testing** - Test prompt variants with statistical analysis

## Quick Start

```python
from fastagentic import (
    PromptTemplate,
    PromptVariable,
    PromptRegistry,
    ABTest,
    PromptVariant,
)

# Create a template
template = PromptTemplate(
    name="support_agent",
    content="""
You are a {{role}} assistant for {{company}}.

{{#if formal}}
Please maintain a professional tone at all times.
{{/if}}

{{#each guidelines}}
- {{this}}
{{/each}}
""",
    variables=[
        PromptVariable(name="role", required=True),
        PromptVariable(name="company", required=True),
        PromptVariable(name="formal", type=VariableType.BOOLEAN, default=True),
        PromptVariable(name="guidelines", type=VariableType.LIST),
    ],
)

# Render the template
prompt = template.render(
    role="support",
    company="Acme Inc",
    guidelines=["Be helpful", "Be concise", "Escalate when needed"],
)
```

## Prompt Templates

### Variable Types

```python
from fastagentic.prompts import VariableType

class VariableType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
```

### Template Syntax

**Variable Substitution:**
```
Hello, {{name}}!
```

**Conditionals:**
```
{{#if premium}}
You have access to premium features.
{{else}}
Upgrade to unlock premium features.
{{/if}}
```

**Loops:**
```
Available tools:
{{#each tools}}
- {{name}}: {{description}}
{{/each}}
```

**Filters:**
```
{{name | uppercase}}
{{description | truncate:100}}
{{items | join:", "}}
```

### Variable Validation

```python
from fastagentic import PromptVariable, VariableType

# Required variable
PromptVariable(name="user_name", required=True)

# With default value
PromptVariable(name="tone", default="friendly")

# With type validation
PromptVariable(name="max_tokens", type=VariableType.INTEGER)

# With custom validator
PromptVariable(
    name="email",
    validator=lambda x: "@" in x,
)
```

## Prompt Registry

Centralized storage for prompt templates with versioning:

```python
from fastagentic import PromptRegistry, PromptVersion

registry = PromptRegistry()

# Register a template
registry.register(template)

# Get and render
prompt = registry.render("support_agent", role="support", company="Acme")

# Version management
registry.register(template_v2, version="2.0.0")

# Get specific version
prompt = registry.render("support_agent", version="1.0.0", ...)

# List versions
versions = registry.list_versions("support_agent")
for v in versions:
    print(f"{v.version}: {v.description}")
```

### Version Pinning

```python
# Pin to specific version
registry.set_default_version("support_agent", "1.2.0")

# Pin to latest
registry.set_default_version("support_agent", "latest")

# Environment-based versioning
registry.set_environment_version(
    "support_agent",
    production="1.0.0",
    staging="2.0.0-beta",
)
```

## A/B Testing

Test prompt variants to optimize performance:

```python
from fastagentic import ABTest, PromptVariant

# Create test with variants
test = ABTest(
    name="greeting_test",
    variants=[
        PromptVariant(
            name="formal",
            template=formal_template,
            weight=50,
        ),
        PromptVariant(
            name="casual",
            template=casual_template,
            weight=50,
        ),
    ],
)

# Select variant for user (sticky assignment)
variant = test.select_variant(user_id="user-123")
prompt = variant.template.render(...)

# Record metrics
test.record_impression(variant.name)

# After user interaction...
test.record_conversion(variant.name)

# Get results
results = test.get_results()
print(f"Control conversion: {results.variants['formal'].conversion_rate:.2%}")
print(f"Treatment conversion: {results.variants['casual'].conversion_rate:.2%}")
print(f"Winner: {results.winner}")
```

### Selection Strategies

```python
from fastagentic.prompts import VariantSelectionStrategy

# Random selection (default)
test = ABTest(
    name="test",
    variants=variants,
    strategy=VariantSelectionStrategy.RANDOM,
)

# Sticky - same user always gets same variant
test = ABTest(
    name="test",
    variants=variants,
    strategy=VariantSelectionStrategy.STICKY,
)

# Round robin
test = ABTest(
    name="test",
    variants=variants,
    strategy=VariantSelectionStrategy.ROUND_ROBIN,
)

# Gradual rollout
test = ABTest(
    name="test",
    variants=variants,
    strategy=VariantSelectionStrategy.GRADUAL_ROLLOUT,
    rollout_percent=10,  # Start with 10% getting new variant
)
```

### Statistical Analysis

```python
# Get detailed results
results = test.get_results()

print(f"Total impressions: {results.total_impressions}")
print(f"Test duration: {results.duration_hours:.1f} hours")
print(f"Statistical significance: {results.is_significant}")
print(f"Confidence level: {results.confidence:.2%}")

# Complete the test
final = test.complete_test(winner="casual")  # Or auto-select winner
print(f"Test completed. Winner: {final.winner}")
```

## Integration with App

```python
from fastagentic import App, prompt

app = App(title="My Agent")

# Register prompts via decorator
@prompt(name="system", description="System prompt")
def system_prompt() -> str:
    return "You are a helpful assistant."

# Dynamic prompts with variables
@prompt(name="greeting", description="Greeting prompt")
def greeting_prompt(name: str, formal: bool = True) -> str:
    if formal:
        return f"Good day, {name}. How may I assist you?"
    return f"Hey {name}! What's up?"
```

## Best Practices

1. **Version everything** - Never modify prompts in place; create new versions
2. **Use variables** - Parameterize prompts for flexibility and reuse
3. **Test before deploying** - A/B test significant prompt changes
4. **Monitor metrics** - Track conversion rates, user satisfaction, task completion
5. **Document changes** - Add descriptions to versions explaining what changed
6. **Gradual rollout** - Use gradual rollout for risky changes
