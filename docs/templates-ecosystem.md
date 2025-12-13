# Template Ecosystem

FastAgentic v1.1 introduces a comprehensive template ecosystem with marketplace, versioning, and composition features.

## Quick Start

```python
from fastagentic.templates import (
    Template,
    TemplateMetadata,
    TemplateVariable,
    TemplateFile,
    LocalRegistry,
    Marketplace,
)

# Create a template
template = Template(
    metadata=TemplateMetadata(
        name="my-agent",
        description="A simple agent template",
        author="Your Name",
    ),
    variables=[
        TemplateVariable(name="project_name", description="Project name"),
    ],
    files=[
        TemplateFile(path="main.py", content="PROJECT = '{{ project_name }}'"),
    ],
)

# Render to directory
template.render("/path/to/output", {"project_name": "my-project"})
```

## Templates

### Template Variables

```python
from fastagentic.templates import TemplateVariable

# String variable
name_var = TemplateVariable(
    name="project_name",
    description="Name of the project",
    type="string",
    default="my-project",
    min_length=3,
    max_length=50,
    pattern=r"^[a-z][a-z0-9-]*$",
)

# Choice variable
framework_var = TemplateVariable(
    name="framework",
    description="Agent framework to use",
    type="choice",
    choices=["pydanticai", "langgraph", "crewai"],
)

# Boolean variable
streaming_var = TemplateVariable(
    name="enable_streaming",
    description="Enable streaming responses",
    type="boolean",
    default=True,
)
```

### Template Files

```python
from fastagentic.templates import TemplateFile

# Template file with variable substitution
main_py = TemplateFile(
    path="src/main.py",
    content='''
from fastagentic import App

app = App(title="{{ project_name }}")
''',
    is_template=True,
)

# Static file (no substitution)
data_file = TemplateFile(
    path="data/config.json",
    content='{"key": "value"}',
    is_template=False,
)

# Executable script
script = TemplateFile(
    path="scripts/run.sh",
    content="#!/bin/bash\npython main.py",
    executable=True,
)
```

### Template Versions

```python
from fastagentic.templates import TemplateVersion
from datetime import datetime

version = TemplateVersion(
    version="1.2.0",
    release_date=datetime.now(),
    changelog="Added streaming support",
    min_fastagentic_version="1.0.0",
)

# Check compatibility
if version.is_compatible("1.1.0"):
    print("Compatible!")
```

## Registries

### Local Registry

```python
from fastagentic.templates import LocalRegistry

# Create registry
registry = LocalRegistry("/path/to/templates")

# Add template
registry.add_template(template)

# List templates
templates = registry.list_templates()
templates = registry.list_templates(framework="pydanticai")

# Get template
template = registry.get_template("my-template")
template = registry.get_template("my-template", version="1.0.0")

# Search
results = registry.search("chatbot")

# Remove
registry.remove_template("old-template")
```

### Remote Registry

```python
from fastagentic.templates import RemoteRegistry, RemoteRegistryConfig

config = RemoteRegistryConfig(
    url="https://templates.fastagentic.dev",
    api_key="your-key",
    cache_ttl=3600,
)

registry = RemoteRegistry(config)
templates = registry.list_templates()
```

### Enterprise Registry

```python
from fastagentic.templates import EnterpriseRegistry, EnterpriseConfig
from fastagentic.templates.base import TemplateCategory

config = EnterpriseConfig(
    url="https://templates.mycompany.com",
    api_key="ent-key",
    organization="mycompany",
    require_approval=True,
    allowed_categories=[TemplateCategory.AGENT, TemplateCategory.WORKFLOW],
)

registry = EnterpriseRegistry(config)

# Publish private template
result = registry.publish_template(template, private=True)

# Get usage stats
stats = registry.get_usage_stats("my-template")

# Audit log
logs = registry.get_audit_log(template_name="my-template")
```

## Marketplace

### Browsing Templates

```python
from fastagentic.templates import Marketplace
from fastagentic.templates.base import TemplateCategory

marketplace = Marketplace()

# Browse
templates = marketplace.browse(
    category=TemplateCategory.RAG,
    framework="llamaindex",
    sort="popular",
)

# Get popular/recent/top-rated
popular = marketplace.get_popular(limit=10)
recent = marketplace.get_recent(limit=10)
top_rated = marketplace.get_top_rated(limit=10)

# By framework
pydanticai_templates = marketplace.get_by_framework("pydanticai")

# Search
results = marketplace.search("rag chatbot")
```

### Ratings and Reviews

```python
from fastagentic.templates import TemplateRating, TemplateReview

# Rate a template
marketplace.rate_template("rag-chatbot", 5)

# Submit review
review = TemplateReview(
    template_name="rag-chatbot",
    user_id="user-123",
    title="Excellent template!",
    content="Easy to set up and works great.",
    rating=5,
)
marketplace.submit_review(review)

# Get reviews
reviews = marketplace.get_reviews("rag-chatbot", sort="helpful")

# Mark helpful
marketplace.mark_review_helpful("rag-chatbot", "review-id")
```

### Publishing

```python
# Publish to marketplace
result = marketplace.publish_template(template)

# Update existing
result = marketplace.update_template("my-template", updated_template)

# Deprecate
marketplace.deprecate_template(
    "old-template",
    message="Use new-template instead",
    replacement="new-template",
)
```

## Template Composition

Combine multiple templates into one.

```python
from fastagentic.templates import (
    TemplateComposer,
    CompositionConfig,
    LocalRegistry,
)

registry = LocalRegistry("/path/to/templates")
composer = TemplateComposer(registry)

# Define composition
config = CompositionConfig(
    base_template="pydanticai-agent",
    include_templates=["rag-module", "auth-module"],
    override_strategy="merge",  # merge, replace, skip
    resolve_conflicts="base_wins",  # base_wins, include_wins, prompt
)

# Preview composition
preview = composer.preview_composition(config)
print(f"Total files: {preview['total_files']}")
print(f"Conflicts: {preview['potential_conflicts']}")

# Compose
result = composer.compose(config)

# Check conflicts
if result.has_unresolved_conflicts():
    for conflict in result.get_unresolved_conflicts():
        print(f"Conflict: {conflict.path}")

# Render composed template
result.template.render("/output", {"project_name": "composed-project"})

# Save composition
composer.save_composition(result, "/saved-templates/composed")
```

### Conflict Resolution

```python
config = CompositionConfig(
    base_template="base",
    include_templates=["module"],
    override_strategy="merge",  # Try to merge conflicting files
    resolve_conflicts="base_wins",  # Base template wins on conflict
)

# For merge strategy:
# - Python files: merge imports, concatenate content
# - JSON files: deep merge objects
# - YAML files: deep merge objects
# - Other files: keep base
```

## Best Practices

1. **Use meaningful variable names**: `project_name` not `pn`
2. **Provide sensible defaults**: Reduce required inputs
3. **Document variables**: Clear descriptions help users
4. **Version your templates**: Track changes with TemplateVersion
5. **Make templates composable**: Enable `composable=True`
6. **Test before publishing**: Validate with different inputs
