# Template Repository Specification

FastAgentic templates are hosted in a separate repository to enable independent versioning, community contributions, and dynamic discovery.

**Repository:** `github.com/fastagentic/fastagentic-templates`

## Repository Structure

```
fastagentic-templates/
├── index.json                    # Template registry (auto-generated)
├── schema/
│   └── template.schema.json      # JSON Schema for template metadata
├── scripts/
│   ├── build-index.py            # Generates index.json from templates
│   ├── validate.py               # Validates template structure
│   └── test-templates.py         # Tests all templates build correctly
├── templates/
│   ├── official/                 # Official FastAgentic templates
│   │   ├── pydanticai/
│   │   ├── langgraph/
│   │   ├── crewai/
│   │   ├── langchain/
│   │   └── minimal/
│   └── community/                # Community-contributed templates
│       ├── autogen/
│       ├── dspy/
│       └── ...
├── .github/
│   └── workflows/
│       ├── validate.yml          # PR validation
│       ├── build-index.yml       # Auto-rebuild index on merge
│       └── release.yml           # Tag-based releases
├── CONTRIBUTING.md
└── README.md
```

## Index Schema (`index.json`)

The index is the source of truth for template discovery:

```json
{
  "$schema": "./schema/template.schema.json",
  "version": "1.0.0",
  "generated_at": "2025-01-15T10:30:00Z",
  "templates": [
    {
      "name": "pydanticai",
      "display_name": "PydanticAI",
      "description": "Type-safe agents with Pydantic validation and dependency injection",
      "category": "official",
      "framework": "pydanticai",
      "version": "0.2.0",
      "min_fastagentic_version": "0.2.0",
      "path": "templates/official/pydanticai",
      "author": {
        "name": "FastAgentic Team",
        "url": "https://github.com/fastagentic"
      },
      "tags": ["type-safe", "structured-output", "recommended"],
      "features": {
        "streaming": true,
        "checkpointing": true,
        "mcp": true,
        "a2a": true
      },
      "dependencies": {
        "pydantic-ai": ">=0.1.0",
        "fastagentic": ">=0.2.0"
      },
      "created_at": "2025-01-01T00:00:00Z",
      "updated_at": "2025-01-15T00:00:00Z",
      "downloads": 1250,
      "stars": 45
    },
    {
      "name": "langgraph",
      "display_name": "LangGraph",
      "description": "Stateful graph workflows with node-level checkpointing",
      "category": "official",
      "framework": "langgraph",
      "version": "0.2.0",
      "min_fastagentic_version": "0.2.0",
      "path": "templates/official/langgraph",
      "author": {
        "name": "FastAgentic Team",
        "url": "https://github.com/fastagentic"
      },
      "tags": ["workflows", "graphs", "human-in-the-loop"],
      "features": {
        "streaming": true,
        "checkpointing": true,
        "mcp": true,
        "a2a": true,
        "interrupts": true
      },
      "dependencies": {
        "langgraph": ">=0.2.0",
        "fastagentic": ">=0.2.0"
      }
    },
    {
      "name": "autogen",
      "display_name": "AutoGen Multi-Agent",
      "description": "Microsoft AutoGen multi-agent conversations",
      "category": "community",
      "framework": "autogen",
      "version": "0.1.0",
      "min_fastagentic_version": "0.2.0",
      "path": "templates/community/autogen",
      "author": {
        "name": "Community Contributor",
        "github": "contributor-username"
      },
      "tags": ["multi-agent", "conversations", "microsoft"],
      "features": {
        "streaming": true,
        "checkpointing": false,
        "mcp": true,
        "a2a": true
      },
      "dependencies": {
        "pyautogen": ">=0.2.0",
        "fastagentic": ">=0.2.0"
      },
      "verified": true,
      "verification_date": "2025-01-10T00:00:00Z"
    }
  ],
  "categories": {
    "official": {
      "display_name": "Official Templates",
      "description": "Maintained by the FastAgentic team",
      "badge": "official"
    },
    "community": {
      "display_name": "Community Templates",
      "description": "Contributed by the community",
      "badge": "community"
    },
    "experimental": {
      "display_name": "Experimental",
      "description": "Cutting-edge templates, may have breaking changes",
      "badge": "experimental"
    }
  }
}
```

## Template Structure

Each template directory must contain:

```
templates/{category}/{name}/
├── template.json                 # Template metadata (required)
├── app.py                        # FastAgentic entry point (required)
├── agents/                       # Agent definitions
│   └── main.py
├── models/                       # Pydantic models
│   ├── inputs.py
│   └── outputs.py
├── tools/                        # Tool definitions
│   └── __init__.py
├── config/
│   └── settings.yaml
├── tests/
│   ├── test_agent.py
│   └── test_contracts.py
├── .env.example                  # Environment template (required)
├── pyproject.toml                # Dependencies (required)
├── Dockerfile                    # Container build (required)
├── docker-compose.yml            # Local dev stack
├── k8s/                          # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
└── README.md                     # Template documentation (required)
```

## Template Metadata (`template.json`)

Each template must have a `template.json` file:

```json
{
  "$schema": "https://raw.githubusercontent.com/fastagentic/fastagentic-templates/main/schema/template.schema.json",
  "name": "pydanticai",
  "display_name": "PydanticAI",
  "description": "Type-safe agents with Pydantic validation and dependency injection",
  "version": "0.2.0",
  "min_fastagentic_version": "0.2.0",
  "framework": "pydanticai",
  "author": {
    "name": "FastAgentic Team",
    "url": "https://github.com/fastagentic"
  },
  "license": "MIT",
  "tags": ["type-safe", "structured-output", "recommended"],
  "features": {
    "streaming": true,
    "checkpointing": true,
    "mcp": true,
    "a2a": true
  },
  "prompts": [
    {
      "name": "project_name",
      "message": "Project name",
      "default": "my-agent",
      "validate": "^[a-z][a-z0-9-]*$"
    },
    {
      "name": "llm_provider",
      "message": "LLM provider",
      "type": "select",
      "choices": ["openai", "anthropic", "google", "azure"],
      "default": "openai"
    },
    {
      "name": "durable_store",
      "message": "Durable store",
      "type": "select",
      "choices": ["redis", "postgres", "none"],
      "default": "redis"
    },
    {
      "name": "auth_enabled",
      "message": "Enable authentication?",
      "type": "confirm",
      "default": false
    }
  ],
  "post_create": [
    "pip install -e .",
    "cp .env.example .env"
  ],
  "post_create_message": "Project created! Edit .env with your API keys, then run: fastagentic run --reload"
}
```

## Index Build Script

The `scripts/build-index.py` script scans all templates and generates `index.json`:

```python
#!/usr/bin/env python3
"""Build the template index from template directories."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

TEMPLATES_DIR = Path("templates")
INDEX_FILE = Path("index.json")
SCHEMA_FILE = Path("schema/template.schema.json")


def scan_templates() -> list[dict]:
    """Scan all template directories and collect metadata."""
    templates = []

    for category in ["official", "community", "experimental"]:
        category_dir = TEMPLATES_DIR / category
        if not category_dir.exists():
            continue

        for template_dir in sorted(category_dir.iterdir()):
            if not template_dir.is_dir():
                continue

            metadata_file = template_dir / "template.json"
            if not metadata_file.exists():
                print(f"Warning: {template_dir} missing template.json, skipping")
                continue

            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Add computed fields
                metadata["category"] = category
                metadata["path"] = str(template_dir)

                # Preserve stats if they exist in current index
                metadata.setdefault("downloads", 0)
                metadata.setdefault("stars", 0)
                metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

                templates.append(metadata)
                print(f"Added: {category}/{metadata['name']}")

            except json.JSONDecodeError as e:
                print(f"Error parsing {metadata_file}: {e}")
                continue

    return templates


def merge_stats(templates: list[dict], existing_index: dict) -> list[dict]:
    """Merge download/star stats from existing index."""
    existing = {t["name"]: t for t in existing_index.get("templates", [])}

    for template in templates:
        if template["name"] in existing:
            old = existing[template["name"]]
            template["downloads"] = old.get("downloads", 0)
            template["stars"] = old.get("stars", 0)
            template["created_at"] = old.get("created_at", template["created_at"])

    return templates


def build_index() -> dict:
    """Build the complete index."""
    # Load existing index for stats preservation
    existing_index = {}
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            existing_index = json.load(f)

    templates = scan_templates()
    templates = merge_stats(templates, existing_index)

    # Sort: official first, then by name
    templates.sort(key=lambda t: (
        0 if t["category"] == "official" else 1,
        t["name"]
    ))

    return {
        "$schema": str(SCHEMA_FILE),
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "templates": templates,
        "categories": {
            "official": {
                "display_name": "Official Templates",
                "description": "Maintained by the FastAgentic team",
                "badge": "official"
            },
            "community": {
                "display_name": "Community Templates",
                "description": "Contributed by the community",
                "badge": "community"
            },
            "experimental": {
                "display_name": "Experimental",
                "description": "Cutting-edge templates, may have breaking changes",
                "badge": "experimental"
            }
        }
    }


def main():
    index = build_index()

    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nGenerated {INDEX_FILE} with {len(index['templates'])} templates")


if __name__ == "__main__":
    main()
```

## Validation Script

The `scripts/validate.py` script validates template structure:

```python
#!/usr/bin/env python3
"""Validate template structure and metadata."""

import json
import sys
from pathlib import Path

import jsonschema

REQUIRED_FILES = [
    "template.json",
    "app.py",
    ".env.example",
    "pyproject.toml",
    "Dockerfile",
    "README.md",
]

REQUIRED_DIRS = [
    "tests",
]


def validate_template(template_dir: Path) -> list[str]:
    """Validate a single template, return list of errors."""
    errors = []

    # Check required files
    for filename in REQUIRED_FILES:
        if not (template_dir / filename).exists():
            errors.append(f"Missing required file: {filename}")

    # Check required directories
    for dirname in REQUIRED_DIRS:
        if not (template_dir / dirname).is_dir():
            errors.append(f"Missing required directory: {dirname}")

    # Validate template.json
    metadata_file = template_dir / "template.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Validate against schema
            schema_file = Path("schema/template.schema.json")
            if schema_file.exists():
                with open(schema_file) as f:
                    schema = json.load(f)
                jsonschema.validate(metadata, schema)

            # Additional validations
            if "name" not in metadata:
                errors.append("template.json missing 'name' field")
            if "version" not in metadata:
                errors.append("template.json missing 'version' field")
            if "min_fastagentic_version" not in metadata:
                errors.append("template.json missing 'min_fastagentic_version' field")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in template.json: {e}")
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")

    # Validate pyproject.toml has fastagentic dependency
    pyproject = template_dir / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        if "fastagentic" not in content:
            errors.append("pyproject.toml must include fastagentic dependency")

    # Validate tests exist
    tests_dir = template_dir / "tests"
    if tests_dir.is_dir():
        test_files = list(tests_dir.glob("test_*.py"))
        if not test_files:
            errors.append("tests/ directory must contain test_*.py files")

    return errors


def main():
    templates_dir = Path("templates")
    all_errors = {}

    for category in ["official", "community", "experimental"]:
        category_dir = templates_dir / category
        if not category_dir.exists():
            continue

        for template_dir in category_dir.iterdir():
            if not template_dir.is_dir():
                continue

            errors = validate_template(template_dir)
            if errors:
                all_errors[str(template_dir)] = errors

    if all_errors:
        print("Validation errors found:\n")
        for path, errors in all_errors.items():
            print(f"{path}:")
            for error in errors:
                print(f"  - {error}")
        sys.exit(1)
    else:
        print("All templates valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

## GitHub Actions

### PR Validation (`.github/workflows/validate.yml`)

```yaml
name: Validate Templates

on:
  pull_request:
    paths:
      - 'templates/**'
      - 'schema/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install jsonschema

      - name: Validate templates
        run: python scripts/validate.py

      - name: Test template builds
        run: python scripts/test-templates.py

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install ruff

      - name: Lint Python files
        run: ruff check templates/
```

### Auto-Build Index (`.github/workflows/build-index.yml`)

```yaml
name: Build Index

on:
  push:
    branches: [main]
    paths:
      - 'templates/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Build index
        run: python scripts/build-index.py

      - name: Commit index
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: rebuild template index"
          file_pattern: index.json
```

## CLI Integration

The FastAgentic CLI fetches templates from the repository:

```python
# fastagentic/cli/templates.py

import httpx
from functools import lru_cache

TEMPLATES_INDEX_URL = "https://raw.githubusercontent.com/fastagentic/fastagentic-templates/main/index.json"
TEMPLATES_BASE_URL = "https://github.com/fastagentic/fastagentic-templates/archive/refs/heads/main.zip"


@lru_cache(maxsize=1)
def fetch_template_index() -> dict:
    """Fetch the template index from the repository."""
    response = httpx.get(TEMPLATES_INDEX_URL, timeout=10)
    response.raise_for_status()
    return response.json()


def list_templates(category: str | None = None) -> list[dict]:
    """List available templates."""
    index = fetch_template_index()
    templates = index["templates"]

    if category:
        templates = [t for t in templates if t["category"] == category]

    return templates


def get_template(name: str) -> dict | None:
    """Get template metadata by name."""
    index = fetch_template_index()
    for template in index["templates"]:
        if template["name"] == name:
            return template
    return None
```

## Local Template Override

Users can add local templates that override or supplement remote ones:

```bash
# ~/.fastagentic/templates/
my-custom-template/
├── template.json
├── app.py
└── ...
```

```python
# CLI checks local templates first
def resolve_template(name: str) -> Path | str:
    """Resolve template to local path or remote URL."""
    # Check local templates first
    local_dir = Path.home() / ".fastagentic" / "templates" / name
    if local_dir.exists():
        return local_dir

    # Fall back to remote
    template = get_template(name)
    if template:
        return f"{TEMPLATES_BASE_URL}#{template['path']}"

    raise TemplateNotFoundError(f"Template '{name}' not found")
```

## Next Steps

- [Contributing Templates](contributing.md) - How to contribute
- [Template Reference](index.md) - Available templates
- [CLI Usage](../cli.md) - `fastagentic new` command
