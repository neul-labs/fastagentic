"""Tests for FastAgentic prompts module."""

import pytest

from fastagentic.prompts.registry import PromptMetadata, PromptRegistry, PromptVersion
from fastagentic.prompts.template import (
    PromptTemplate,
    PromptVariable,
    VariableType,
    render_template,
)
from fastagentic.prompts.testing import (
    ABTest,
    ABTestResult,
    PromptVariant,
    VariantMetrics,
    VariantSelectionStrategy,
)


class TestPromptVariable:
    """Tests for PromptVariable class."""

    def test_basic_variable(self):
        var = PromptVariable(name="name", type=VariableType.STRING)
        assert var.name == "name"
        assert var.required

    def test_variable_validation_required(self):
        var = PromptVariable(name="name", required=True)
        valid, error = var.validate(None)
        assert not valid
        assert "required" in error

    def test_variable_validation_type(self):
        var = PromptVariable(name="count", type=VariableType.NUMBER)
        valid, _ = var.validate(42)
        assert valid

        valid, error = var.validate("not a number")
        assert not valid
        assert "expected number" in error

    def test_variable_with_default(self):
        var = PromptVariable(name="role", default="user")
        valid, _ = var.validate(None)
        assert valid

    def test_variable_custom_validator(self):
        var = PromptVariable(
            name="email",
            validator=lambda v: "@" in v,
        )
        valid, _ = var.validate("test@example.com")
        assert valid

        valid, error = var.validate("invalid")
        assert not valid
        assert "failed custom validation" in error


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_simple_template(self):
        template = PromptTemplate(
            name="greeting",
            content="Hello, {{name}}!",
            variables=[PromptVariable(name="name")],
        )
        result = template.render(name="Alice")
        assert result == "Hello, Alice!"

    def test_template_validation(self):
        template = PromptTemplate(
            name="test",
            content="{{name}} {{age}}",
            variables=[
                PromptVariable(name="name", required=True),
                PromptVariable(name="age", type=VariableType.NUMBER),
            ],
        )
        valid, errors = template.validate(name="Alice", age=30)
        assert valid
        assert len(errors) == 0

        valid, errors = template.validate(age=30)
        assert not valid
        assert len(errors) == 1

    def test_template_with_defaults(self):
        template = PromptTemplate(
            name="test",
            content="{{greeting}}, {{name}}!",
            variables=[
                PromptVariable(name="greeting", default="Hello"),
                PromptVariable(name="name"),
            ],
        )
        result = template.render(name="Bob")
        assert result == "Hello, Bob!"

    def test_template_render_safe(self):
        template = PromptTemplate(
            name="test",
            content="{{name}}",
            variables=[PromptVariable(name="name", required=True)],
        )
        result, errors = template.render_safe()
        assert result is None
        assert len(errors) == 1

        result, errors = template.render_safe(name="Alice")
        assert result == "Alice"
        assert len(errors) == 0

    def test_get_required_variables(self):
        template = PromptTemplate(
            name="test",
            content="{{a}} {{b}} {{c}}",
            variables=[
                PromptVariable(name="a", required=True),
                PromptVariable(name="b", required=False),
                PromptVariable(name="c", required=True, default="default"),
            ],
        )
        required = template.get_required_variables()
        assert required == ["a"]


class TestRenderTemplate:
    """Tests for render_template function."""

    def test_simple_substitution(self):
        result = render_template("Hello, {{name}}!", {"name": "World"})
        assert result == "Hello, World!"

    def test_multiple_variables(self):
        result = render_template(
            "{{greeting}}, {{name}}!",
            {"greeting": "Hi", "name": "Alice"},
        )
        assert result == "Hi, Alice!"

    def test_conditional_true(self):
        result = render_template(
            "{{#if premium}}VIP Access{{/if}}",
            {"premium": True},
        )
        assert "VIP Access" in result

    def test_conditional_false(self):
        result = render_template(
            "{{#if premium}}VIP Access{{/if}}",
            {"premium": False},
        )
        assert "VIP Access" not in result

    def test_each_loop(self):
        result = render_template(
            "{{#each items}}{{this}}{{/each}}",
            {"items": ["a", "b", "c"]},
        )
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_filter_upper(self):
        result = render_template("{{name | upper}}", {"name": "alice"})
        assert result == "ALICE"

    def test_filter_lower(self):
        result = render_template("{{name | lower}}", {"name": "ALICE"})
        assert result == "alice"

    def test_filter_capitalize(self):
        result = render_template("{{name | capitalize}}", {"name": "alice"})
        assert result == "Alice"

    def test_filter_trim(self):
        result = render_template("{{text | trim}}", {"text": "  hello  "})
        assert result == "hello"


class TestPromptRegistry:
    """Tests for PromptRegistry class."""

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        registry = PromptRegistry()
        template = PromptTemplate(
            name="greeting",
            content="Hello, {{name}}!",
            variables=[PromptVariable(name="name")],
        )

        await registry.register(template, version="1.0.0")

        result = await registry.get("greeting")
        assert result is not None
        assert result.name == "greeting"

    @pytest.mark.asyncio
    async def test_version_management(self):
        registry = PromptRegistry()

        # Register v1
        template_v1 = PromptTemplate(
            name="greeting",
            content="Hello, {{name}}!",
        )
        await registry.register(template_v1, version="1.0.0")

        # Register v2
        template_v2 = PromptTemplate(
            name="greeting",
            content="Hi, {{name}}! How are you?",
        )
        await registry.register(template_v2, version="2.0.0")

        # Get specific versions
        v1 = await registry.get("greeting", version="1.0.0")
        v2 = await registry.get("greeting", version="2.0.0")

        assert "Hello" in v1.content
        assert "Hi" in v2.content

    @pytest.mark.asyncio
    async def test_activate_version(self):
        registry = PromptRegistry()

        template = PromptTemplate(name="test", content="v1")
        await registry.register(template, version="1.0.0", activate=True)

        template2 = PromptTemplate(name="test", content="v2")
        await registry.register(template2, version="2.0.0", activate=False)

        # v1 should still be active
        active = await registry.get("test")
        assert active.content == "v1"

        # Activate v2
        await registry.activate("test", "2.0.0")
        active = await registry.get("test")
        assert active.content == "v2"

    @pytest.mark.asyncio
    async def test_render(self):
        registry = PromptRegistry()
        template = PromptTemplate(
            name="greeting",
            content="Hello, {{user_name}}!",
            variables=[PromptVariable(name="user_name")],
        )
        await registry.register(template)

        result = await registry.render("greeting", user_name="Alice")
        assert result == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_list_prompts(self):
        registry = PromptRegistry()
        await registry.register(PromptTemplate(name="a", content="a"))
        await registry.register(PromptTemplate(name="b", content="b"))

        prompts = await registry.list_prompts()
        assert "a" in prompts
        assert "b" in prompts

    @pytest.mark.asyncio
    async def test_delete(self):
        registry = PromptRegistry()
        await registry.register(PromptTemplate(name="test", content="test"))

        result = await registry.get("test")
        assert result is not None

        await registry.delete("test")
        result = await registry.get("test")
        assert result is None


class TestPromptVersion:
    """Tests for PromptVersion class."""

    def test_content_hash(self):
        template = PromptTemplate(name="test", content="hello")
        version = PromptVersion(version="1.0.0", template=template)

        assert version.content_hash != ""
        assert len(version.content_hash) == 12


class TestPromptMetadata:
    """Tests for PromptMetadata class."""

    def test_timestamps(self):
        metadata = PromptMetadata(author="alice")
        assert metadata.created_datetime is not None
        assert metadata.updated_datetime is not None


class TestVariantMetrics:
    """Tests for VariantMetrics class."""

    def test_conversion_rate(self):
        metrics = VariantMetrics(impressions=100, conversions=25)
        assert metrics.conversion_rate == 0.25

    def test_conversion_rate_zero_impressions(self):
        metrics = VariantMetrics()
        assert metrics.conversion_rate == 0.0

    def test_avg_latency(self):
        metrics = VariantMetrics(impressions=10, total_latency_ms=1000)
        assert metrics.avg_latency_ms == 100.0


class TestABTest:
    """Tests for ABTest class."""

    def test_requires_two_variants(self):
        with pytest.raises(ValueError, match="at least 2 variants"):
            ABTest(
                name="test",
                variants=[
                    PromptVariant(
                        name="only-one",
                        template=PromptTemplate(name="t", content="t"),
                    )
                ],
            )

    def test_select_variant(self):
        test = ABTest(
            name="test",
            variants=[
                PromptVariant(
                    name="control",
                    template=PromptTemplate(name="t", content="control"),
                ),
                PromptVariant(
                    name="challenger",
                    template=PromptTemplate(name="t", content="challenger"),
                ),
            ],
        )

        variant = test.select_variant()
        assert variant.name in ["control", "challenger"]

    def test_sticky_selection(self):
        test = ABTest(
            name="test",
            variants=[
                PromptVariant(
                    name="control",
                    template=PromptTemplate(name="t", content="control"),
                ),
                PromptVariant(
                    name="challenger",
                    template=PromptTemplate(name="t", content="challenger"),
                ),
            ],
            strategy=VariantSelectionStrategy.STICKY,
        )

        # Same user should get same variant
        variant1 = test.select_variant(user_id="user-123")
        variant2 = test.select_variant(user_id="user-123")
        assert variant1.name == variant2.name

    def test_record_metrics(self):
        test = ABTest(
            name="test",
            variants=[
                PromptVariant(
                    name="control",
                    template=PromptTemplate(name="t", content="control"),
                ),
                PromptVariant(
                    name="challenger",
                    template=PromptTemplate(name="t", content="challenger"),
                ),
            ],
        )

        test.record_impression("control")
        test.record_conversion("control")
        test.record_latency("control", 150)
        test.record_tokens("control", 100)
        test.record_cost("control", 0.01)

        results = test.get_results()
        assert results.variant_metrics["control"].impressions == 1
        assert results.variant_metrics["control"].conversions == 1

    def test_get_results(self):
        test = ABTest(
            name="test",
            variants=[
                PromptVariant(
                    name="control",
                    template=PromptTemplate(name="t", content="control"),
                ),
                PromptVariant(
                    name="challenger",
                    template=PromptTemplate(name="t", content="challenger"),
                ),
            ],
        )

        results = test.get_results()
        assert results.test_name == "test"
        assert not results.is_complete

    def test_complete_test(self):
        test = ABTest(
            name="test",
            variants=[
                PromptVariant(
                    name="control",
                    template=PromptTemplate(name="t", content="control"),
                ),
                PromptVariant(
                    name="challenger",
                    template=PromptTemplate(name="t", content="challenger"),
                ),
            ],
        )

        results = test.complete_test(winner="challenger")
        assert results.is_complete
        assert results.winner == "challenger"


class TestABTestResult:
    """Tests for ABTestResult class."""

    def test_get_summary(self):
        result = ABTestResult(
            test_name="test",
            variant_metrics={
                "control": VariantMetrics(impressions=100, conversions=20),
                "challenger": VariantMetrics(impressions=100, conversions=30),
            },
        )

        summary = result.get_summary()
        assert summary["test_name"] == "test"
        assert summary["total_impressions"] == 200
        assert "control" in summary["variants"]
