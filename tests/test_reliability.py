"""Tests for reliability patterns."""

import asyncio

import pytest

from fastagentic.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    FallbackChain,
    RateLimit,
    RateLimitError,
    RetryError,
    RetryPolicy,
    Timeout,
    TimeoutError,
)


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test that successful functions don't retry."""
        call_count = 0

        async def success():
            nonlocal call_count
            call_count += 1
            return "success"

        policy = RetryPolicy(max_attempts=3)
        result = await policy.execute(success)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that failures trigger retries."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate_limit error")
            return "success"

        policy = RetryPolicy(max_attempts=3, initial_delay_ms=10, jitter=False)
        result = await policy.execute(fail_then_succeed)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exhausted(self):
        """Test that max attempts results in RetryError."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("rate_limit error")

        policy = RetryPolicy(max_attempts=3, initial_delay_ms=10, jitter=False)

        with pytest.raises(RetryError) as exc_info:
            await policy.execute(always_fail)

        assert exc_info.value.attempts == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_matching_error(self):
        """Test that non-matching errors don't trigger retries."""
        call_count = 0

        async def fail_with_other_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("some other error")

        policy = RetryPolicy(max_attempts=3, retry_on=["rate_limit"])

        with pytest.raises(ValueError):
            await policy.execute(fail_with_other_error)

        assert call_count == 1

    def test_get_delay_exponential(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(
            backoff="exponential",
            initial_delay_ms=1000,
            multiplier=2.0,
            jitter=False,
        )

        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0

    def test_get_delay_linear(self):
        """Test linear backoff calculation."""
        policy = RetryPolicy(
            backoff="linear",
            initial_delay_ms=1000,
            jitter=False,
        )

        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 3.0

    def test_get_delay_fixed(self):
        """Test fixed backoff calculation."""
        policy = RetryPolicy(
            backoff="fixed",
            initial_delay_ms=1000,
            jitter=False,
        )

        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 1.0
        assert policy.get_delay(3) == 1.0


class TestTimeout:
    """Tests for Timeout."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test that fast functions complete successfully."""

        async def fast():
            return "done"

        timeout = Timeout(total_ms=1000)
        result = await timeout.execute(fast)

        assert result == "done"

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test that slow functions raise TimeoutError."""

        async def slow():
            await asyncio.sleep(1)
            return "done"

        timeout = Timeout(total_ms=50)

        with pytest.raises(TimeoutError) as exc_info:
            await timeout.execute(slow)

        assert exc_info.value.timeout_ms == 50

    @pytest.mark.asyncio
    async def test_llm_call_timeout(self):
        """Test LLM-specific timeout."""

        async def llm_call():
            return "response"

        timeout = Timeout(llm_call_ms=500)
        result = await timeout.execute_llm_call(llm_call)

        assert result == "response"

    def test_timeout_properties(self):
        """Test timeout conversion properties."""
        timeout = Timeout(
            total_ms=300000,
            llm_call_ms=60000,
            tool_call_ms=30000,
            checkpoint_ms=5000,
        )

        assert timeout.total_seconds == 300.0
        assert timeout.llm_call_seconds == 60.0
        assert timeout.tool_call_seconds == 30.0
        assert timeout.checkpoint_seconds == 5.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_closed_state_allows_requests(self):
        """Test that closed circuit allows requests."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def success():
            return "ok"

        result = await breaker.execute(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after failure threshold."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            failure_window_ms=60000,
        )

        async def fail():
            raise Exception("error")

        for _ in range(3):
            try:
                await breaker.execute(fail)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_requests(self):
        """Test that open circuit rejects requests."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.trip()

        async def success():
            return "ok"

        with pytest.raises(CircuitOpenError):
            await breaker.execute(success)

    @pytest.mark.asyncio
    async def test_reset_closes_circuit(self):
        """Test that reset returns circuit to closed state."""
        breaker = CircuitBreaker()
        breaker.trip()

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestFallbackChain:
    """Tests for FallbackChain."""

    def test_get_all_models(self):
        """Test getting all models in chain."""
        chain = FallbackChain(
            primary="gpt-4o",
            fallbacks=[
                {"model": "gpt-4o-mini"},
                {"model": "gpt-3.5-turbo"},
            ],
        )

        models = chain.get_all_models()
        assert models == ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

    def test_get_fallback_for_rate_limit(self):
        """Test fallback selection for rate limit errors."""
        chain = FallbackChain(
            primary="gpt-4o",
            fallbacks=[
                {"model": "gpt-4o-mini", "on": ["rate_limit"]},
            ],
        )

        fallback = chain.get_fallback_for_error(Exception("429 rate limit exceeded"))
        assert fallback == "gpt-4o-mini"

    def test_no_fallback_for_unmatched_error(self):
        """Test that unmatched errors return no fallback."""
        chain = FallbackChain(
            primary="gpt-4o",
            fallbacks=[
                {"model": "gpt-4o-mini", "on": ["rate_limit"]},
            ],
        )

        fallback = chain.get_fallback_for_error(Exception("validation error"))
        assert fallback is None

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self):
        """Test fallback execution on error."""
        chain = FallbackChain(
            primary="gpt-4o",
            fallbacks=[
                {"model": "gpt-4o-mini", "on": ["rate_limit"]},
            ],
        )

        used_models = []

        async def llm_call(*, model: str) -> str:
            used_models.append(model)
            if model == "gpt-4o":
                raise Exception("429 rate_limit")
            return f"response from {model}"

        result = await chain.execute_with_fallback(llm_call)

        assert result == "response from gpt-4o-mini"
        assert used_models == ["gpt-4o", "gpt-4o-mini"]


class TestRateLimit:
    """Tests for RateLimit."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        """Test that requests under limit are allowed."""
        limiter = RateLimit(rpm=10, window_seconds=60)

        result = await limiter.check_request()
        assert result is True

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimit(rpm=2, window_seconds=60)

        await limiter.check_request()
        await limiter.check_request()

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.check_request()

        assert exc_info.value.limit == 2

    @pytest.mark.asyncio
    async def test_token_limit(self):
        """Test token-based rate limiting."""
        limiter = RateLimit(tpm=100, window_seconds=60)

        await limiter.check_tokens(50)
        await limiter.check_tokens(30)

        with pytest.raises(RateLimitError):
            await limiter.check_tokens(30)

    @pytest.mark.asyncio
    async def test_per_user_limiting(self):
        """Test per-user rate limiting."""
        limiter = RateLimit(rpm=2, by="user", window_seconds=60)

        # User 1 makes 2 requests
        await limiter.check_request("user1")
        await limiter.check_request("user1")

        # User 1 is blocked
        with pytest.raises(RateLimitError):
            await limiter.check_request("user1")

        # User 2 can still make requests
        result = await limiter.check_request("user2")
        assert result is True

    def test_get_usage(self):
        """Test getting usage statistics."""
        limiter = RateLimit(rpm=10, tpm=1000, window_seconds=60)

        usage = limiter.get_usage()

        assert usage["requests"] == 0
        assert usage["requests_limit"] == 10
        assert usage["tokens"] == 0
        assert usage["tokens_limit"] == 1000

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting rate limit counters."""
        limiter = RateLimit(rpm=2, window_seconds=60)

        await limiter.check_request()
        await limiter.check_request()

        limiter.reset()

        # Should be able to make requests again
        result = await limiter.check_request()
        assert result is True
