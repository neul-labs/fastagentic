"""Fuzzy and concurrency tests for reliability patterns.

These tests cover:
- Rate limiting under concurrent load
- Circuit breaker timing edge cases
- Race conditions in state management
"""

import asyncio

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from fastagentic.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from fastagentic.reliability.rate_limit import RateLimit, RateLimitError


class TestRateLimitConcurrency:
    """Concurrency tests for RateLimit."""

    @pytest.fixture
    def rate_limit(self):
        return RateLimit(rpm=10, tpm=None, window_seconds=60)

    @pytest.mark.asyncio
    async def test_concurrent_requests_at_limit(self, rate_limit):
        """Multiple concurrent requests at limit boundary.

        This tests race conditions in the rate limit check.
        """

        # Make concurrent requests up to the limit
        async def make_request():
            try:
                await rate_limit.check_request("user-1")
                return True
            except RateLimitError:
                return False

        results = await asyncio.gather(*[make_request() for _ in range(10)])

        # All 10 should succeed
        assert all(results), f"Not all requests succeeded: {results}"

        # Next request should fail
        with pytest.raises(RateLimitError):
            await rate_limit.check_request("user-1")

    @pytest.mark.asyncio
    async def test_concurrent_burst_over_limit(self, rate_limit):
        """Burst of requests exceeding limit concurrently."""

        async def make_request():
            try:
                await rate_limit.check_request("user-burst")
                return True
            except RateLimitError:
                return False

        # Try 20 concurrent requests with limit of 10
        results = await asyncio.gather(*[make_request() for _ in range(20)])

        allowed = sum(1 for r in results if r)
        denied = sum(1 for r in results if not r)

        # Should allow exactly 10
        assert allowed == 10, f"Expected 10 allowed, got {allowed}"
        assert denied == 10, f"Expected 10 denied, got {denied}"

    @pytest.mark.asyncio
    async def test_different_users_independent(self):
        """Different users should have independent limits when by='user'."""
        # Create a rate limiter with per-user scoping
        rate_limit = RateLimit(rpm=10, tpm=None, window_seconds=60, by="user")

        # User 1 uses up their limit
        for _ in range(10):
            await rate_limit.check_request("user-a")

        # User 2 should still be able to make requests
        async def user_b_request():
            try:
                await rate_limit.check_request("user-b")
                return True
            except RateLimitError:
                return False

        results = await asyncio.gather(*[user_b_request() for _ in range(5)])

        assert all(results), "User B blocked by User A's rate limit"

    @pytest.mark.asyncio
    async def test_global_rate_limit_shared(self):
        """Global rate limit should be shared across all users."""
        rate_limit = RateLimit(rpm=10, tpm=None, window_seconds=60, by="global")

        # Multiple users share the same limit
        allowed = 0
        for i in range(15):
            try:
                await rate_limit.check_request(f"user-{i}")
                allowed += 1
            except RateLimitError:
                pass

        assert allowed == 10, f"Expected 10 global, got {allowed}"

    @pytest.mark.asyncio
    async def test_cleanup_during_concurrent_access(self):
        """Test cleanup doesn't race with concurrent checks."""
        rate_limit = RateLimit(rpm=100, tpm=None, window_seconds=1)  # Short window

        # Add some requests
        for _ in range(50):
            await rate_limit.check_request("user-cleanup")

        # Wait for window to pass
        await asyncio.sleep(1.1)

        # Now run concurrent checks - old entries should be cleaned
        async def check_request():
            try:
                await rate_limit.check_request("user-cleanup")
                return True
            except RateLimitError:
                return False

        results = await asyncio.gather(*[check_request() for _ in range(50)])

        # Old entries should be cleaned, new ones allowed
        allowed = sum(1 for r in results if r)
        assert allowed == 50, f"Cleanup race: only {allowed}/50 allowed"


class TestRateLimitBoundaryConditions:
    """Boundary condition tests for rate limiting."""

    @given(
        rpm=st.integers(min_value=1, max_value=100),
        request_count=st.integers(min_value=1, max_value=50),
    )
    @pytest.mark.asyncio
    async def test_exact_limit_boundary(self, rpm: int, request_count: int):
        """Test behavior at exact limit boundary."""
        assume(request_count <= rpm + 10)  # Keep test reasonable

        rate_limit = RateLimit(rpm=rpm, tpm=None, window_seconds=60)

        allowed = 0
        for _ in range(request_count):
            try:
                await rate_limit.check_request("test-user")
                allowed += 1
            except RateLimitError:
                pass

        if request_count <= rpm:
            assert allowed == request_count
        else:
            assert allowed == rpm

    @pytest.mark.asyncio
    async def test_none_identifier_handling(self):
        """None identifier should fall back to 'unknown'."""
        rate_limit = RateLimit(rpm=5, tpm=None, window_seconds=60, by="user")

        # Multiple None identifiers share same bucket
        for _ in range(5):
            await rate_limit.check_request(None)

        with pytest.raises(RateLimitError):
            await rate_limit.check_request(None)

    @pytest.mark.asyncio
    async def test_empty_string_identifier(self):
        """Empty string identifier should be handled."""
        rate_limit = RateLimit(rpm=5, tpm=None, window_seconds=60, by="user")

        for _ in range(5):
            await rate_limit.check_request("")

        with pytest.raises(RateLimitError):
            await rate_limit.check_request("")


class TestCircuitBreakerTiming:
    """Timing edge case tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            failure_threshold=3,
            reset_timeout_ms=1000,  # 1 second
            failure_window_ms=5000,  # 5 seconds
            half_open_requests=2,
        )

    @pytest.mark.asyncio
    async def test_opens_at_threshold(self, breaker):
        """Circuit should open at exactly failure_threshold.

        Mutation: >= changed to > would require 4 failures
        """

        async def failing_func():
            raise RuntimeError("Simulated failure")

        # Record exactly threshold failures
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN, "Circuit didn't open at threshold"

    @pytest.mark.asyncio
    async def test_below_threshold_stays_closed(self, breaker):
        """Circuit should stay closed below threshold.

        Mutation: >= changed to > would open at 2
        """

        async def failing_func():
            raise RuntimeError("Simulated failure")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.CLOSED, "Circuit opened below threshold"

    @pytest.mark.asyncio
    async def test_reset_after_timeout(self, breaker):
        """Circuit should transition to half-open after reset_timeout."""

        async def failing_func():
            raise RuntimeError("Simulated failure")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for the timeout
        await asyncio.sleep(1.1)

        # Next attempt should succeed (transitions to half-open, then success)
        result = await breaker.execute(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_just_before_timeout_stays_open(self, breaker):
        """Circuit should stay open just before timeout."""

        async def failing_func():
            raise RuntimeError("Simulated failure")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        # Wait just under the timeout
        await asyncio.sleep(0.5)

        # Should still be blocked
        with pytest.raises(CircuitOpenError):
            await breaker.execute(failing_func)

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self, breaker):
        """Enough successes in half-open should close circuit."""

        async def failing_func():
            raise RuntimeError("Simulated failure")

        async def success_func():
            return "success"

        # Open circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        # Wait for reset
        await asyncio.sleep(1.1)

        # Record successes (need half_open_requests = 2 successes)
        for _ in range(2):
            await breaker.execute(success_func)

        assert breaker.state == CircuitState.CLOSED, (
            "Circuit didn't close after half-open successes"
        )

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker):
        """Failure in half-open should reopen circuit."""

        async def failing_func():
            raise RuntimeError("Simulated failure")

        # Open circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        # Wait for reset
        await asyncio.sleep(1.1)

        # Failure in half-open should reopen
        with pytest.raises(RuntimeError):
            await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN, "Circuit didn't reopen after half-open failure"


class TestCircuitBreakerFailureWindow:
    """Tests for failure window cleanup."""

    @pytest.mark.asyncio
    async def test_old_failures_cleaned(self):
        """Failures outside window should not count.

        Mutation: Cleanup logic errors
        """
        breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout_ms=1000,
            failure_window_ms=500,  # 0.5 second window
            half_open_requests=1,
        )

        async def failing_func():
            raise RuntimeError("Simulated failure")

        # Record 2 failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        # Wait for failures to age out
        await asyncio.sleep(0.6)

        # Record 1 more - should not open (old failures cleaned)
        with pytest.raises(RuntimeError):
            await breaker.execute(failing_func)

        assert breaker.state == CircuitState.CLOSED, "Old failures not cleaned from window"

    @pytest.mark.asyncio
    async def test_failures_within_window_counted(self):
        """All failures within window should count."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout_ms=1000,
            failure_window_ms=5000,  # 5 second window
            half_open_requests=1,
        )

        async def failing_func():
            raise RuntimeError("Simulated failure")

        # Quick succession of failures
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)
            await asyncio.sleep(0.1)

        assert breaker.state == CircuitState.OPEN, "Failures within window not counted correctly"


class TestCircuitBreakerConcurrency:
    """Concurrency tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_concurrent_failures(self):
        """Concurrent failures should be handled safely."""
        breaker = CircuitBreaker(
            failure_threshold=10,
            reset_timeout_ms=1000,
            failure_window_ms=5000,
            half_open_requests=1,
        )

        async def failing_func():
            raise RuntimeError("Simulated failure")

        async def try_execute():
            try:
                await breaker.execute(failing_func)
            except (RuntimeError, CircuitOpenError):
                pass

        # Record failures concurrently
        await asyncio.gather(*[try_execute() for _ in range(10)])

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_concurrent_success_checks(self):
        """Concurrent success executions should be safe."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout_ms=100,
            failure_window_ms=5000,
            half_open_requests=1,
        )

        async def success_func():
            return True

        # Multiple concurrent checks
        results = await asyncio.gather(*[breaker.execute(success_func) for _ in range(20)])

        # All should return True
        assert all(results)
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerEdgeCases:
    """Edge case tests for CircuitBreaker."""

    @given(
        threshold=st.integers(min_value=1, max_value=10),
        timeout=st.integers(min_value=100, max_value=1000),
    )
    @pytest.mark.asyncio
    async def test_parameter_combinations(self, threshold: int, timeout: int):
        """Various parameter combinations should work."""
        breaker = CircuitBreaker(
            failure_threshold=threshold,
            reset_timeout_ms=timeout,
            failure_window_ms=timeout * 5,
            half_open_requests=1,
        )

        # Should start closed
        assert breaker.state == CircuitState.CLOSED

        async def failing_func():
            raise RuntimeError("Simulated failure")

        # Should open after threshold failures
        for _ in range(threshold):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_in_closed_state(self):
        """Success in closed state should not change state."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout_ms=1000,
            failure_window_ms=5000,
            half_open_requests=1,
        )

        async def success_func():
            return "success"

        result = await breaker.execute(success_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_state_transitions(self):
        """Test multiple full state transition cycles."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            reset_timeout_ms=100,
            failure_window_ms=5000,
            half_open_requests=1,
        )

        async def failing_func():
            raise RuntimeError("Simulated failure")

        async def success_func():
            return "success"

        for cycle in range(3):
            # Closed -> Open
            for _ in range(2):
                with pytest.raises(RuntimeError):
                    await breaker.execute(failing_func)
            assert breaker.state == CircuitState.OPEN, f"Cycle {cycle}: Not opened"

            # Open -> Half-Open -> Closed
            await asyncio.sleep(0.15)
            await breaker.execute(success_func)
            assert breaker.state == CircuitState.CLOSED, f"Cycle {cycle}: Not closed"

    @pytest.mark.asyncio
    async def test_reset_method(self):
        """Manual reset should work."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            reset_timeout_ms=10000,
            failure_window_ms=5000,
            half_open_requests=1,
        )

        async def failing_func():
            raise RuntimeError("Simulated failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_trip_method(self):
        """Manual trip should open circuit."""
        breaker = CircuitBreaker(
            failure_threshold=10,
            reset_timeout_ms=1000,
            failure_window_ms=5000,
            half_open_requests=1,
        )

        assert breaker.state == CircuitState.CLOSED

        # Manual trip
        breaker.trip()

        assert breaker.state == CircuitState.OPEN
