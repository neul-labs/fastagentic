"""FastAgentic SDK client."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from fastagentic.sdk.exceptions import (
    ConnectionError,
    FastAgenticError,
    StreamError,
    raise_for_status,
)
from fastagentic.sdk.models import RunRequest, RunResponse, StreamEvent, ToolResult


@dataclass
class ClientConfig:
    """Client configuration.

    Attributes:
        base_url: Base URL of the FastAgentic server
        api_key: API key for authentication
        timeout: Default request timeout in seconds
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries in seconds
        headers: Additional headers to include
        verify_ssl: Whether to verify SSL certificates
    """

    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True

    def get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class AsyncFastAgenticClient:
    """Async client for FastAgentic services.

    Example:
        async with AsyncFastAgenticClient(
            base_url="http://localhost:8000",
            api_key="your-api-key",
        ) as client:
            # Run an endpoint
            response = await client.run(
                endpoint="/triage",
                input={"ticket": "..."},
            )
            print(response.output)

            # Stream a response
            async for event in client.stream(
                endpoint="/chat",
                input={"message": "Hello"},
            ):
                if event.type == StreamEventType.TOKEN:
                    print(event.data, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        config: ClientConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize async client.

        Args:
            base_url: Base URL of the FastAgentic server
            api_key: API key for authentication
            config: Full client configuration
            **kwargs: Additional config options
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for the SDK client. Install with: pip install httpx"
            )

        if config:
            self.config = config
        else:
            self.config = ClientConfig(
                base_url=base_url,
                api_key=api_key,
                **kwargs,
            )

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AsyncFastAgenticClient:
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self.config.get_headers(),
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        return self._client

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic."""
        client = await self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(method, path, **kwargs)

                if response.status_code >= 400:
                    try:
                        data = response.json()
                    except (ValueError, httpx.DecodingError):
                        data = {"error": response.text}
                    raise_for_status(response.status_code, data)

                return response.json()

            except httpx.ConnectError as e:
                last_error = ConnectionError(f"Failed to connect: {e}")
            except httpx.TimeoutException as e:
                last_error = FastAgenticError(f"Request timed out: {e}")
            except FastAgenticError:
                raise
            except Exception as e:
                last_error = FastAgenticError(f"Request failed: {e}")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise last_error or FastAgenticError("Request failed")

    async def run(
        self,
        endpoint: str,
        input: dict[str, Any],
        *,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunResponse:
        """Run an endpoint and wait for the result.

        Args:
            endpoint: Endpoint path
            input: Input data
            timeout: Request timeout
            metadata: Additional metadata

        Returns:
            RunResponse with the result
        """
        request = RunRequest(
            endpoint=endpoint,
            input=input,
            stream=False,
            timeout=timeout,
            metadata=metadata or {},
        )

        data = await self._request(
            "POST",
            endpoint,
            json=request.to_dict(),
            timeout=timeout or self.config.timeout,
        )

        return RunResponse.from_dict(data)

    async def stream(
        self,
        endpoint: str,
        input: dict[str, Any],
        *,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream an endpoint response.

        Args:
            endpoint: Endpoint path
            input: Input data
            timeout: Request timeout
            metadata: Additional metadata

        Yields:
            StreamEvent for each event
        """
        client = await self._ensure_client()

        request = RunRequest(
            endpoint=endpoint,
            input=input,
            stream=True,
            timeout=timeout,
            metadata=metadata or {},
        )

        try:
            async with client.stream(
                "POST",
                endpoint,
                json=request.to_dict(),
                headers={"Accept": "text/event-stream"},
                timeout=timeout or self.config.timeout,
            ) as response:
                if response.status_code >= 400:
                    content = await response.aread()
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        data = {"error": content.decode()}
                    raise_for_status(response.status_code, data)

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            yield StreamEvent.from_dict(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Invalid stream data: {e}")

        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise FastAgenticError(f"Stream timed out: {e}")

    async def get_run(self, run_id: str) -> RunResponse:
        """Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            RunResponse
        """
        data = await self._request("GET", f"/runs/{run_id}")
        return RunResponse.from_dict(data)

    async def cancel_run(self, run_id: str) -> RunResponse:
        """Cancel a run.

        Args:
            run_id: Run ID

        Returns:
            RunResponse with updated status
        """
        data = await self._request("POST", f"/runs/{run_id}/cancel")
        return RunResponse.from_dict(data)

    async def wait_for_run(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> RunResponse:
        """Wait for a run to complete.

        Args:
            run_id: Run ID
            timeout: Maximum wait time
            poll_interval: Time between polls

        Returns:
            RunResponse when complete
        """
        import random

        deadline = time.time() + timeout if timeout else None
        current_interval = poll_interval
        attempt = 0

        while True:
            response = await self.get_run(run_id)
            if response.is_complete:
                return response

            if deadline and time.time() > deadline:
                raise FastAgenticError(f"Timeout waiting for run {run_id}")

            # Exponential backoff with jitter
            await asyncio.sleep(current_interval)
            attempt += 1
            # Add jitter (±25%) and cap at 10 seconds
            jitter = current_interval * 0.25 * random.uniform(-1, 1)
            current_interval = min(current_interval * 1.5 + jitter, 10.0)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools.

        Returns:
            List of tool definitions
        """
        data = await self._request("GET", "/tools")
        return data.get("tools", [])

    async def invoke_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke a tool directly.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        data = await self._request(
            "POST",
            f"/tools/{name}",
            json={"arguments": arguments},
        )
        return ToolResult.from_dict(data)

    async def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Health status
        """
        return await self._request("GET", "/health")

    async def info(self) -> dict[str, Any]:
        """Get server info.

        Returns:
            Server information
        """
        return await self._request("GET", "/info")


class FastAgenticClient:
    """Sync client for FastAgentic services.

    Example:
        with FastAgenticClient(
            base_url="http://localhost:8000",
            api_key="your-api-key",
        ) as client:
            # Run an endpoint
            response = client.run(
                endpoint="/triage",
                input={"ticket": "..."},
            )
            print(response.output)

            # Stream a response
            for event in client.stream(
                endpoint="/chat",
                input={"message": "Hello"},
            ):
                if event.type == StreamEventType.TOKEN:
                    print(event.data, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        config: ClientConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize sync client.

        Args:
            base_url: Base URL of the FastAgentic server
            api_key: API key for authentication
            config: Full client configuration
            **kwargs: Additional config options
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for the SDK client. Install with: pip install httpx"
            )

        if config:
            self.config = config
        else:
            self.config = ClientConfig(
                base_url=base_url,
                api_key=api_key,
                **kwargs,
            )

        self._client: httpx.Client | None = None

    def __enter__(self) -> FastAgenticClient:
        """Enter context."""
        self._ensure_client()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context."""
        self.close()

    def _ensure_client(self) -> httpx.Client:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=self.config.get_headers(),
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        return self._client

    def close(self) -> None:
        """Close the client."""
        if self._client:
            self._client.close()
            self._client = None

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic."""
        client = self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = client.request(method, path, **kwargs)

                if response.status_code >= 400:
                    try:
                        data = response.json()
                    except (ValueError, httpx.DecodingError):
                        data = {"error": response.text}
                    raise_for_status(response.status_code, data)

                return response.json()

            except httpx.ConnectError as e:
                last_error = ConnectionError(f"Failed to connect: {e}")
            except httpx.TimeoutException as e:
                last_error = FastAgenticError(f"Request timed out: {e}")
            except FastAgenticError:
                raise
            except Exception as e:
                last_error = FastAgenticError(f"Request failed: {e}")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2**attempt)
                time.sleep(delay)

        raise last_error or FastAgenticError("Request failed")

    def run(
        self,
        endpoint: str,
        input: dict[str, Any],
        *,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunResponse:
        """Run an endpoint and wait for the result.

        Args:
            endpoint: Endpoint path
            input: Input data
            timeout: Request timeout
            metadata: Additional metadata

        Returns:
            RunResponse with the result
        """
        request = RunRequest(
            endpoint=endpoint,
            input=input,
            stream=False,
            timeout=timeout,
            metadata=metadata or {},
        )

        data = self._request(
            "POST",
            endpoint,
            json=request.to_dict(),
            timeout=timeout or self.config.timeout,
        )

        return RunResponse.from_dict(data)

    def stream(
        self,
        endpoint: str,
        input: dict[str, Any],
        *,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream an endpoint response.

        Args:
            endpoint: Endpoint path
            input: Input data
            timeout: Request timeout
            metadata: Additional metadata

        Yields:
            StreamEvent for each event
        """
        client = self._ensure_client()

        request = RunRequest(
            endpoint=endpoint,
            input=input,
            stream=True,
            timeout=timeout,
            metadata=metadata or {},
        )

        try:
            with client.stream(
                "POST",
                endpoint,
                json=request.to_dict(),
                headers={"Accept": "text/event-stream"},
                timeout=timeout or self.config.timeout,
            ) as response:
                if response.status_code >= 400:
                    content = response.read()
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        data = {"error": content.decode()}
                    raise_for_status(response.status_code, data)

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            yield StreamEvent.from_dict(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Invalid stream data: {e}")

        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise FastAgenticError(f"Stream timed out: {e}")

    def get_run(self, run_id: str) -> RunResponse:
        """Get a run by ID."""
        data = self._request("GET", f"/runs/{run_id}")
        return RunResponse.from_dict(data)

    def cancel_run(self, run_id: str) -> RunResponse:
        """Cancel a run."""
        data = self._request("POST", f"/runs/{run_id}/cancel")
        return RunResponse.from_dict(data)

    def wait_for_run(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> RunResponse:
        """Wait for a run to complete."""
        import random

        deadline = time.time() + timeout if timeout else None
        current_interval = poll_interval
        attempt = 0

        while True:
            response = self.get_run(run_id)
            if response.is_complete:
                return response

            if deadline and time.time() > deadline:
                raise FastAgenticError(f"Timeout waiting for run {run_id}")

            # Exponential backoff with jitter
            time.sleep(current_interval)
            attempt += 1
            # Add jitter (±25%) and cap at 10 seconds
            jitter = current_interval * 0.25 * random.uniform(-1, 1)
            current_interval = min(current_interval * 1.5 + jitter, 10.0)

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools."""
        data = self._request("GET", "/tools")
        return data.get("tools", [])

    def invoke_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke a tool directly."""
        data = self._request(
            "POST",
            f"/tools/{name}",
            json={"arguments": arguments},
        )
        return ToolResult.from_dict(data)

    def health(self) -> dict[str, Any]:
        """Check server health."""
        return self._request("GET", "/health")

    def info(self) -> dict[str, Any]:
        """Get server info."""
        return self._request("GET", "/info")
