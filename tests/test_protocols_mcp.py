"""Tests for MCP protocol endpoints."""

from fastapi.testclient import TestClient

from fastagentic import App
from fastagentic.protocols.mcp import MCP_VERSION, configure_mcp


class TestMCPDiscoveryEndpoint:
    """Tests for MCP discovery endpoint."""

    def test_discovery_returns_protocol_version(self):
        """Test discovery endpoint returns correct protocol version."""
        app = App(title="Test App", version="1.0.0")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/discovery")

        assert response.status_code == 200
        data = response.json()
        assert data["protocolVersion"] == MCP_VERSION

    def test_discovery_returns_server_info(self):
        """Test discovery endpoint returns server info."""
        app = App(title="Test Server", version="2.0.0")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/discovery")

        assert response.status_code == 200
        data = response.json()
        assert data["serverInfo"]["name"] == "Test Server"
        assert data["serverInfo"]["version"] == "2.0.0"

    def test_discovery_returns_capabilities(self):
        """Test discovery endpoint returns capabilities."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/discovery")

        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert data["capabilities"]["tools"] is True
        assert data["capabilities"]["resources"] is True
        assert data["capabilities"]["prompts"] is True

    def test_discovery_with_custom_capabilities(self):
        """Test discovery with custom capabilities override."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp", capabilities={"tools": False})

        client = TestClient(app.fastapi)
        response = client.get("/mcp/discovery")

        assert response.status_code == 200
        data = response.json()
        assert data["capabilities"]["tools"] is False

    def test_discovery_when_disabled(self):
        """Test discovery returns 404 when MCP is disabled."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp", enabled=False)

        client = TestClient(app.fastapi)
        response = client.get("/mcp/discovery")

        # When disabled, the endpoint is not registered
        assert response.status_code == 404


class TestMCPToolsEndpoint:
    """Tests for MCP tools endpoints."""

    def test_list_tools_returns_empty(self):
        """Test listing tools returns empty list when no tools registered."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_call_tool_not_found(self):
        """Test calling a non-existent tool returns 404."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.post("/mcp/tools/nonexistent_tool", json={"arguments": {}})

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_call_tool_invalid_name(self):
        """Test calling tool with invalid name returns 422."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.post("/mcp/tools/invalid@name!", json={"arguments": {}})

        assert response.status_code == 422


class TestMCPResourcesEndpoint:
    """Tests for MCP resources endpoints."""

    def test_list_resources_returns_structure(self):
        """Test listing resources returns proper structure."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/resources")

        assert response.status_code == 200
        data = response.json()
        assert "resources" in data
        assert isinstance(data["resources"], list)

    def test_read_resource_not_found(self):
        """Test reading non-existent resource returns 404."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/resources/nonexistent")

        assert response.status_code == 404


class TestMCPPromptsEndpoint:
    """Tests for MCP prompts endpoints."""

    def test_list_prompts_returns_structure(self):
        """Test listing prompts returns proper structure."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.get("/mcp/prompts")

        assert response.status_code == 200
        data = response.json()
        assert "prompts" in data
        assert isinstance(data["prompts"], list)

    def test_get_prompt_not_found(self):
        """Test getting non-existent prompt returns 404."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)
        response = client.post("/mcp/prompts/nonexistent", json={"arguments": {}})

        assert response.status_code == 404


class TestMCPEndpointIntegration:
    """Integration tests for MCP endpoints."""

    def test_discovery_and_listing(self):
        """Test that discovery and listing endpoints work."""
        app = App(title="Test App")
        configure_mcp(app, path_prefix="/mcp")

        client = TestClient(app.fastapi)

        # Check discovery
        discovery_resp = client.get("/mcp/discovery")
        assert discovery_resp.status_code == 200

        # Check tools listing
        tools_resp = client.get("/mcp/tools")
        assert tools_resp.status_code == 200
        assert "tools" in tools_resp.json()

        # Check resources listing
        resources_resp = client.get("/mcp/resources")
        assert resources_resp.status_code == 200
        assert "resources" in resources_resp.json()

        # Check prompts listing
        prompts_resp = client.get("/mcp/prompts")
        assert prompts_resp.status_code == 200
        assert "prompts" in prompts_resp.json()

    def test_mcp_version_is_valid(self):
        """Test that MCP version is a valid format."""
        assert MCP_VERSION == "2025-11-25"
        # Version should follow YYYY-MM-DD format
        parts = MCP_VERSION.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # Year
        assert len(parts[1]) == 2  # Month
        assert len(parts[2]) == 2  # Day
