"""Tests for A2A protocol."""

import pytest
from fastapi.testclient import TestClient

from fastagentic import App
from fastagentic.decorators import agent_endpoint
from fastagentic.protocols.a2a import (
    A2A_VERSION,
    A2ATask,
    InMemoryTaskStore,
    TaskStatus,
    configure_a2a,
)


class TestA2ATaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestA2ATask:
    """Tests for A2ATask dataclass."""

    def test_create_task(self):
        """Test creating an A2A task."""
        task = A2ATask(task_id="test-123", skill="test_skill", input={"message": "hello"})
        assert task.task_id == "test-123"
        assert task.skill == "test_skill"
        assert task.input["message"] == "hello"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.cancelled is False

    def test_task_with_custom_status(self):
        """Test task with custom initial status."""
        task = A2ATask(task_id="test-456", skill="skill2", input={}, status=TaskStatus.RUNNING)
        assert task.status == TaskStatus.RUNNING

    def test_task_with_result(self):
        """Test task with result."""
        task = A2ATask(
            task_id="test-789",
            skill="skill3",
            input={},
            status=TaskStatus.COMPLETED,
            result={"output": "success"},
        )
        assert task.result == {"output": "success"}


class TestInMemoryTaskStore:
    """Tests for InMemoryTaskStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh task store."""
        return InMemoryTaskStore()

    @pytest.mark.asyncio
    async def test_create_task(self, store):
        """Test creating a task."""
        task = A2ATask(task_id="create-1", skill="test_skill", input={"key": "value"})
        await store.create(task)

        retrieved = await store.get("create-1")
        assert retrieved is not None
        assert retrieved.task_id == "create-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting non-existent task."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_task(self, store):
        """Test updating a task."""
        task = A2ATask(task_id="update-1", skill="test", input={})
        await store.create(task)

        task.status = TaskStatus.RUNNING
        task.result = {"updated": True}
        await store.update(task)

        retrieved = await store.get("update-1")
        assert retrieved.status == TaskStatus.RUNNING
        assert retrieved.result == {"updated": True}

    @pytest.mark.asyncio
    async def test_cancel_pending_task(self, store):
        """Test cancelling a pending task."""
        task = A2ATask(task_id="cancel-1", skill="test", input={})
        await store.create(task)

        result = await store.cancel("cancel-1")
        assert result is True

        retrieved = await store.get("cancel-1")
        assert retrieved.status == TaskStatus.CANCELLED
        assert retrieved.cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_completed_task_fails(self, store):
        """Test cancelling a completed task fails."""
        task = A2ATask(
            task_id="cancel-2",
            skill="test",
            input={},
            status=TaskStatus.COMPLETED,
            result={"done": True},
        )
        await store.create(task)

        result = await store.cancel("cancel-2")
        assert result is False

        retrieved = await store.get("cancel-2")
        assert retrieved.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_list_by_skill(self, store):
        """Test listing tasks by skill."""
        await store.create(A2ATask(task_id="1", skill="skill_a", input={}))
        await store.create(A2ATask(task_id="2", skill="skill_b", input={}))
        await store.create(A2ATask(task_id="3", skill="skill_a", input={}))

        skill_a_tasks = await store.list_by_skill("skill_a")
        assert len(skill_a_tasks) == 2

        skill_b_tasks = await store.list_by_skill("skill_b")
        assert len(skill_b_tasks) == 1

    @pytest.mark.asyncio
    async def test_delete_task(self, store):
        """Test deleting a task."""
        task = A2ATask(task_id="delete-1", skill="test", input={})
        await store.create(task)

        result = await store.delete("delete-1")
        assert result is True

        retrieved = await store.get("delete-1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting non-existent task returns False."""
        result = await store.delete("nonexistent")
        assert result is False


class TestA2APingEndpoint:
    """Tests for A2A ping endpoint."""

    def test_ping_returns_ok(self):
        """Test ping endpoint returns OK."""
        app = App(title="Test Agent")
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.get("/a2a/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == A2A_VERSION
        assert data["agent"] == "Test Agent"

    def test_ping_when_disabled(self):
        """Test ping returns 404 when A2A is disabled."""
        app = App(title="Test Agent")
        configure_a2a(app, enabled=False)

        client = TestClient(app.fastapi)
        response = client.get("/a2a/ping")
        assert response.status_code == 404


class TestA2AAgentsEndpoint:
    """Tests for A2A agents listing endpoint."""

    def test_list_agents(self):
        """Test listing agents."""
        app = App(
            title="My Agent",
            version="1.0.0",
        )
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.get("/a2a/agents")

        assert response.status_code == 200
        data = response.json()
        agents = data["agents"]
        assert len(agents) == 1
        assert agents[0]["name"] == "My Agent"
        assert agents[0]["version"] == "1.0.0"
        assert "a2a/v" in agents[0]["protocols"][0]


class TestA2ATasksEndpoints:
    """Tests for A2A task endpoints."""

    def test_get_nonexistent_task(self):
        """Test getting a non-existent task returns 404."""
        app = App(title="Test Agent")
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.get("/a2a/tasks/nonexistent-id")

        assert response.status_code == 404

    def test_cancel_nonexistent_task(self):
        """Test cancelling non-existent task returns 404."""
        app = App(title="Test Agent")
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.delete("/a2a/tasks/nonexistent-id")

        assert response.status_code == 404

    def test_create_task_missing_skill(self):
        """Test creating task without skill returns 400."""
        app = App(title="Test Agent")
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.post("/a2a/tasks", json={"input": {"message": "hello"}})

        assert response.status_code == 400
        assert "skill" in response.json()["error"].lower()


class TestA2AWithEndpoints:
    """Tests for A2A with registered agent endpoints."""

    def test_create_task_with_skill(self):
        """Test creating task with registered skill."""

        @agent_endpoint(path="/test-endpoint", name="test_endpoint", a2a_skill="my_test_skill")
        async def my_handler(input: dict) -> dict:
            return {"result": "handled"}

        app = App(title="Test Agent")
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.post(
            "/a2a/tasks", json={"skill": "my_test_skill", "input": {"message": "test"}}
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["skill"] == "my_test_skill"
        assert data["status"] == "pending"

    def test_create_task_skill_not_found(self):
        """Test creating task with unknown skill returns 404."""

        @agent_endpoint(path="/handler", name="test_endpoint", a2a_skill="known_skill")
        async def handler(input: dict) -> dict:
            return {}

        app = App(title="Test Agent")
        configure_a2a(app)

        client = TestClient(app.fastapi)
        response = client.post("/a2a/tasks", json={"skill": "unknown_skill", "input": {}})

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"].lower()
        assert "known_skill" in data.get("available_skills", [])


class TestA2AProtocolVersion:
    """Tests for A2A protocol version."""

    def test_version_format(self):
        """Test version follows expected format."""
        # Version should be something like "0.3"
        assert "." in A2A_VERSION
        parts = A2A_VERSION.split(".")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)
