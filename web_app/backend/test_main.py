from fastapi.testclient import TestClient
from web_app.backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_execute_allowed_command():
    response = client.post("/api/execute", json={"command": "echo Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "stdout" in data
    assert "Hello" in data["stdout"]
    assert data["returncode"] == 0

def test_execute_forbidden_command():
    response = client.post("/api/execute", json={"command": "rm -rf /"})
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]

def test_execute_empty_command():
    response = client.post("/api/execute", json={"command": ""})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

    response = client.post("/api/execute", json={"command": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_execute_invalid_shlex():
    response = client.post("/api/execute", json={"command": 'echo "unclosed quote'})
    assert response.status_code == 400
    assert "Invalid command format" in response.json()["detail"]

from fastapi.websockets import WebSocketDisconnect
from unittest.mock import patch, AsyncMock
import pytest

def test_run_script_valid():
    with patch("web_app.backend.main.asyncio.create_task") as mock_create_task:
        response = client.post("/api/run_script", json={"script_type": "pretraining"})
        assert response.status_code == 200
        assert "Started pretraining script" in response.json()["message"]
        mock_create_task.assert_called_once()

def test_run_script_invalid():
    response = client.post("/api/run_script", json={"script_type": "invalid_script"})
    assert response.status_code == 400
    assert "Unknown script type" in response.json()["detail"]

def test_websocket_logs():
    with client.websocket_connect("/ws/logs") as websocket:
        # Simulate broadcasting a message internally
        import asyncio
        from web_app.backend.main import manager

        # We need to run this in an event loop because broadcast is async
        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager.broadcast("Test log message"))

        data = websocket.receive_text()
        assert data == "Test log message"
