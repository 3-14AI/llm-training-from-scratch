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

        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(manager.broadcast("Test log message"))
        finally:
            loop.close()

        data = websocket.receive_text()
        assert data == "Test log message"

import os

def test_save_config():
    test_config_name = "test_config_123"
    payload = {
        "config_name": test_config_name,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    }

    response = client.post("/api/save_config", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "saved successfully" in data["message"]

    config_path = data["path"]
    assert os.path.exists(config_path)

    with open(config_path, "r") as f:
        content = f.read()

    assert "[training]" in content
    assert "epochs = 10" in content
    assert "batch_size = 32" in content
    assert "learning_rate = 0.001" in content

    # Cleanup
    os.remove(config_path)


def test_save_config_invalid_name():
    payload = {
        "config_name": "../malicious",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    }

    response = client.post("/api/save_config", json=payload)
    assert response.status_code == 422 # Unprocessable Entity from Pydantic

def test_get_configs():
    response = client.get("/api/configs")
    assert response.status_code == 200
    data = response.json()
    assert "file_configs" in data
    assert "experiment_configs" in data
    assert isinstance(data["file_configs"], list)
    assert isinstance(data["experiment_configs"], list)

def test_quick_launch_invalid():
    response = client.post("/api/quick_launch", json={"config_name": ""})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_quick_launch_valid_experiment():
    with patch("web_app.backend.main.asyncio.create_task") as mock_create_task:
        response = client.post("/api/quick_launch", json={"config_name": "s1_small_baseline"})
        assert response.status_code == 200
        assert "Quick launch started" in response.json()["message"]
        mock_create_task.assert_called_once()

from unittest.mock import mock_open

@patch("os.path.exists")
def test_get_metrics_no_file(mock_exists):
    # Если файла нет, возвращаются мок-данные
    mock_exists.return_value = False
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "epochs" in data
    assert "loss" in data
    assert "perplexity" in data
    assert data["epochs"] == [1, 2, 3, 4, 5]

@patch("os.path.exists")
@patch("builtins.open", new_callable=mock_open, read_data='{"epochs": [1, 2], "loss": [3.0, 2.5], "perplexity": [10.0, 8.0]}')
def test_get_metrics_with_file(mock_file, mock_exists):
    # Если файл есть, данные считываются из него
    mock_exists.return_value = True
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["epochs"] == [1, 2]
    assert data["loss"] == [3.0, 2.5]
    assert data["perplexity"] == [10.0, 8.0]

def test_quick_launch_valid_file():
    with patch("web_app.backend.main.asyncio.create_task") as mock_create_task:
        response = client.post("/api/quick_launch", json={"config_name": "some_config.toml"})
        assert response.status_code == 200
        assert "Quick launch started" in response.json()["message"]
        mock_create_task.assert_called_once()
