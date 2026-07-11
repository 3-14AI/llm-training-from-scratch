from fastapi.testclient import TestClient
from web_app.backend.main import app

client = TestClient(app)

ADMIN_HEADERS = {"X-API-Key": "admin_secret"}
VIEWER_HEADERS = {"X-API-Key": "viewer_secret"}

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_execute_allowed_command():
    response = client.post("/api/execute", json={"command": "echo Hello"}, headers=ADMIN_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "stdout" in data
    assert "Hello" in data["stdout"]
    assert data["returncode"] == 0

def test_execute_forbidden_command():
    response = client.post("/api/execute", json={"command": "rm -rf /"}, headers=ADMIN_HEADERS)
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]

def test_execute_empty_command():
    response = client.post("/api/execute", json={"command": ""}, headers=ADMIN_HEADERS)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

    response = client.post("/api/execute", json={"command": "   "}, headers=ADMIN_HEADERS)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_execute_invalid_shlex():
    response = client.post("/api/execute", json={"command": 'echo "unclosed quote'}, headers=ADMIN_HEADERS)
    assert response.status_code == 400
    assert "Invalid command format" in response.json()["detail"]

from fastapi.websockets import WebSocketDisconnect
from unittest.mock import patch, AsyncMock
import pytest

def test_run_script_valid():
    with patch("web_app.backend.main.asyncio.create_task") as mock_create_task:
        response = client.post("/api/run_script", json={"script_type": "pretraining"}, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "Started pretraining script" in response.json()["message"]
        mock_create_task.assert_called_once()

def test_run_script_invalid():
    response = client.post("/api/run_script", json={"script_type": "invalid_script"}, headers=ADMIN_HEADERS)
    assert response.status_code == 400
    assert "Unknown script type" in response.json()["detail"]

def test_websocket_logs():
    with client.websocket_connect("/ws/logs?token=viewer_secret") as websocket:
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

    response = client.post("/api/save_config", json=payload, headers=ADMIN_HEADERS)
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

    response = client.post("/api/save_config", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 422 # Unprocessable Entity from Pydantic

def test_get_configs():
    response = client.get("/api/configs", headers=VIEWER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "file_configs" in data
    assert "experiment_configs" in data
    assert isinstance(data["file_configs"], list)
    assert isinstance(data["experiment_configs"], list)

def test_quick_launch_invalid():
    response = client.post("/api/quick_launch", json={"config_name": ""}, headers=ADMIN_HEADERS)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_quick_launch_valid_experiment():
    with patch("web_app.backend.main.asyncio.create_task") as mock_create_task:
        response = client.post("/api/quick_launch", json={"config_name": "s1_small_baseline"}, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "Quick launch started" in response.json()["message"]
        mock_create_task.assert_called_once()

def test_get_artifacts():
    response = client.get("/api/artifacts", headers=VIEWER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data
    assert isinstance(data["artifacts"], list)

@pytest.mark.asyncio
def test_inference_valid():
    # Patch asyncio.create_subprocess_exec directly to mock the subprocess behavior
    with patch("web_app.backend.main.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        # Define mock process behavior
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'{"status": "success", "prompt": "Hello", "generated_text": "Hello world"}', b'')
        mock_exec.return_value = mock_process

        payload = {
            "prompt": "Hello",
            "max_tokens": 10,
            "temperature": 0.5,
            "model_path": "dummy.pth"
        }

        # We need to run client.post directly, but the endpoint is async. FastAPI TestClient handles this automatically.
        response = client.post("/api/inference", json=payload, headers=VIEWER_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["generated_text"] == "Hello world"
        mock_exec.assert_called_once()

def test_inference_empty_prompt():
    payload = {
        "prompt": "",
        "max_tokens": 10
    }
    response = client.post("/api/inference", json=payload, headers=VIEWER_HEADERS)
    assert response.status_code == 400
    assert "Prompt cannot be empty" in response.json()["detail"]

@pytest.mark.asyncio
def test_inference_script_failure():
    with patch("web_app.backend.main.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b'', b'Error loading model')
        mock_exec.return_value = mock_process

        payload = {
            "prompt": "Test"
        }
        response = client.post("/api/inference", json=payload, headers=VIEWER_HEADERS)
        assert response.status_code == 500
        assert "Inference process failed" in response.json()["detail"]

@patch("web_app.backend.main.HfApi")
def test_export_model_valid_file(MockHfApi):
    mock_api_instance = MockHfApi.return_value

    # create dummy file
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "checkpoints_finetune")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "dummy_model.pth")
    with open(temp_path, "wb") as f:
        f.write(b"dummy")

    try:
        rel_path = os.path.relpath(temp_path, project_root)

        payload = {
            "model_path": rel_path,
            "hf_token": "dummy_token",
            "repo_id": "dummy_user/dummy_repo"
        }
        response = client.post("/api/export_model", json=payload, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "Successfully exported" in response.json()["message"]

        MockHfApi.assert_called_once_with(token="dummy_token")
        mock_api_instance.repo_info.assert_called_once_with(repo_id="dummy_user/dummy_repo")
        mock_api_instance.upload_file.assert_called_once()

    finally:
        os.remove(temp_path)

@patch("web_app.backend.main.HfApi")
def test_export_model_valid_dir(MockHfApi):
    mock_api_instance = MockHfApi.return_value

    # create dummy dir
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "checkpoints_finetune", "dummy_dir")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        rel_path = os.path.relpath(temp_dir, project_root)

        payload = {
            "model_path": rel_path,
            "hf_token": "dummy_token",
            "repo_id": "dummy_user/dummy_repo"
        }
        response = client.post("/api/export_model", json=payload, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "Successfully exported" in response.json()["message"]

        MockHfApi.assert_called_once_with(token="dummy_token")
        mock_api_instance.repo_info.assert_called_once_with(repo_id="dummy_user/dummy_repo")
        mock_api_instance.upload_folder.assert_called_once()

    finally:
        os.rmdir(temp_dir)

def test_export_model_path_traversal():
    payload = {
        "model_path": "../../etc/passwd",
        "hf_token": "dummy_token",
        "repo_id": "dummy_user/dummy_repo"
    }
    response = client.post("/api/export_model", json=payload, headers=ADMIN_HEADERS)
    response = client.post("/api/export_model", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 403
    assert "Path traversal is not allowed" in response.json()["detail"]

def test_export_model_not_found():
    payload = {
        "model_path": "non_existent_path",
        "hf_token": "dummy_token",
        "repo_id": "dummy_user/dummy_repo"
    }
    response = client.post("/api/export_model", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 404

from unittest.mock import mock_open

@patch("os.path.exists")
def test_get_metrics_no_file(mock_exists):
    # Если файла нет, возвращаются мок-данные
    mock_exists.return_value = False
    response = client.get("/api/metrics", headers=VIEWER_HEADERS)
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
    response = client.get("/api/metrics", headers=VIEWER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["epochs"] == [1, 2]
    assert data["loss"] == [3.0, 2.5]
    assert data["perplexity"] == [10.0, 8.0]

def test_quick_launch_valid_file():
    with patch("web_app.backend.main.asyncio.create_task") as mock_create_task:
        response = client.post("/api/quick_launch", json={"config_name": "some_config.toml"}, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "Quick launch started" in response.json()["message"]
        mock_create_task.assert_called_once()

def test_missing_auth():
    response = client.get("/api/configs")
    assert response.status_code == 401

    response = client.post("/api/execute", json={"command": "echo Hello"})
    assert response.status_code == 401

def test_invalid_auth():
    response = client.get("/api/configs", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401

def test_viewer_access_admin_endpoint():
    response = client.post("/api/execute", json={"command": "echo Hello"}, headers=VIEWER_HEADERS)
    assert response.status_code == 403
