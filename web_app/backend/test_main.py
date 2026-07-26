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
from unittest.mock import patch, AsyncMock, MagicMock
from web_app.backend.main import active_processes
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


def test_upload_artifact_valid():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "configs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "test_upload_file.txt")

    # Ensure it doesn't exist before test
    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        with open("dummy_local.txt", "wb") as f:
            f.write(b"uploaded content")

        with open("dummy_local.txt", "rb") as f:
            files = {"file": ("test_upload_file.txt", f, "text/plain")}
            data = {"directory": "configs"}
            response = client.post("/api/upload_artifact", data=data, files=files, headers=ADMIN_HEADERS)

        assert response.status_code == 200
        assert "uploaded successfully" in response.json()["message"]
        assert os.path.exists(temp_path)
        with open(temp_path, "rb") as f:
            assert f.read() == b"uploaded content"
    finally:
        if os.path.exists("dummy_local.txt"):
            os.remove("dummy_local.txt")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_upload_artifact_unauthorized():
    with open("dummy_local.txt", "wb") as f:
        f.write(b"content")

    try:
        with open("dummy_local.txt", "rb") as f:
            files = {"file": ("test.txt", f, "text/plain")}
            data = {"directory": "configs"}
            response = client.post("/api/upload_artifact", data=data, files=files, headers=VIEWER_HEADERS)
            assert response.status_code == 403
    finally:
        if os.path.exists("dummy_local.txt"):
            os.remove("dummy_local.txt")

def test_upload_artifact_invalid_directory():
    with open("dummy_local.txt", "wb") as f:
        f.write(b"content")

    try:
        with open("dummy_local.txt", "rb") as f:
            files = {"file": ("test.txt", f, "text/plain")}
            data = {"directory": "invalid_dir"}
            response = client.post("/api/upload_artifact", data=data, files=files, headers=ADMIN_HEADERS)
            assert response.status_code == 403
            assert "Cannot upload files outside of artifact directories" in response.json()["detail"]
    finally:
        if os.path.exists("dummy_local.txt"):
            os.remove("dummy_local.txt")

def test_delete_artifact_valid():
    # create dummy file
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "configs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "dummy_to_delete.txt")
    with open(temp_path, "wb") as f:
        f.write(b"dummy")

    try:
        rel_path = os.path.relpath(temp_path, project_root)
        payload = {"artifact_path": rel_path}

        response = client.post("/api/delete_artifact", json=payload, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert not os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_delete_artifact_unauthorized():
    payload = {"artifact_path": "configs/test.txt"}
    response = client.post("/api/delete_artifact", json=payload, headers=VIEWER_HEADERS)
    assert response.status_code == 403

def test_delete_artifact_path_traversal():
    payload = {"artifact_path": "../../etc/passwd"}
    response = client.post("/api/delete_artifact", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 403
    assert "Path traversal is not allowed" in response.json()["detail"]

def test_delete_artifact_outside_allowed_dirs():
    # create dummy file outside of artifact dirs but inside project
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "web_app")
    temp_path = os.path.join(temp_dir, "dummy_to_delete.txt")
    with open(temp_path, "wb") as f:
        f.write(b"dummy")

    try:
        rel_path = os.path.relpath(temp_path, project_root)
        payload = {"artifact_path": rel_path}

        response = client.post("/api/delete_artifact", json=payload, headers=ADMIN_HEADERS)
        assert response.status_code == 403
        assert "Cannot delete files outside of artifact directories" in response.json()["detail"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_delete_artifact_not_found():
    payload = {"artifact_path": "configs/non_existent_file.txt"}
    response = client.post("/api/delete_artifact", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 404
    assert "Artifact not found" in response.json()["detail"]


def test_download_artifact_valid():
    # create dummy file
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "configs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "dummy_to_download.txt")
    with open(temp_path, "wb") as f:
        f.write(b"dummy content")

    try:
        rel_path = os.path.relpath(temp_path, project_root)

        response = client.get(f"/api/download_artifact?artifact_path={rel_path}", headers=VIEWER_HEADERS)
        assert response.status_code == 200
        assert response.content == b"dummy content"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_download_artifact_unauthorized():
    response = client.get("/api/download_artifact?artifact_path=configs/test.txt")
    assert response.status_code == 401

def test_download_artifact_path_traversal():
    response = client.get("/api/download_artifact?artifact_path=../../etc/passwd", headers=VIEWER_HEADERS)
    assert response.status_code == 403
    assert "Path traversal is not allowed" in response.json()["detail"]

def test_download_artifact_outside_allowed_dirs():
    # create dummy file outside of artifact dirs but inside project
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "web_app")
    temp_path = os.path.join(temp_dir, "dummy_to_download.txt")
    with open(temp_path, "wb") as f:
        f.write(b"dummy")

    try:
        rel_path = os.path.relpath(temp_path, project_root)

        response = client.get(f"/api/download_artifact?artifact_path={rel_path}", headers=VIEWER_HEADERS)
        assert response.status_code == 403
        assert "Cannot access files outside of artifact directories" in response.json()["detail"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_download_artifact_not_found():
    response = client.get("/api/download_artifact?artifact_path=configs/non_existent_file.txt", headers=VIEWER_HEADERS)
    assert response.status_code == 404
    assert "Artifact not found" in response.json()["detail"]

def test_download_artifact_is_directory():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    temp_dir = os.path.join(project_root, "configs", "dummy_dir")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        rel_path = os.path.relpath(temp_dir, project_root)
        response = client.get(f"/api/download_artifact?artifact_path={rel_path}", headers=VIEWER_HEADERS)
        assert response.status_code == 400
        assert "Cannot download a directory" in response.json()["detail"]
    finally:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


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

def test_get_datasets_authorized():
    response = client.get("/api/datasets", headers=VIEWER_HEADERS)
    assert response.status_code == 200
    assert "datasets" in response.json()
    assert isinstance(response.json()["datasets"], list)

def test_get_datasets_unauthorized():
    response = client.get("/api/datasets")
    assert response.status_code == 401

def test_upload_dataset_authorized():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "data")

    # Ensure cleanup
    test_file_path = os.path.join(data_dir, "test_dataset.txt")
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

    try:
        response = client.post(
            "/api/upload_dataset",
            headers=ADMIN_HEADERS,
            files={"file": ("test_dataset.txt", b"dummy dataset content")}
        )
        assert response.status_code == 200
        assert "uploaded successfully" in response.json()["message"]
        assert os.path.exists(test_file_path)
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_upload_dataset_viewer_forbidden():
    response = client.post(
        "/api/upload_dataset",
        headers=VIEWER_HEADERS,
        files={"file": ("test_dataset.txt", b"dummy dataset content")}
    )
    assert response.status_code == 403

def test_upload_dataset_unauthorized():
    response = client.post(
        "/api/upload_dataset",
        files={"file": ("test_dataset.txt", b"dummy dataset content")}
    )
    assert response.status_code == 401

def test_delete_dataset_valid():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    temp_path = os.path.join(data_dir, "dummy_to_delete.txt")
    with open(temp_path, "wb") as f:
        f.write(b"dummy")

    try:
        rel_path = os.path.relpath(temp_path, project_root)
        payload = {"dataset_path": rel_path}

        response = client.post("/api/delete_dataset", json=payload, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        assert not os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_delete_dataset_unauthorized():
    payload = {"dataset_path": "data/test.txt"}
    response = client.post("/api/delete_dataset", json=payload, headers=VIEWER_HEADERS)
    assert response.status_code == 403

def test_delete_dataset_path_traversal():
    payload = {"dataset_path": "../../etc/passwd"}
    response = client.post("/api/delete_dataset", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 403
    assert "Path traversal is not allowed" in response.json()["detail"]

def test_delete_dataset_not_found():
    payload = {"dataset_path": "data/non_existent_file.txt"}
    response = client.post("/api/delete_dataset", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 404
    assert "Dataset not found" in response.json()["detail"]

def test_preview_dataset_valid():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    temp_path = os.path.join(data_dir, "dummy_to_preview.txt")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("line 1\nline 2\nline 3\n")

    try:
        rel_path = os.path.relpath(temp_path, project_root)
        response = client.get(f"/api/preview_dataset?dataset_path={rel_path}&lines=2", headers=VIEWER_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "preview" in data
        assert data["preview"] == "line 1\nline 2"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_preview_dataset_unauthorized():
    response = client.get("/api/preview_dataset?dataset_path=data/test.txt")
    assert response.status_code == 401

def test_preview_dataset_path_traversal():
    response = client.get("/api/preview_dataset?dataset_path=../../etc/passwd", headers=VIEWER_HEADERS)
    assert response.status_code == 403
    assert "Path traversal is not allowed" in response.json()["detail"]

def test_preview_dataset_not_found():
    response = client.get("/api/preview_dataset?dataset_path=data/non_existent_file.txt", headers=VIEWER_HEADERS)
    assert response.status_code == 404
    assert "Dataset not found" in response.json()["detail"]


def test_download_dataset_valid():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    temp_path = os.path.join(data_dir, "dummy_to_download.txt")
    with open(temp_path, "wb") as f:
        f.write(b"dummy content")

    try:
        rel_path = os.path.relpath(temp_path, project_root)

        response = client.get(f"/api/download_dataset?dataset_path={rel_path}", headers=VIEWER_HEADERS)
        assert response.status_code == 200
        assert response.content == b"dummy content"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_download_dataset_unauthorized():
    response = client.get("/api/download_dataset?dataset_path=data/test.txt")
    assert response.status_code == 401

def test_download_dataset_path_traversal():
    response = client.get("/api/download_dataset?dataset_path=../../etc/passwd", headers=VIEWER_HEADERS)
    assert response.status_code == 403
    assert "Path traversal is not allowed" in response.json()["detail"]

def test_download_dataset_not_found():
    response = client.get("/api/download_dataset?dataset_path=data/non_existent_file.txt", headers=VIEWER_HEADERS)
    assert response.status_code == 404
    assert "Dataset not found" in response.json()["detail"]

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

def test_run_script_already_running():
    active_processes["pretraining"] = MagicMock()
    try:
        with patch("web_app.backend.main.asyncio.create_task"):
            response = client.post("/api/run_script", json={"script_type": "pretraining"}, headers=ADMIN_HEADERS)
            assert response.status_code == 400
            assert "already running" in response.json()["detail"]
    finally:
        del active_processes["pretraining"]

def test_get_active_scripts():
    active_processes["finetuning"] = MagicMock()
    try:
        response = client.get("/api/active_scripts", headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "finetuning" in response.json()["active_scripts"]
    finally:
        del active_processes["finetuning"]

def test_stop_script_valid():
    mock_process = MagicMock()
    active_processes["evaluation"] = mock_process
    try:
        response = client.post("/api/stop_script", json={"script_type": "evaluation"}, headers=ADMIN_HEADERS)
        assert response.status_code == 200
        assert "Sent termination signal" in response.json()["message"]
        mock_process.terminate.assert_called_once()
    finally:
        if "evaluation" in active_processes:
            del active_processes["evaluation"]

def test_stop_script_not_running():
    response = client.post("/api/stop_script", json={"script_type": "nonexistent"}, headers=ADMIN_HEADERS)
    assert response.status_code == 404
    assert "not running" in response.json()["detail"]

def test_stop_all_scripts_valid():
    mock_process1 = MagicMock()
    mock_process2 = MagicMock()
    active_processes["script1"] = mock_process1
    active_processes["script2"] = mock_process2
    try:
        response = client.post("/api/stop_all_scripts", headers=ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "Sent termination signal to 2 scripts." in data["message"]
        assert "script1" in data["stopped"]
        assert "script2" in data["stopped"]
        mock_process1.terminate.assert_called_once()
        mock_process2.terminate.assert_called_once()
    finally:
        if "script1" in active_processes:
            del active_processes["script1"]
        if "script2" in active_processes:
            del active_processes["script2"]

def test_stop_all_scripts_empty():
    response = client.post("/api/stop_all_scripts", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "No active scripts to stop." in data["message"]
    assert len(data["stopped"]) == 0

def test_stop_all_scripts_unauthorized():
    response = client.post("/api/stop_all_scripts", headers=VIEWER_HEADERS)
    assert response.status_code == 403

def test_clear_logs_authorized(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "test1.log").write_text("test")
    sub_dir = logs_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "test2.log").write_text("test")

    # Mock project root to tmp_path
    original_abspath = os.path.abspath
    monkeypatch.setattr("os.path.abspath", lambda x: str(tmp_path) if "../../" in x else original_abspath(x))

    response = client.post("/api/clear_logs", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Логи успешно очищены."
    assert data["deleted_count"] == 2
    assert not (logs_dir / "test1.log").exists()
    assert not sub_dir.exists()
    assert logs_dir.exists()

def test_clear_logs_empty(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Mock project root to tmp_path
    original_abspath = os.path.abspath
    monkeypatch.setattr("os.path.abspath", lambda x: str(tmp_path) if "../../" in x else original_abspath(x))

    response = client.post("/api/clear_logs", headers=ADMIN_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Логи успешно очищены."
    assert data["deleted_count"] == 0

def test_clear_logs_unauthorized():
    response = client.post("/api/clear_logs", headers=VIEWER_HEADERS)
    assert response.status_code == 403

def test_system_stats_authorized():
    response = client.get("/api/system_stats", headers=VIEWER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "cpu_percent" in data
    assert "ram_percent" in data
    assert "ram_used_mb" in data
    assert "ram_total_mb" in data
    assert "disk_percent" in data
    assert "disk_used_gb" in data
    assert "disk_total_gb" in data
    assert "network_sent_mb" in data
    assert "network_recv_mb" in data
    assert "uptime_seconds" in data
    assert "disk_read_mb" in data
    assert "disk_write_mb" in data
    assert "swap_percent" in data
    assert "swap_used_mb" in data
    assert "swap_total_mb" in data

def test_system_stats_unauthorized():
    response = client.get("/api/system_stats")
    assert response.status_code == 401
