import os
import shlex
import subprocess
from pydantic import BaseModel

import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, UploadFile, File, Form
from fastapi.security import APIKeyHeader
from typing import List

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Ролевая модель: словарь токенов и соответствующих ролей
TOKENS = {
    "admin_secret": "admin",
    "viewer_secret": "viewer"
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(api_key: str = Depends(api_key_header)):
    """
    Проверяет валидность токена и возвращает роль пользователя.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")
    role = TOKENS.get(api_key)
    if not role:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return role

async def require_admin(role: str = Depends(get_current_user)):
    """
    Проверяет, что пользователь имеет роль admin.
    """
    if role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required")
    return role

ALLOWED_COMMANDS = {"ls", "echo", "pwd", "whoami", "python", "python3"}

class CommandRequest(BaseModel):
    command: str

@app.post("/api/execute", dependencies=[Depends(require_admin)])
def execute_command(req: CommandRequest):
    """
    Выполняет разрешенную shell-команду и возвращает результат.
    """
    if not req.command or not req.command.strip():
        raise HTTPException(status_code=400, detail="Command cannot be empty")

    try:
        parts = shlex.split(req.command)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid command format: {str(e)}")

    if not parts:
        raise HTTPException(status_code=400, detail="Command is invalid")

    base_cmd = parts[0]
    if base_cmd not in ALLOWED_COMMANDS:
        raise HTTPException(status_code=403, detail=f"Command '{base_cmd}' is not allowed")

    try:
        result = subprocess.run(parts, capture_output=True, text=True, timeout=10, shell=False)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




from pydantic import BaseModel, Field

class ConfigRequest(BaseModel):
    config_name: str = Field(..., description="Name of the configuration", pattern=r"^[a-zA-Z0-9_-]+$")
    epochs: int = Field(..., description="Number of epochs")
    batch_size: int = Field(..., description="Batch size")
    learning_rate: float = Field(..., description="Learning rate")

class ScriptRequest(BaseModel):
    script_type: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

active_processes = {}

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(None)):
    if not token or token not in TOKENS:
        await websocket.close(code=1008, reason="Unauthorized")
        return
    await manager.connect(websocket)
    try:
        while True:
            # We don't really expect messages from the client, but keep the connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def run_process_and_broadcast(script_type: str, command_parts: List[str]):
    process = await asyncio.create_subprocess_exec(
        *command_parts,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    active_processes[script_type] = process

    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if line:
                await manager.broadcast(f"[{prefix}] {line.decode('utf-8').rstrip()}")
            else:
                break

    await asyncio.gather(
        read_stream(process.stdout, "STDOUT"),
        read_stream(process.stderr, "STDERR")
    )
    await process.wait()
    if script_type in active_processes:
        del active_processes[script_type]
    await manager.broadcast(f"[SYSTEM] Process {script_type} finished with exit code {process.returncode}")


@app.post("/api/run_script", dependencies=[Depends(require_admin)])
async def run_script(req: ScriptRequest):
    scripts = {
        "pretraining": ["python", "pretraining/pretrain.py"],
        "finetuning": ["python", "fine_tuning/finetune.py"],
        "evaluation": ["python", "scripts/run_evaluation.py"]
    }

    if req.script_type not in scripts:
        raise HTTPException(status_code=400, detail=f"Unknown script type: {req.script_type}")

    if req.script_type in active_processes:
        raise HTTPException(status_code=400, detail=f"Script {req.script_type} is already running.")

    command_parts = scripts[req.script_type]

    # Run the process in the background
    asyncio.create_task(run_process_and_broadcast(req.script_type, command_parts))

    return {"message": f"Started {req.script_type} script in the background."}


@app.get("/api/active_scripts", dependencies=[Depends(get_current_user)])
async def get_active_scripts():
    """
    Returns a list of currently running scripts.
    """
    return {"active_scripts": list(active_processes.keys())}

class StopScriptRequest(BaseModel):
    script_type: str

@app.post("/api/stop_script", dependencies=[Depends(require_admin)])
async def stop_script(req: StopScriptRequest):
    """
    Stops a running script.
    """
    if req.script_type not in active_processes:
        raise HTTPException(status_code=404, detail=f"Script {req.script_type} is not running.")

    process = active_processes[req.script_type]
    try:
        process.terminate()
        # The wait() in run_process_and_broadcast will handle removal and broadcasting
        return {"message": f"Sent termination signal to {req.script_type}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop script: {str(e)}")




class QuickLaunchRequest(BaseModel):
    config_name: str

@app.get("/api/configs", dependencies=[Depends(get_current_user)])
async def get_configs():
    """
    Возвращает список доступных конфигураций.
    """
    # 1. Из файлов в директории configs/
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")
    file_configs = []
    if os.path.exists(config_dir):
        for f in os.listdir(config_dir):
            if f.endswith(".toml") or f.endswith(".py"):
                file_configs.append(f)

    # 2. Из experiment_configs.py (словарь ALL_EXPERIMENTS)
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
        from scripts.experiment_configs import ALL_EXPERIMENTS
        experiment_configs = list(ALL_EXPERIMENTS.keys())
    except Exception:
        experiment_configs = []

    return {
        "file_configs": file_configs,
        "experiment_configs": experiment_configs
    }

@app.post("/api/quick_launch", dependencies=[Depends(require_admin)])
async def quick_launch(req: QuickLaunchRequest):
    """
    Запускает эксперимент с указанным конфигом.
    """
    if not req.config_name:
        raise HTTPException(status_code=400, detail="config_name cannot be empty")

    # Проверяем, это конфиг из файлов или из experiments
    if req.config_name.endswith(".toml") or req.config_name.endswith(".py"):
        # Это файл, возможно мы захотим его запустить каким-то скриптом (условно pretrain.py)
        # Передадим как аргумент
        command_parts = ["python", "pretraining/pretrain.py", "--config_file", req.config_name]
    else:
        # Это конфиг из ALL_EXPERIMENTS, используем experiment_runner
        # Запустим e2e для быстроты или full в зависимости от задачи (захардкодим e2e для тестов)
        command_parts = ["python", "scripts/experiment_runner.py", "--exp_name", req.config_name]

    # Запускаем в бэкграунде
    asyncio.create_task(run_process_and_broadcast(f"quick_launch_{req.config_name}", command_parts))

    return {"message": f"Quick launch started for config: {req.config_name}"}

@app.post("/api/save_config", dependencies=[Depends(require_admin)])
async def save_config(req: ConfigRequest):
    """
    Сохраняет конфигурацию обучения в configs/{config_name}.toml
    """
    import os
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, f"{req.config_name}.toml")

    toml_content = f"""[training]
epochs = {req.epochs}
batch_size = {req.batch_size}
learning_rate = {req.learning_rate}
"""

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml_content)
        return {"message": f"Config {req.config_name} saved successfully.", "path": config_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")

import json

@app.get("/api/metrics", dependencies=[Depends(get_current_user)])
async def get_metrics():
    """
    Возвращает историю метрик (loss, perplexity) для отображения на графиках.
    Считывает данные из файла logs/metrics.json, если он существует.
    Иначе возвращает мок-данные.
    """
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs/metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")

    # Возвращаем мок-данные, если файла нет
    return {
        "epochs": [1, 2, 3, 4, 5],
        "loss": [2.5, 2.0, 1.8, 1.5, 1.2],
        "perplexity": [12.0, 7.5, 6.0, 4.5, 3.2]
    }

from huggingface_hub import HfApi

class ExportModelRequest(BaseModel):
    model_path: str = Field(..., description="Path to the model to export (e.g., checkpoints_finetune/model.pth)")
    hf_token: str = Field(..., description="Hugging Face access token")
    repo_id: str = Field(..., description="Target Hugging Face repository ID (e.g., username/repo-name)")

@app.post("/api/export_model", dependencies=[Depends(require_admin)])
async def export_model(req: ExportModelRequest):
    """
    Экспортирует модель на Hugging Face Hub.
    """
    if not req.model_path or not req.hf_token or not req.repo_id:
        raise HTTPException(status_code=400, detail="All fields are required")

    # Path traversal and absolute path protection
    clean_model_path = req.model_path.lstrip('/')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_model_path = os.path.abspath(os.path.join(project_root, clean_model_path))

    if not full_model_path.startswith(project_root):
        raise HTTPException(status_code=403, detail="Invalid path: Path traversal is not allowed")

    if not os.path.exists(full_model_path):
        raise HTTPException(status_code=404, detail=f"Model path not found: {req.model_path}")

    try:
        api = HfApi(token=req.hf_token)

        # Verify repo exists or create it
        try:
            api.repo_info(repo_id=req.repo_id)
        except Exception:
            # Try to create it
            try:
                api.create_repo(repo_id=req.repo_id, exist_ok=True)
            except Exception as e:
                 raise HTTPException(status_code=500, detail=f"Failed to create repo: {str(e)}")

        file_name = os.path.basename(full_model_path)

        # Upload file
        if os.path.isdir(full_model_path):
            api.upload_folder(
                folder_path=full_model_path,
                repo_id=req.repo_id,
            )
        else:
            api.upload_file(
                path_or_fileobj=full_model_path,
                path_in_repo=file_name,
                repo_id=req.repo_id,
            )

        return {"message": f"Successfully exported to {req.repo_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export model: {str(e)}")

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="The prompt text to generate from")
    max_tokens: int = Field(20, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(0, description="Top-K sampling")
    top_p: float = Field(1.0, description="Top-p (nucleus) sampling")
    repetition_penalty: float = Field(1.0, description="Repetition penalty")
    model_path: str = Field("", description="Path to the model checkpoint to use")

@app.post("/api/inference", dependencies=[Depends(get_current_user)])
async def run_inference(req: InferenceRequest):
    """
    Запускает инференс модели и возвращает сгенерированный текст.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    command = [
        "python", "scripts/run_inference.py",
        "--prompt", req.prompt,
        "--max_tokens", str(req.max_tokens),
        "--temperature", str(req.temperature),
        "--top_k", str(req.top_k),
        "--top_p", str(req.top_p),
        "--repetition_penalty", str(req.repetition_penalty)
    ]
    if req.model_path:
        command.extend(["--model_path", req.model_path])

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference process failed: {stderr.decode('utf-8')}")

        output = stdout.decode('utf-8').strip()
        # Parse the JSON output from the script
        try:
            import json
            result = json.loads(output)
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            return {"generated_text": result.get("generated_text", "")}
        except json.JSONDecodeError:
            # Fallback if not JSON
            return {"generated_text": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {str(e)}")


class DeleteArtifactRequest(BaseModel):
    artifact_path: str = Field(..., description="Relative path of the artifact to delete")

from fastapi.responses import FileResponse

@app.get("/api/download_artifact", dependencies=[Depends(get_current_user)])
async def download_artifact(artifact_path: str):
    """
    Downloads an artifact file.
    """
    if not artifact_path:
        raise HTTPException(status_code=400, detail="artifact_path is required")

    clean_path = artifact_path.lstrip('/')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.abspath(os.path.join(project_root, clean_path))

    if not full_path.startswith(project_root):
        raise HTTPException(status_code=403, detail="Invalid path: Path traversal is not allowed")

    # Double check it belongs to one of the allowed directories
    artifact_dirs = ["configs", "logs", "checkpoints", "checkpoints_finetune"]
    allowed = False
    for d in artifact_dirs:
        if full_path.startswith(os.path.abspath(os.path.join(project_root, d))):
            allowed = True
            break

    if not allowed:
        raise HTTPException(status_code=403, detail="Cannot access files outside of artifact directories")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Artifact not found")

    if os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail="Cannot download a directory")

    return FileResponse(path=full_path, filename=os.path.basename(full_path))


@app.post("/api/upload_artifact", dependencies=[Depends(require_admin)])
async def upload_artifact(
    directory: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Uploads an artifact (file) to the specified directory.
    """
    if not directory or not file.filename:
        raise HTTPException(status_code=400, detail="directory and file are required")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Check if the target directory is allowed
    allowed_dirs = ["configs", "logs", "checkpoints", "checkpoints_finetune"]
    if directory not in allowed_dirs:
        raise HTTPException(status_code=403, detail="Cannot upload files outside of artifact directories")

    target_dir = os.path.join(project_root, directory)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Prevent path traversal in filename
    clean_filename = os.path.basename(file.filename)
    target_path = os.path.join(target_dir, clean_filename)

    import shutil
    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"message": f"Artifact {clean_filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload artifact: {str(e)}")


@app.post("/api/delete_artifact", dependencies=[Depends(require_admin)])
async def delete_artifact(req: DeleteArtifactRequest):
    """
    Deletes an artifact (file or directory).
    """
    if not req.artifact_path:
        raise HTTPException(status_code=400, detail="artifact_path is required")

    clean_path = req.artifact_path.lstrip('/')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.abspath(os.path.join(project_root, clean_path))

    if not full_path.startswith(project_root):
        raise HTTPException(status_code=403, detail="Invalid path: Path traversal is not allowed")

    # Double check it belongs to one of the allowed directories
    artifact_dirs = ["configs", "logs", "checkpoints", "checkpoints_finetune"]
    allowed = False
    for d in artifact_dirs:
        if full_path.startswith(os.path.abspath(os.path.join(project_root, d))):
            allowed = True
            break

    if not allowed:
        raise HTTPException(status_code=403, detail="Cannot delete files outside of artifact directories")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        if os.path.isdir(full_path):
            import shutil
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
        return {"message": "Artifact deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete artifact: {str(e)}")


@app.get("/api/datasets", dependencies=[Depends(get_current_user)])
def get_datasets():
    """
    Возвращает список датасетов из директории data/ проекта.
    """
    import os
    import time

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    datasets = []

    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_root)
                stat = os.stat(file_path)

                datasets.append({
                    "name": file,
                    "path": rel_path,
                    "size": stat.st_size,
                    "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                    "type": "dataset"
                })

    return {"datasets": datasets}

@app.post("/api/upload_dataset", dependencies=[Depends(require_admin)])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a dataset file to the data/ directory.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Prevent path traversal in filename
    clean_filename = os.path.basename(file.filename)
    target_path = os.path.join(data_dir, clean_filename)

    import shutil
    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"message": f"Dataset {clean_filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")

class DeleteDatasetRequest(BaseModel):
    dataset_path: str = Field(..., description="Relative path of the dataset to delete")

@app.post("/api/delete_dataset", dependencies=[Depends(require_admin)])
async def delete_dataset(req: DeleteDatasetRequest):
    """
    Deletes a dataset.
    """
    if not req.dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")

    clean_path = req.dataset_path.lstrip('/')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.abspath(os.path.join(project_root, clean_path))

    if not full_path.startswith(project_root):
        raise HTTPException(status_code=403, detail="Invalid path: Path traversal is not allowed")

    data_dir = os.path.abspath(os.path.join(project_root, "data"))
    if not full_path.startswith(data_dir):
        raise HTTPException(status_code=403, detail="Cannot delete files outside of data directory")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        if os.path.isdir(full_path):
            import shutil
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
        return {"message": "Dataset deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@app.get("/api/preview_dataset", dependencies=[Depends(get_current_user)])
async def preview_dataset(dataset_path: str, lines: int = 10):
    """
    Returns a preview (first N lines) of a dataset text file.
    """
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")

    clean_path = dataset_path.lstrip('/')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.abspath(os.path.join(project_root, clean_path))

    if not full_path.startswith(project_root):
        raise HTTPException(status_code=403, detail="Invalid path: Path traversal is not allowed")

    data_dir = os.path.abspath(os.path.join(project_root, "data"))
    if not full_path.startswith(data_dir):
        raise HTTPException(status_code=403, detail="Cannot access files outside of data directory")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        preview_lines = []
        with open(full_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    break
                preview_lines.append(line.rstrip('\n'))

        preview_content = "\n".join(preview_lines)
        return {"preview": preview_content}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a valid text file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {str(e)}")


@app.get("/api/download_dataset", dependencies=[Depends(get_current_user)])
async def download_dataset(dataset_path: str):
    """
    Downloads a dataset file.
    """
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")

    clean_path = dataset_path.lstrip('/')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.abspath(os.path.join(project_root, clean_path))

    if not full_path.startswith(project_root):
        raise HTTPException(status_code=403, detail="Invalid path: Path traversal is not allowed")

    data_dir = os.path.abspath(os.path.join(project_root, "data"))
    if not full_path.startswith(data_dir):
        raise HTTPException(status_code=403, detail="Cannot access files outside of data directory")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    if os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail="Cannot download a directory")

    return FileResponse(path=full_path, filename=os.path.basename(full_path))


@app.get("/api/artifacts", dependencies=[Depends(get_current_user)])
def get_artifacts():
    """
    Возвращает список артефактов (конфиги, логи, чекпоинты) из корневой директории проекта.
    """
    import os
    import time

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    artifact_dirs = ["configs", "logs", "checkpoints", "checkpoints_finetune"]

    artifacts = []

    for d in artifact_dirs:
        dir_path = os.path.join(project_root, d)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, project_root)
                    stat = os.stat(file_path)

                    artifacts.append({
                        "name": file,
                        "path": rel_path,
                        "size": stat.st_size,
                        "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                        "type": d
                    })

    return {"artifacts": artifacts}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Construct absolute path for the frontend directory based on the location of main.py
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
