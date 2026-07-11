import os
import shlex
import subprocess
from pydantic import BaseModel

import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
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

async def run_process_and_broadcast(command_parts: List[str]):
    process = await asyncio.create_subprocess_exec(
        *command_parts,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

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
    await manager.broadcast(f"[SYSTEM] Process finished with exit code {process.returncode}")


@app.post("/api/run_script", dependencies=[Depends(require_admin)])
async def run_script(req: ScriptRequest):
    scripts = {
        "pretraining": ["python", "pretraining/pretrain.py"],
        "finetuning": ["python", "fine_tuning/finetune.py"],
        "evaluation": ["python", "scripts/run_evaluation.py"]
    }

    if req.script_type not in scripts:
        raise HTTPException(status_code=400, detail=f"Unknown script type: {req.script_type}")

    command_parts = scripts[req.script_type]

    # Run the process in the background
    asyncio.create_task(run_process_and_broadcast(command_parts))

    return {"message": f"Started {req.script_type} script in the background."}


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
    asyncio.create_task(run_process_and_broadcast(command_parts))

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
        "--temperature", str(req.temperature)
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
