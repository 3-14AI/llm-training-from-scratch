import os
import shlex
import subprocess
from pydantic import BaseModel

import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from typing import List

from fastapi.staticfiles import StaticFiles

app = FastAPI()

ALLOWED_COMMANDS = {"ls", "echo", "pwd", "whoami", "python", "python3"}

class CommandRequest(BaseModel):
    command: str

@app.post("/api/execute")
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
async def websocket_endpoint(websocket: WebSocket):
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


@app.post("/api/run_script")
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

@app.get("/api/configs")
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

@app.post("/api/quick_launch")
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

@app.post("/api/save_config")
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

@app.get("/api/metrics")
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

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Construct absolute path for the frontend directory based on the location of main.py
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
