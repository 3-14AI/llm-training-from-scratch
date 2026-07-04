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

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Construct absolute path for the frontend directory based on the location of main.py
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
