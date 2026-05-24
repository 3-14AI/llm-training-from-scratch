import os
import shlex
import subprocess
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
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


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Construct absolute path for the frontend directory based on the location of main.py
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
