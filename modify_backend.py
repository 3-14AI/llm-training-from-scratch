import re

with open("web_app/backend/main.py", "r", encoding="utf-8") as f:
    content = f.read()

imports = """
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from typing import List
"""
content = re.sub(r"from fastapi import FastAPI, HTTPException", imports, content)

new_endpoints = """
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
"""

content = content.replace('@app.get("/api/health")', new_endpoints + '\n@app.get("/api/health")')

with open("web_app/backend/main.py", "w", encoding="utf-8") as f:
    f.write(content)
