import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Construct absolute path for the frontend directory based on the location of main.py
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
