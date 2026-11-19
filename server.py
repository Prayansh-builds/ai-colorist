"""
server.py - Part 4: FastAPI Server for AI Colorist (Steps 15 to 17).
Provides RESTful APIs for photo grading, manual plan execution, and asynchronous video grading.
"""
import os
import uuid
import shutil
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import colorist
import brain as _brain
import masks
import aimasks
import video

app = FastAPI(title="AI Colorist API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./storage/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./storage/outputs")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Install masking hooks into colorist and brain
masks.install()
aimasks.install()

# Initialize AI brain
brain = _brain.Brain(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# In-memory video task tracker
TASKS: Dict[str, Dict[str, Any]] = {}

class PlanRequest(BaseModel):
    command: str
    preset: Optional[str] = None
    preset_strength: Optional[float] = 1.0

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "brain_ready": brain is not None}

@app.get("/api/v1/presets")
def get_presets():
    return {"presets": list(colorist.PRESETS.keys())}

@app.post("/api/v1/grade/photo")
async def grade_photo(
    file: UploadFile = File(...),
    command: str = Form(...),
    strength: float = Form(1.0),
    skin_protect: float = Form(0.6)
):
    if not brain:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    in_path = os.path.join(UPLOAD_DIR, f"{file_id}_in{ext}")
    out_path = os.path.join(OUTPUT_DIR, f"{file_id}_graded.jpg")

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    bgr = cv2.imread(in_path)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Call AI brain and colorist engine
    plan = brain(command, img)
    full = colorist.build_edit(img, plan)
    graded, report = colorist.safe_finish(img, full, strength=strength, skin_protect=skin_protect)

    rgb8 = (np.clip(graded, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "file_id": file_id,
        "download_url": f"/api/v1/files/{file_id}_graded.jpg",
        "plan": plan,
        "report": report
    }

@app.get("/api/v1/files/{filename}")
def download_file(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

# ---------- Background Video Task Runner ----------
def _run_video_grading(task_id: str, in_path: str, out_path: str, command: str, preset: Optional[str]):
    try:
        TASKS[task_id]["status"] = "processing"
        
        if preset and preset in colorist.PRESETS:
            edit_fn = lambda f: colorist.apply_preset(f, preset)
        elif brain:
            # Sample first frame to build shot grade
            samples = video.sample_frames(in_path, count=1, max_side=640)
            if samples:
                plan = brain(command, samples[0])
                edit_fn = lambda f: colorist.safe_finish(f, colorist.build_edit(f, plan))[0]
            else:
                edit_fn = lambda f: f
        else:
            edit_fn = lambda f: f

        video.process_video(in_path, out_path, edit_fn, max_side=720)
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["download_url"] = f"/api/v1/files/{os.path.basename(out_path)}"
    except Exception as e:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)

@app.post("/api/v1/grade/video")
async def grade_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    command: Optional[str] = Form(""),
    preset: Optional[str] = Form(None)
):
    task_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    in_path = os.path.join(UPLOAD_DIR, f"{task_id}_in{ext}")
    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_graded.mp4")

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    TASKS[task_id] = {
        "status": "queued",
        "task_id": task_id,
        "input_file": file.filename
    }

    background_tasks.add_task(_run_video_grading, task_id, in_path, out_path, command or "", preset)
    return {"task_id": task_id, "status": "queued"}

@app.get("/api/v1/tasks/{task_id}")
def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS[task_id]
