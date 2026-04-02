"""API endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from core.models import get_teacher_models, get_student_models
from server.schemas import RunRequest, TaskConfigSchema, JobStatus
from server.jobs import job_manager

router = APIRouter(prefix="/api")
TASKS_DIR = Path(__file__).parent.parent / "tasks"


@router.get("/tasks")
def list_tasks() -> list[str]:
    if not TASKS_DIR.exists():
        return []
    return [f.stem for f in TASKS_DIR.glob("*.json")]


@router.get("/tasks/{name}")
def get_task(name: str) -> dict:
    path = TASKS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Task '{name}' not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



@router.post("/run")
def start_run(request: RunRequest) -> dict:
    if job_manager.is_busy():
        raise HTTPException(409, "An optimization job is already running")
    job_id = job_manager.start_job(request)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> JobStatus:
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return status


@router.get("/jobs/{job_id}/prompt/{model:path}")
def get_prompt(job_id: str, model: str) -> PlainTextResponse:
    prompt = job_manager.get_prompt(job_id, model)
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    return PlainTextResponse(prompt, media_type="text/plain")


@router.get("/models")
def list_models() -> dict:
    return {
        "teacher": get_teacher_models(),
        "student": get_student_models(),
    }
