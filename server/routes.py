"""API endpoints."""

import json
import logging
from pathlib import Path

import dspy
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from core.models import get_teacher_models, get_student_models, get_model_cost
from core.task_config import TaskConfig, FieldDef, MetricDef
from core.modules import get_module
from core.metrics_builtin import build_composite_metric
from core.engine import split_examples, evaluate
from server.schemas import RunRequest, CustomEvalRequest, TaskConfigSchema, JobStatus
from server.jobs import job_manager

log = logging.getLogger("dspy-demo.routes")

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


@router.post("/eval-custom")
def eval_custom(request: CustomEvalRequest) -> dict:
    """Run evaluation with custom guidelines. No optimization -- fast eval only."""
    log.info("Custom eval: model=%s, trials=%d", request.model, request.num_trials)

    task_config = TaskConfig(
        name=request.task.name,
        description=request.task.description,
        guidelines=request.custom_guidelines,  # Override with custom text
        module_key=request.task.module_key,
        fields=[FieldDef(**f.model_dump()) for f in request.task.fields],
        metrics=[MetricDef(**m.model_dump()) for m in request.task.metrics],
        examples=request.task.examples,
        train_ratio=request.task.train_ratio,
        val_ratio=request.task.val_ratio,
        test_ratio=request.task.test_ratio,
    )

    composite_fn = build_composite_metric(task_config.metrics)
    _trainset, _valset, testset = split_examples(task_config)

    student_lm = dspy.LM(request.model, cache=False, temperature=0)
    dspy.configure(lm=student_lm)
    module = get_module(task_config)

    scores, latency = evaluate(testset, module, composite_fn,
                               num_trials=request.num_trials,
                               threads=request.threads)

    cost = get_model_cost(request.model) or {"input_cost": None, "output_cost": None}

    log.info("Custom eval complete: composite=%.1f%%", scores["composite"][0] * 100)

    return {
        "scores": {k: {"mean": v[0], "std": v[1]} for k, v in scores.items()},
        "latency": {"mean": latency[0], "std": latency[1]},
        "cost": cost,
    }


@router.get("/models")
def list_models() -> dict:
    return {
        "teacher": get_teacher_models(),
        "student": get_student_models(),
    }
