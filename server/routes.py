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
from core.engine import split_examples, evaluate, parse_exported_prompt, apply_edited_prompt
from server.schemas import RunRequest, EditedEvalRequest, TaskConfigSchema, JobStatus
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


@router.post("/eval-edited")
def eval_edited(request: EditedEvalRequest) -> dict:
    """Re-evaluate with edited prompts. No optimization -- fast eval only.

    Each column's edited prompt text is parsed back into instructions + demos,
    applied to a fresh module, and evaluated on the held-out test set.
    """
    log.info("Edited eval: %d columns, trials=%d", len(request.columns), request.num_trials)

    task_config = TaskConfig(
        name=request.task.name,
        description=request.task.description,
        guidelines=request.task.guidelines,
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

    results = []
    for col in request.columns:
        log.info("  Evaluating edited %s on %s...", col.label, col.model)

        lm = dspy.LM(col.model, cache=False, temperature=0)
        dspy.configure(lm=lm)

        module = get_module(task_config)

        # Parse the edited prompt and apply instructions + demos
        parsed_stages = parse_exported_prompt(col.edited_prompt)
        if parsed_stages:
            apply_edited_prompt(module, parsed_stages)

        scores, latency = evaluate(testset, module, composite_fn,
                                   num_trials=request.num_trials,
                                   threads=request.threads)

        cost = get_model_cost(col.model) or {"input_cost": None, "output_cost": None}

        log.info("  %s composite: %.1f%%", col.label, scores["composite"][0] * 100)

        results.append({
            "label": col.label,
            "model": col.model.split("/")[-1],
            "scores": {k: {"mean": v[0], "std": v[1]} for k, v in scores.items()},
            "latency": {"mean": latency[0], "std": latency[1]},
            "cost": cost,
        })

    return {"columns": results}


@router.get("/models")
def list_models() -> dict:
    return {
        "teacher": get_teacher_models(),
        "student": get_student_models(),
    }
