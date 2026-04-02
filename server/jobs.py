"""Background job manager for DSPy optimization runs."""

import logging
import threading
import time
import traceback
import uuid

from core.task_config import TaskConfig, FieldDef, MetricDef
from core.engine import run_full_pipeline
from server.schemas import RunRequest, JobStatus

log = logging.getLogger("dspy-demo.jobs")


class JobManager:
    def __init__(self):
        self._jobs: dict[str, JobStatus] = {}
        self._prompts: dict[str, dict[str, str]] = {}  # job_id -> {model: prompt_text}
        self._lock = threading.Lock()
        self._running_thread: threading.Thread | None = None

    def is_busy(self) -> bool:
        with self._lock:
            return any(j.status in ("pending", "running") for j in self._jobs.values())

    def start_job(self, request: RunRequest) -> str:
        if self.is_busy():
            raise RuntimeError("A job is already running")

        job_id = uuid.uuid4().hex[:8]
        self._jobs[job_id] = JobStatus(
            job_id=job_id, status="pending", current_step="Queued", progress_pct=0
        )
        self._prompts[job_id] = {}

        log.info("Job %s created: task=%s, teacher=%s, students=%s, trials=%d",
                 job_id, request.task.name, request.teacher_model,
                 request.student_models, request.num_eval_trials)

        thread = threading.Thread(target=self._run, args=(job_id, request), daemon=True)
        self._running_thread = thread
        thread.start()
        return job_id

    def get_status(self, job_id: str) -> JobStatus | None:
        return self._jobs.get(job_id)

    def get_prompt(self, job_id: str, model: str) -> str | None:
        return self._prompts.get(job_id, {}).get(model)

    def _update(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs[job_id]
            for k, v in kwargs.items():
                setattr(job, k, v)

    def _run(self, job_id: str, request: RunRequest):
        self._update(job_id, status="running")
        log.info("Job %s started", job_id)
        t0 = time.perf_counter()

        try:
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

            def on_progress(step: str, pct: int):
                self._update(job_id, current_step=step, progress_pct=pct)

            results = run_full_pipeline(
                task_config=task_config,
                teacher_model=request.teacher_model,
                student_models=request.student_models,
                num_trials=request.num_eval_trials,
                threads=request.threads,
                on_progress=on_progress,
            )

            # Store prompts for download
            for model, data in results.get("students", {}).items():
                if "prompt" in data:
                    self._prompts[job_id][model] = data["prompt"]

            elapsed = time.perf_counter() - t0
            log.info("Job %s completed in %.1fs", job_id, elapsed)

            self._update(job_id, status="completed", progress_pct=100,
                         current_step="Done!", results=results)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            log.error("Job %s failed after %.1fs: %s: %s", job_id, elapsed, type(e).__name__, e)
            self._update(job_id, status="failed", error=f"{type(e).__name__}: {e}",
                         current_step="Failed")
            traceback.print_exc()


job_manager = JobManager()
