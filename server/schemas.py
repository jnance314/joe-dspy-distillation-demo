"""Pydantic models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldDefSchema(BaseModel):
    name: str
    description: str
    field_type: str  # "input" or "output"
    value_type: str = "str"


class MetricDefSchema(BaseModel):
    name: str
    metric_type: str  # exact_match, f1_phrases, rule_quality, custom
    weight: float
    target_field: str
    rule_config: dict = Field(default_factory=dict)
    custom_code: str = ""


class TaskConfigSchema(BaseModel):
    name: str
    description: str
    guidelines: str
    module_key: str = "brand_voice"
    fields: list[FieldDefSchema]
    metrics: list[MetricDefSchema]
    examples: list[dict]
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


class RunRequest(BaseModel):
    task: TaskConfigSchema
    teacher_model: str = "gemini/gemini-3.1-pro-preview"
    student_models: list[str] = Field(default_factory=lambda: ["gemini/gemini-2.5-flash-lite"])
    num_eval_trials: int = 10
    threads: int = 50


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    current_step: str = ""
    progress_pct: int = 0
    results: dict | None = None
    error: str | None = None
