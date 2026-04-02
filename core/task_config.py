"""
TaskConfig: a plain data object that fully describes any DSPy optimization task.

No DSPy imports here — this is purely a data layer that serializes to/from JSON.
The engine and signature_factory consume these to build DSPy objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class FieldDef:
    """One input or output field in the DSPy signature."""
    name: str
    description: str
    field_type: str  # "input" or "output"
    value_type: str = "str"  # "str", "bool", "float"


@dataclass
class MetricDef:
    """One sub-metric in the composite evaluation."""
    name: str
    metric_type: str  # "exact_match", "f1_phrases", "rule_quality", "custom"
    weight: float
    target_field: str  # which output field this metric scores
    rule_config: dict = field(default_factory=dict)  # for rule_quality
    custom_code: str = ""  # for custom type


@dataclass
class TaskConfig:
    """Complete task definition — everything needed to run DSPy optimization."""
    name: str
    description: str  # becomes the Signature docstring
    guidelines: str  # injected as an input field at runtime
    fields: list[FieldDef]
    metrics: list[MetricDef]
    examples: list[dict]  # plain dicts keyed by field name
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    @property
    def input_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if f.field_type == "input"]

    @property
    def output_fields(self) -> list[FieldDef]:
        return [f for f in self.fields if f.field_type == "output"]

    @property
    def input_field_names(self) -> list[str]:
        return [f.name for f in self.input_fields]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> TaskConfig:
        return cls(
            name=data["name"],
            description=data["description"],
            guidelines=data["guidelines"],
            fields=[FieldDef(**f) for f in data["fields"]],
            metrics=[MetricDef(**m) for m in data["metrics"]],
            examples=data["examples"],
            train_ratio=data.get("train_ratio", 0.6),
            val_ratio=data.get("val_ratio", 0.2),
            test_ratio=data.get("test_ratio", 0.2),
        )

    @classmethod
    def load(cls, path: str | Path) -> TaskConfig:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
