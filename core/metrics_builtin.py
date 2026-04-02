"""
Built-in metric factories for common evaluation patterns.

Each factory returns a callable with signature (example, pred, trace=None) -> float,
compatible with DSPy's optimizer interface.
"""

import re

from core.task_config import MetricDef


# -- Metric factories --------------------------------------------------------

def make_exact_match(field_name: str):
    """Returns 1.0 if the field values match (case-insensitive, stripped), else 0.0."""
    def metric(example, pred, trace=None):
        expected = str(getattr(example, field_name, "")).strip().lower()
        predicted = str(getattr(pred, field_name, "")).strip().lower()
        return 1.0 if expected == predicted else 0.0
    metric.__name__ = f"exact_match_{field_name}"
    return metric


def make_f1_phrases(field_name: str):
    """F1 score on comma-separated phrases with fuzzy substring matching."""
    def _parse(raw: str) -> set[str]:
        if not raw or not raw.strip():
            return set()
        return {p.strip().lower() for p in raw.split(",") if p.strip()}

    def metric(example, pred, trace=None):
        expected = _parse(str(getattr(example, field_name, "")))
        predicted = _parse(str(getattr(pred, field_name, "")))

        if not expected and not predicted:
            return 1.0
        if not expected or not predicted:
            return 0.0

        true_positives = 0
        matched_expected = set()
        for p in predicted:
            for e in expected:
                if p in e or e in p:
                    true_positives += 1
                    matched_expected.add(e)
                    break

        precision = true_positives / len(predicted) if predicted else 0.0
        recall = len(matched_expected) / len(expected) if expected else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    metric.__name__ = f"f1_phrases_{field_name}"
    return metric


def make_rule_quality(field_name: str, config: dict):
    """Rule-based quality checks on a text field.

    Config options:
        banned_words: list[str] - phrases that must not appear
        max_sentence_length: int - max words per sentence
        no_passive_voice: bool - flag passive voice patterns
        skip_if_field_equals: dict - {field: value} -> return 1.0 if matched
            (e.g., {"compliant": "true"} skips scoring when copy is compliant)
    """
    banned = [w.lower() for w in config.get("banned_words", [])]
    max_len = config.get("max_sentence_length", 0)
    no_passive = config.get("no_passive_voice", False)
    skip_if = config.get("skip_if_field_equals", {})

    def _has_passive(text: str) -> bool:
        return bool(re.search(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", text, re.IGNORECASE))

    def _avg_sentence_len(text: str) -> float:
        if not text or not text.strip():
            return 0.0
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    def metric(example, pred, trace=None):
        # Skip check if the example matches a skip condition
        for skip_field, skip_val in skip_if.items():
            if str(getattr(example, skip_field, "")).strip().lower() == str(skip_val).strip().lower():
                return 1.0

        text = str(getattr(pred, field_name, "")).strip()
        if not text:
            return 0.0

        checks_passed = 0
        total_checks = 0

        if banned:
            total_checks += 1
            text_lower = text.lower()
            if not any(b in text_lower for b in banned):
                checks_passed += 1

        if max_len > 0:
            total_checks += 1
            if _avg_sentence_len(text) <= max_len:
                checks_passed += 1

        if no_passive:
            total_checks += 1
            if not _has_passive(text):
                checks_passed += 1

        return checks_passed / total_checks if total_checks > 0 else 1.0

    metric.__name__ = f"rule_quality_{field_name}"
    return metric


def make_custom(code_string: str):
    """Execute user-provided Python to create a metric function.

    The code must define: def metric(example, pred, trace=None) -> float
    WARNING: runs exec() -- local-only, never expose on a public server.
    """
    namespace = {"re": re}
    exec(code_string, namespace)
    if "metric" not in namespace or not callable(namespace["metric"]):
        raise ValueError("Custom metric code must define: def metric(example, pred, trace=None) -> float")
    return namespace["metric"]


# -- Composite builder -------------------------------------------------------

_FACTORIES = {
    "exact_match": lambda md: make_exact_match(md.target_field),
    "f1_phrases": lambda md: make_f1_phrases(md.target_field),
    "rule_quality": lambda md: make_rule_quality(md.target_field, md.rule_config),
    "custom": lambda md: make_custom(md.custom_code),
}


def build_composite_metric(metric_defs: list[MetricDef]):
    """Build a single composite metric from a list of MetricDefs.

    Normalizes weights to sum to 1.0. Returns a callable compatible with
    DSPy's optimizer interface.
    """
    sub_metrics = []
    total_weight = sum(md.weight for md in metric_defs)

    for md in metric_defs:
        factory = _FACTORIES.get(md.metric_type)
        if not factory:
            raise ValueError(f"Unknown metric type: {md.metric_type}")
        fn = factory(md)
        normalized_weight = md.weight / total_weight if total_weight > 0 else 0
        sub_metrics.append((md.name, fn, normalized_weight))

    def composite(example, pred, trace=None):
        return sum(w * fn(example, pred, trace) for _, fn, w in sub_metrics)

    composite.__name__ = "composite_metric"
    composite.sub_metrics = sub_metrics  # expose for per-metric reporting
    return composite
