"""
Task-agnostic DSPy optimization engine.

Extracted from demo.py — handles evaluation, optimization, prompt export,
and the full comparison pipeline. Works with any TaskConfig.
"""

import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import dspy

log = logging.getLogger("dspy-demo.engine")

from core.task_config import TaskConfig
from core.modules import get_module
from core.metrics_builtin import build_composite_metric
from core.models import get_model_cost

THREADS = 50


# -- Example splitting -------------------------------------------------------

def split_examples(task_config: TaskConfig) -> tuple[list, list, list]:
    """Split task_config.examples into train/val/test dspy.Example lists."""
    input_names = task_config.input_field_names
    examples = list(task_config.examples)
    random.seed(42)
    random.shuffle(examples)

    n = len(examples)
    n_train = max(1, round(n * task_config.train_ratio))
    n_val = max(1, round(n * task_config.val_ratio))
    train_dicts = examples[:n_train]
    val_dicts = examples[n_train:n_train + n_val]
    test_dicts = examples[n_train + n_val:]

    log.info("Data split: %d train / %d val / %d test (from %d total)",
             len(train_dicts), len(val_dicts), len(test_dicts), n)

    def to_dspy(dicts):
        result = []
        for d in dicts:
            ex = dspy.Example(**d).with_inputs(*input_names)
            result.append(ex)
        return result

    return to_dspy(train_dicts), to_dspy(val_dicts), to_dspy(test_dicts)


# -- Evaluation --------------------------------------------------------------

def _eval_one(ex, program, composite_fn):
    """Evaluate a single example. Returns (sub_scores_dict, latency_ms)."""
    t0 = time.perf_counter()
    kwargs = {f: getattr(ex, f) for f in ex.inputs().keys()}
    pred = program(**kwargs)
    latency = (time.perf_counter() - t0) * 1000

    scores = {}
    for name, fn, _ in composite_fn.sub_metrics:
        scores[name] = fn(ex, pred)
    scores["composite"] = composite_fn(ex, pred)
    return scores, latency


def _stddev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def evaluate_once(dataset, program, composite_fn, threads=THREADS):
    """Single pass parallel evaluation. Returns (scores_dict, avg_latency_ms)."""
    all_scores = {}
    latencies = []

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(_eval_one, ex, program, composite_fn): ex for ex in dataset}
        for future in as_completed(futures):
            scores, latency = future.result()
            latencies.append(latency)
            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)

    avg_scores = {k: sum(v) / len(v) for k, v in all_scores.items()}
    avg_latency = sum(latencies) / len(latencies)
    return avg_scores, avg_latency


def evaluate(dataset, program, composite_fn, num_trials=10, threads=THREADS):
    """Multi-trial evaluation. Returns {metric: (mean, stddev)}, (lat_mean, lat_stddev)."""
    log.info("Starting evaluation: %d trials x %d examples (%d threads)",
             num_trials, len(dataset), threads)
    t0 = time.perf_counter()
    all_scores = {}
    all_latencies = []

    for trial in range(num_trials):
        scores, latency = evaluate_once(dataset, program, composite_fn, threads)
        all_latencies.append(latency)
        for k, v in scores.items():
            all_scores.setdefault(k, []).append(v)
        if (trial + 1) % max(1, num_trials // 5) == 0 or trial == num_trials - 1:
            log.info("  Trial %d/%d done (composite: %.1f%%, latency: %.0fms)",
                     trial + 1, num_trials, scores.get("composite", 0) * 100, latency)

    elapsed = time.perf_counter() - t0
    agg = {k: (sum(v) / len(v), _stddev(v)) for k, v in all_scores.items()}
    lat_agg = (sum(all_latencies) / len(all_latencies), _stddev(all_latencies))
    log.info("Evaluation complete in %.1fs -- composite: %.1f%% +/- %.1f%%, avg latency: %.0fms",
             elapsed, agg["composite"][0] * 100, agg["composite"][1] * 100, lat_agg[0])
    return agg, lat_agg


# -- Monolith builder --------------------------------------------------------

def build_monolith(task_config: TaskConfig, trainset: list) -> dspy.Module:
    """Build a module with all training examples stuffed as few-shot demos."""
    module = get_module(task_config)
    demos = [
        {k: v for k, v in ex.toDict().items() if k not in ("dspy_uuid", "dspy_split")}
        for ex in trainset
    ]
    # Set demos on ALL predictors in the module (supports multi-predictor)
    for _name, predictor in module.named_predictors():
        pred = predictor.predict if hasattr(predictor, "predict") else predictor
        pred.demos = demos
    return module


# -- Optimization ------------------------------------------------------------

def optimize(task_config: TaskConfig, composite_fn, student_lm, teacher_lm,
             trainset, valset, threads=THREADS):
    """Run MIPROv2 optimization. Returns the optimized module."""
    log.info("Starting MIPROv2 optimization (auto=light, %d train, %d val, %d threads)",
             len(trainset), len(valset), threads)
    t0 = time.perf_counter()
    dspy.configure(lm=student_lm)
    optimizer = dspy.MIPROv2(
        metric=composite_fn,
        auto="light",
        num_threads=threads,
        teacher_settings=dict(lm=teacher_lm),
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        verbose=False,
    )
    module = get_module(task_config)
    result = optimizer.compile(module, trainset=trainset, valset=valset, minibatch_size=10)
    elapsed = time.perf_counter() - t0
    log.info("MIPROv2 optimization complete in %.1fs", elapsed)
    return result


# -- Prompt export -----------------------------------------------------------

def export_prompt(optimized_module) -> str:
    """Extract the portable production prompt from an optimized module.

    Supports both single-predictor and multi-predictor (multi-hop) modules.
    Each predictor's instructions and few-shot demos are exported as a separate stage.
    """
    lines = ["=" * 64,
             "PRODUCTION PROMPT - extracted from DSPy optimization",
             "Use this with any LLM API. No DSPy dependency needed.",
             "=" * 64, ""]

    predictors = list(optimized_module.named_predictors())

    for pred_name, predictor_obj in predictors:
        pred = (predictor_obj.predict
                if hasattr(predictor_obj, "predict")
                else predictor_obj)

        stage_label = pred_name.replace(".", " > ")
        lines.append(f"== STAGE: {stage_label} ==\n")

        sig = pred.signature if hasattr(pred, "signature") else None
        if sig and hasattr(sig, "instructions") and sig.instructions:
            lines.append("-- INSTRUCTIONS --\n")
            lines.append(sig.instructions)
            lines.append("")

        demos = pred.demos if hasattr(pred, "demos") else []
        if demos:
            lines.append(f"-- FEW-SHOT EXAMPLES ({len(demos)}) --\n")
            for i, demo in enumerate(demos):
                lines.append(f"--- Example {i + 1} ---")
                for k, v in demo.items():
                    if k not in ("augmented", "dspy_uuid", "dspy_split"):
                        lines.append(f"{k}: {v}")
                lines.append("")

        lines.append("")

    lines.append("== USER MESSAGE TEMPLATE ==\n")
    lines.append("(Fill in your input fields here)")
    return "\n".join(lines)


def parse_exported_prompt(text: str) -> list[dict]:
    """Parse an exported prompt back into stages with instructions and demos.

    Returns a list of dicts, one per stage:
        [{"stage": "check > predict", "instructions": "...", "demos": [{...}, ...]}]
    """
    import re

    stages = []
    # Split by stage markers
    stage_blocks = re.split(r"== STAGE: (.+?) ==", text)

    # stage_blocks: ['header...', 'stage_name', 'stage_content', 'stage_name', 'stage_content', ...]
    i = 1
    while i < len(stage_blocks) - 1:
        stage_name = stage_blocks[i].strip()
        stage_content = stage_blocks[i + 1]
        i += 2

        instructions = ""
        demos = []

        # Extract instructions
        instr_match = re.search(r"-- INSTRUCTIONS --\s*\n(.*?)(?=--|== |$)", stage_content, re.DOTALL)
        if instr_match:
            instructions = instr_match.group(1).strip()

        # Extract demos
        demo_blocks = re.split(r"--- Example \d+ ---", stage_content)
        for block in demo_blocks[1:]:  # skip text before first example
            demo = {}
            # Stop at next section marker
            block = re.split(r"\n(?=--|== )", block)[0]
            for line in block.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                colon_idx = line.find(": ")
                if colon_idx > 0:
                    key = line[:colon_idx].strip()
                    val = line[colon_idx + 2:].strip()
                    demo[key] = val
            if demo:
                demos.append(demo)

        stages.append({
            "stage": stage_name,
            "instructions": instructions,
            "demos": demos,
        })

    return stages


def apply_edited_prompt(module, parsed_stages: list[dict]):
    """Apply parsed prompt edits to a module's predictors.

    Matches stages to predictors by order (first stage -> first predictor, etc.).
    Sets instructions and demos on each predictor.
    """
    predictors = list(module.named_predictors())

    for i, stage in enumerate(parsed_stages):
        if i >= len(predictors):
            break
        _name, predictor_obj = predictors[i]
        pred = (predictor_obj.predict
                if hasattr(predictor_obj, "predict")
                else predictor_obj)

        if stage["instructions"]:
            # Directly set the instructions attribute on the signature
            pred.signature.__doc__ = stage["instructions"]
            if hasattr(pred.signature, "instructions"):
                pred.signature.instructions = stage["instructions"]

        pred.demos = stage["demos"]


# -- Full pipeline -----------------------------------------------------------

def run_full_pipeline(task_config: TaskConfig, teacher_model: str,
                      student_models: list[str], num_trials: int = 10,
                      threads: int = THREADS,
                      on_progress: Callable[[str, int], None] | None = None):
    """Run the complete comparison pipeline.

    Returns a dict with all results:
        {
            "monolith": {"scores": {...}, "latency": (mean, std)},
            "students": {
                "model_name": {
                    "naive": {"scores": {...}, "latency": (mean, std)},
                    "optimized": {"scores": {...}, "latency": (mean, std)},
                    "prompt": "...",
                }
            }
        }
    """
    def progress(step: str, pct: int):
        log.info("[%3d%%] %s", pct, step)
        if on_progress:
            on_progress(step, pct)

    log.info("=" * 60)
    log.info("PIPELINE START: task=%s, teacher=%s, students=%s, trials=%d",
             task_config.name, teacher_model, student_models, num_trials)
    log.info("=" * 60)

    composite_fn = build_composite_metric(task_config.metrics)
    log.info("Composite metric built: %s",
             ", ".join(f"{n} (w={w:.2f})" for n, _, w in composite_fn.sub_metrics))

    trainset, valset, testset = split_examples(task_config)

    progress("Splitting data...", 5)

    # Configure LMs
    teacher_lm = dspy.LM(teacher_model, cache=False, temperature=0)
    teacher_lm_cached = dspy.LM(teacher_model, temperature=0)
    log.info("LMs configured: teacher=%s (cache=off for eval), cached teacher for optimization", teacher_model)

    # -- Monolith baseline --
    progress(f"Evaluating monolith ({teacher_model.split('/')[-1]})...", 10)
    dspy.configure(lm=teacher_lm)
    monolith = build_monolith(task_config, trainset)
    mono_scores, mono_latency = evaluate(testset, monolith, composite_fn, num_trials, threads)

    teacher_cost = get_model_cost(teacher_model) or {"input_cost": None, "output_cost": None}
    monolith_prompt = export_prompt(monolith)
    results = {
        "monolith": {
            "model": teacher_model,
            "scores": {k: {"mean": v[0], "std": v[1]} for k, v in mono_scores.items()},
            "latency": {"mean": mono_latency[0], "std": mono_latency[1]},
            "cost": teacher_cost,
            "prompt": monolith_prompt,
        },
        "students": {},
    }

    log.info("Monolith composite: %.1f%%", mono_scores["composite"][0] * 100)

    # -- Per-student: naive + optimize + eval --
    n_students = len(student_models)
    for i, student_model in enumerate(student_models):
        short = student_model.split("/")[-1]
        log.info("-" * 40)
        log.info("Student %d/%d: %s", i + 1, n_students, student_model)
        log.info("-" * 40)
        student_lm = dspy.LM(student_model, cache=False, temperature=0)
        base_pct = 15 + int(i * 80 / n_students)

        # Naive eval
        progress(f"Evaluating {short} naive...", base_pct)
        dspy.configure(lm=student_lm)
        naive_module = get_module(task_config)
        naive_scores, naive_latency = evaluate(testset, naive_module, composite_fn, num_trials, threads)
        naive_prompt = export_prompt(naive_module)

        # Optimize
        progress(f"Optimizing {short} with MIPROv2...", base_pct + int(20 / n_students))
        optimized = optimize(task_config, composite_fn, student_lm, teacher_lm_cached,
                             trainset, valset, threads)

        # Optimized eval
        progress(f"Evaluating {short} optimized...", base_pct + int(60 / n_students))
        dspy.configure(lm=student_lm)
        opt_scores, opt_latency = evaluate(testset, optimized, composite_fn, num_trials, threads)

        # Export prompt
        prompt = export_prompt(optimized)

        student_cost = get_model_cost(student_model) or {"input_cost": None, "output_cost": None}
        results["students"][student_model] = {
            "naive": {
                "scores": {k: {"mean": v[0], "std": v[1]} for k, v in naive_scores.items()},
                "latency": {"mean": naive_latency[0], "std": naive_latency[1]},
                "prompt": naive_prompt,
            },
            "optimized": {
                "scores": {k: {"mean": v[0], "std": v[1]} for k, v in opt_scores.items()},
                "latency": {"mean": opt_latency[0], "std": opt_latency[1]},
            },
            "cost": student_cost,
            "prompt": prompt,
        }

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("  Monolith: %.1f%% composite", mono_scores["composite"][0] * 100)
    for model, data in results["students"].items():
        short = model.split("/")[-1]
        log.info("  %s naive: %.1f%% | optimized: %.1f%%",
                 short, data["naive"]["scores"]["composite"]["mean"] * 100,
                 data["optimized"]["scores"]["composite"]["mean"] * 100)
    log.info("=" * 60)

    # -- AI-generated summary --
    progress("Generating AI summary...", 95)
    results["summary"] = _generate_summary(results, teacher_model)

    progress("Done!", 100)
    return results


def _generate_summary(results: dict, teacher_model: str) -> str:
    """Use the teacher model to generate a plain-English interpretation of results."""
    try:
        log.info("Generating AI summary with %s", teacher_model)
        summary_lm = dspy.LM(teacher_model, temperature=0.3)

        # Build a text representation of the results for the LM
        lines = []
        mono = results["monolith"]
        lines.append(f"Monolith ({mono['model'].split('/')[-1]}):")
        lines.append(f"  Composite: {mono['scores']['composite']['mean']:.1%}")
        lines.append(f"  Latency: {mono['latency']['mean']:.0f}ms")
        if mono.get("cost", {}).get("input_cost"):
            lines.append(f"  Cost: ${mono['cost']['input_cost']}/M input, ${mono['cost']['output_cost']}/M output")

        for model, data in results["students"].items():
            short = model.split("/")[-1]
            cost = data.get("cost", {})
            cost_str = f"${cost['input_cost']}/M input, ${cost['output_cost']}/M output" if cost.get("input_cost") else "unknown"

            lines.append(f"\n{short} (cost: {cost_str}):")
            lines.append(f"  Naive composite: {data['naive']['scores']['composite']['mean']:.1%}, latency: {data['naive']['latency']['mean']:.0f}ms")
            lines.append(f"  DSPy optimized composite: {data['optimized']['scores']['composite']['mean']:.1%}, latency: {data['optimized']['latency']['mean']:.0f}ms")

            for metric_name in data["naive"]["scores"]:
                if metric_name == "composite":
                    continue
                naive_val = data["naive"]["scores"][metric_name]["mean"]
                opt_val = data["optimized"]["scores"][metric_name]["mean"]
                lines.append(f"  {metric_name}: naive={naive_val:.1%} -> optimized={opt_val:.1%}")

        results_text = "\n".join(lines)

        prompt = f"""You are analyzing the results of a prompt optimization experiment.
The goal was to see if a cheap AI model can match an expensive model's performance
after automatic prompt optimization with DSPy's MIPROv2.

Here are the results:

{results_text}

Write a 3-4 paragraph summary for a technical audience (developers and engineering managers) that:
1. States the key finding clearly in the first sentence
2. Highlights the most impressive improvements (which metrics improved most, by how much)
3. Compares cost and latency tradeoffs between approaches
4. Gives a concrete recommendation on which model to deploy and why

Be direct and specific with numbers. No filler. No hedging."""

        dspy.configure(lm=summary_lm)
        result = dspy.Predict("prompt -> summary")(prompt=prompt)
        log.info("AI summary generated (%d chars)", len(result.summary))
        return result.summary

    except Exception as e:
        log.warning("Failed to generate AI summary: %s", e)
        return ""
