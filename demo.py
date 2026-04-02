#!/usr/bin/env python3
"""
DSPy Brand Voice Compliance Checker Demo

Full-spectrum comparison across models with statistical rigor:
  1. Monolith: 3.1-pro with all context (the CEO's "just use the big model")
  2. Two student models, each tested naive AND DSPy-optimized:
     - gemini-3.1-flash-lite-preview (newer, medium-cheap)
     - gemini-2.5-flash-lite (oldest, cheapest)

All evaluations:
  - Run on a held-out test set (no data leakage)
  - Repeated N trials with mean +/- stddev for statistical confidence
  - temperature=0 for reproducibility
  - Parallelized with 50 threads

Run:
    uv run demo.py
"""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from dotenv import load_dotenv

from brand.guidelines import format_guidelines_prompt
from brand.trainset import trainset
from brand.valset import valset
from brand.testset import testset
from metrics import (
    compliance_accuracy,
    composite_metric,
    phrase_detection_f1,
    suggestion_quality,
)

THREADS = 50
NUM_EVAL_TRIALS = 30


# -- Helpers -----------------------------------------------------------------

def banner(text: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def _eval_one(ex, program):
    """Evaluate a single example. Returns (scores_dict, latency_ms)."""
    t0 = time.perf_counter()
    pred = program(marketing_copy=ex.marketing_copy)
    latency = (time.perf_counter() - t0) * 1000
    return {
        "composite": composite_metric(ex, pred),
        "compliance": compliance_accuracy(ex, pred),
        "phrase_f1": phrase_detection_f1(ex, pred),
        "suggestion": suggestion_quality(ex, pred),
    }, latency


def evaluate_once(dataset, program):
    """Run the program on a dataset in parallel. Returns (scores_dict, avg_latency_ms)."""
    scores = {"composite": [], "compliance": [], "phrase_f1": [], "suggestion": []}
    latencies = []

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(_eval_one, ex, program): ex for ex in dataset}
        for future in as_completed(futures):
            result, latency = future.result()
            latencies.append(latency)
            for k, v in result.items():
                scores[k].append(v)

    avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}
    avg_latency = sum(latencies) / len(latencies)
    return avg_scores, avg_latency


def _stddev(values):
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def evaluate(dataset, program):
    """Run NUM_EVAL_TRIALS evaluations and return aggregated mean +/- stddev.

    Returns: {metric: (mean, stddev)}, (latency_mean, latency_stddev)
    """
    all_scores = {k: [] for k in ["composite", "compliance", "phrase_f1", "suggestion"]}
    all_latencies = []

    for trial in range(NUM_EVAL_TRIALS):
        scores, latency = evaluate_once(dataset, program)
        all_latencies.append(latency)
        for k, v in scores.items():
            all_scores[k].append(v)

    agg_scores = {}
    for k, vals in all_scores.items():
        agg_scores[k] = (sum(vals) / len(vals), _stddev(vals))

    latency_agg = (sum(all_latencies) / len(all_latencies), _stddev(all_latencies))
    return agg_scores, latency_agg


def print_scores(scores: dict, latency: tuple, label: str) -> None:
    print(f"\n  {label} ({NUM_EVAL_TRIALS} trials):")
    for key in ["composite", "compliance", "phrase_f1", "suggestion"]:
        mean, std = scores[key]
        print(f"    {key:<22} {mean:.1%} +/- {std:.1%}")
    lat_mean, lat_std = latency
    print(f"    {'avg latency':<22} {lat_mean:,.0f} +/- {lat_std:,.0f} ms")


# -- DSPy Signature & Module ------------------------------------------------

class BrandCompliance(dspy.Signature):
    """Check if marketing copy complies with brand voice guidelines.
    If non-compliant, identify the problematic phrases and suggest an on-brand replacement."""

    guidelines: str = dspy.InputField(desc="The brand voice guidelines")
    marketing_copy: str = dspy.InputField(desc="The marketing copy to check")
    compliant: str = dspy.OutputField(desc="'true' if copy is on-brand, 'false' if not")
    flagged_phrases: str = dspy.OutputField(
        desc="Comma-separated list of problematic phrases, or empty string if compliant"
    )
    suggestion: str = dspy.OutputField(
        desc="Rewritten on-brand version of the copy, or empty string if already compliant"
    )


class BrandComplianceChecker(dspy.Module):
    def __init__(self, guidelines: str):
        self.guidelines = guidelines
        self.check = dspy.ChainOfThought(BrandCompliance)

    def forward(self, marketing_copy: str):
        return self.check(guidelines=self.guidelines, marketing_copy=marketing_copy)


# -- Optimization helper -----------------------------------------------------

def optimize_for_model(student_lm, teacher_lm_cached, guidelines, label):
    """Run MIPROv2 optimization for a single student model. Returns optimized module."""
    print(f"\n  [{label}] Starting MIPROv2 optimization...")

    dspy.configure(lm=student_lm)
    optimizer = dspy.MIPROv2(
        metric=composite_metric,
        auto="light",
        num_threads=THREADS,
        teacher_settings=dict(lm=teacher_lm_cached),
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        verbose=False,
    )

    optimized = optimizer.compile(
        BrandComplianceChecker(guidelines=guidelines),
        trainset=trainset,
        valset=valset,
        minibatch_size=10,
    )

    print(f"  [{label}] Optimization complete!")
    return optimized


# -- Main --------------------------------------------------------------------

def main():
    load_dotenv()

    teacher_model = "gemini/gemini-3.1-pro-preview"
    student_models = [
        ("gemini/gemini-3.1-flash-lite-preview", "$0.25"),
        ("gemini/gemini-2.5-flash-lite", "$0.10"),
    ]

    banner("STEP 1 - Configure models")

    # temperature=0 for reproducibility, cache=False for honest latency
    teacher_lm = dspy.LM(teacher_model, cache=False, temperature=0)
    teacher_lm_cached = dspy.LM(teacher_model, temperature=0)
    student_lms = {
        model: dspy.LM(model, cache=False, temperature=0) for model, _ in student_models
    }

    print(f"  Teacher / monolith:  {teacher_model} ($2.00/M input)")
    for model, cost in student_models:
        print(f"  Student:             {model} ({cost}/M input)")
    print(f"  Training examples:   {len(trainset)}")
    print(f"  Validation examples: {len(valset)}")
    print(f"  Test examples:       {len(testset)} (held out)")
    print(f"  Eval trials:         {NUM_EVAL_TRIALS} (mean +/- stddev)")
    print(f"  Temperature:         0 (deterministic)")
    print(f"  Parallelism:         {THREADS} threads")
    print(f"  Cache:               DISABLED for evaluations")

    guidelines = format_guidelines_prompt()

    # -- Step 2: Monolith baseline -------------------------------------------

    banner("STEP 2 - Monolith baseline (3.1-pro, all context)")
    print("  The 'throw the smartest model at it' approach.\n")

    dspy.configure(lm=teacher_lm)

    monolith = BrandComplianceChecker(guidelines=guidelines)
    monolith.check.predict.demos = [
        {
            "marketing_copy": ex.marketing_copy,
            "compliant": ex.compliant,
            "flagged_phrases": ex.flagged_phrases,
            "suggestion": ex.suggestion,
        }
        for ex in trainset
    ]

    monolith_scores, monolith_latency = evaluate(testset, monolith)
    print_scores(monolith_scores, monolith_latency, "Monolith (3.1-pro)")

    # -- Step 3: Naive baselines ---------------------------------------------

    banner("STEP 3 - Naive baselines (zero-shot, no optimization)")

    naive_results = {}
    for model, cost in student_models:
        short = model.split("/")[-1]
        print(f"\n  Evaluating {short} naive ({NUM_EVAL_TRIALS} trials)...")
        dspy.configure(lm=student_lms[model])
        naive = BrandComplianceChecker(guidelines=guidelines)
        scores, latency = evaluate(testset, naive)
        naive_results[model] = (scores, latency)
        print_scores(scores, latency, f"Naive ({short})")

    # -- Step 4: Optimize with MIPROv2 for each student model ----------------

    banner("STEP 4 - Optimizing with MIPROv2 for each student model")

    optimized_programs = {}
    for model, cost in student_models:
        short = model.split("/")[-1]
        optimized_programs[model] = optimize_for_model(
            student_lms[model], teacher_lm_cached, guidelines, short,
        )

    # -- Step 5: Evaluate optimized models -----------------------------------

    banner("STEP 5 - Optimized evals on HELD-OUT test set")

    optimized_results = {}
    for model, cost in student_models:
        short = model.split("/")[-1]
        print(f"\n  Evaluating {short} optimized ({NUM_EVAL_TRIALS} trials)...")
        dspy.configure(lm=student_lms[model])
        scores, latency = evaluate(testset, optimized_programs[model])
        optimized_results[model] = (scores, latency)
        print_scores(scores, latency, f"DSPy optimized ({short})")

    # -- Step 6: Full comparison table ---------------------------------------

    banner("STEP 6 - Full comparison (held-out test, mean of 3 trials)")

    # Build column data
    columns = []
    columns.append(("Monolith", "3.1-pro", "$2.00",
                     monolith_scores, monolith_latency))
    for model, cost in student_models:
        short = model.split("/")[-1]
        columns.append(("Naive", short, cost,
                         *naive_results[model]))
        columns.append(("DSPy", short, cost,
                         *optimized_results[model]))

    # Print table
    cw = 16  # column width
    n_cols = len(columns)

    # Header: approach
    print(f"\n  {'':.<20}", end="")
    for label, _, _, _, _ in columns:
        print(f" {label:>{cw}}", end="")
    print()

    # Header: model
    print(f"  {'model':.<20}", end="")
    for _, short, _, _, _ in columns:
        display = short if len(short) <= cw else short[:cw-2] + ".."
        print(f" {display:>{cw}}", end="")
    print()

    # Header: cost
    print(f"  {'cost (in/1M)':.<20}", end="")
    for _, _, cost, _, _ in columns:
        print(f" {cost:>{cw}}", end="")
    print()

    sep = 20 + (cw + 1) * n_cols
    print(f"  {'-' * sep}")

    # Metric rows with +/- stddev
    for key in ["composite", "compliance", "phrase_f1", "suggestion"]:
        print(f"  {key:.<20}", end="")
        for _, _, _, scores, _ in columns:
            mean, std = scores[key]
            if std < 0.001:
                print(f" {mean:>{cw}.1%}", end="")
            else:
                cell = f"{mean:.1%}+/-{std:.1%}"
                print(f" {cell:>{cw}}", end="")
        print()

    # Latency row
    print(f"  {'latency (ms)':.<20}", end="")
    for _, _, _, _, (lat_mean, lat_std) in columns:
        if lat_std < 1:
            print(f" {lat_mean:>{cw},.0f}", end="")
        else:
            cell = f"{lat_mean:,.0f}+/-{lat_std:,.0f}"
            print(f" {cell:>{cw}}", end="")
    print()

    # -- Step 7: Show what DSPy generated for cheapest model -----------------

    cheapest_model = student_models[-1][0]
    cheapest_short = cheapest_model.split("/")[-1]

    banner(f"STEP 7 - What DSPy generated for {cheapest_short}")

    optimized = optimized_programs[cheapest_model]
    predictor = optimized.check.predict if hasattr(optimized.check, "predict") else optimized.check

    if hasattr(predictor, "demos") and predictor.demos:
        print(f"\n  Few-shot demos selected: {len(predictor.demos)}")
        for i, demo in enumerate(predictor.demos):
            copy_text = demo.get("marketing_copy", "")
            compliant_val = demo.get("compliant", "")
            print(f"\n  --- Demo {i + 1} ---")
            if copy_text:
                print(f"  Copy:      {copy_text[:100]}{'...' if len(copy_text) > 100 else ''}")
            if compliant_val:
                print(f"  Compliant: {compliant_val}")

    sig = predictor.signature if hasattr(predictor, "signature") else None
    if sig and hasattr(sig, "instructions") and sig.instructions:
        instr = sig.instructions
        if len(instr) > 500:
            print(f"\n  Optimized instructions (first 500 chars):\n")
            print(f"  {instr[:500]}...")
            print(f"\n  (Full instructions: {len(instr)} chars -- see saved JSON)")
        else:
            print(f"\n  Optimized instructions:\n  {instr}")

    # -- Step 8: Save + extract portable prompt ------------------------------

    banner("STEP 8 - Save & extract production prompts")

    for model, cost in student_models:
        short = model.split("/")[-1]
        save_path = f"optimized_{short}.json"
        optimized_programs[model].save(save_path)
        print(f"  Saved: {save_path}")

    cheapest_save = f"optimized_{cheapest_short}.json"
    prompt_path = "production_prompt.txt"
    with open(cheapest_save, "r") as f:
        program_data = json.load(f)

    predictor_data = program_data.get("check.predict", {})
    instructions = predictor_data.get("signature", {}).get("instructions", "")
    demos = predictor_data.get("demos", [])

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("=" * 64 + "\n")
        f.write(f"PRODUCTION PROMPT for {cheapest_short}\n")
        f.write("Extracted from DSPy optimization. No DSPy dependency needed.\n")
        f.write("=" * 64 + "\n\n")
        f.write("== SYSTEM INSTRUCTIONS ==\n\n")
        f.write(instructions + "\n\n")
        if demos:
            f.write("== FEW-SHOT EXAMPLES ==\n\n")
            for i, demo in enumerate(demos):
                f.write(f"--- Example {i + 1} ---\n")
                f.write(f"Marketing Copy: {demo.get('marketing_copy', '')}\n")
                if demo.get("reasoning"):
                    f.write(f"Reasoning: {demo['reasoning']}\n")
                f.write(f"Compliant: {demo.get('compliant', '')}\n")
                f.write(f"Flagged Phrases: {demo.get('flagged_phrases', '')}\n")
                f.write(f"Suggestion: {demo.get('suggestion', '')}\n\n")
        f.write("== USER MESSAGE TEMPLATE ==\n\n")
        f.write("Marketing Copy: {INSERT_COPY_HERE}\n")

    print(f"  Extracted portable prompt: {prompt_path}")
    print()
    print("  Production usage (no DSPy needed):")
    print(f"    1. Read {prompt_path}")
    print("    2. Use SYSTEM INSTRUCTIONS as your system prompt")
    print("    3. Include FEW-SHOT EXAMPLES in the conversation")
    print("    4. Send user's copy as the final message")
    print("    5. Works with any LLM API")

    # -- Done ----------------------------------------------------------------

    banner("DONE")
    print()
    print(f"  All scores are mean of {NUM_EVAL_TRIALS} trials at temperature=0 on held-out test set.")
    print()
    print("  Summary:")
    m_comp, _ = monolith_scores["composite"]
    m_lat, _ = monolith_latency
    print(f"    Monolith (3.1-pro):  {m_comp:.1%} composite, {m_lat:,.0f}ms, $2.00/M")
    for model, cost in student_models:
        short = model.split("/")[-1]
        ns, nl = naive_results[model]
        os_, ol = optimized_results[model]
        n_comp, _ = ns["composite"]
        n_lat, _ = nl
        o_comp, _ = os_["composite"]
        o_lat, _ = ol
        print(f"    {short} naive:  {n_comp:.1%}, {n_lat:,.0f}ms, {cost}/M")
        print(f"    {short} DSPy:   {o_comp:.1%}, {o_lat:,.0f}ms, {cost}/M")
    print()


if __name__ == "__main__":
    main()
