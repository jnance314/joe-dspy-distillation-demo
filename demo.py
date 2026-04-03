#!/usr/bin/env python3
"""
DSPy Prompt Optimizer - CLI Demo

Runs the full optimization pipeline from the command line using the
shared core engine. Defaults to the brand voice compliance task but
can run any task JSON via --task.

Run:
    uv run demo.py
    uv run demo.py --task tasks/persona_adherence.json
    uv run demo.py --task tasks/research_synthesizer.json
    uv run demo.py --trials 3    # fewer trials for quick testing
"""

import argparse
import os

from dotenv import load_dotenv

from core.task_config import TaskConfig
from core.engine import run_full_pipeline, export_prompt
from core.models import get_model_cost


def banner(text: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="DSPy Prompt Optimizer CLI")
    parser.add_argument("--task", default="tasks/brand_voice_liquid_death.json",
                        help="Path to task config JSON")
    parser.add_argument("--teacher", default="gemini/gemini-3.1-pro-preview",
                        help="Teacher model ID")
    parser.add_argument("--students", nargs="+",
                        default=["gemini/gemini-2.5-flash-lite"],
                        help="Student model IDs")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of evaluation trials")
    parser.add_argument("--threads", type=int, default=50,
                        help="Parallel threads for evaluation")
    args = parser.parse_args()

    task_config = TaskConfig.load(args.task)

    banner("DSPy Prompt Optimizer - CLI")
    print(f"  Task:      {task_config.name}")
    print(f"  Module:    {task_config.module_key}")
    print(f"  Teacher:   {args.teacher}")
    print(f"  Students:  {', '.join(args.students)}")
    print(f"  Examples:  {len(task_config.examples)}")
    print(f"  Trials:    {args.trials}")
    print(f"  Threads:   {args.threads}")

    # Run the full pipeline with progress printing
    def on_progress(step, pct):
        print(f"  [{pct:3d}%] {step}")

    results = run_full_pipeline(
        task_config=task_config,
        teacher_model=args.teacher,
        student_models=args.students,
        num_trials=args.trials,
        threads=args.threads,
        on_progress=on_progress,
    )

    # -- Print results table --
    banner("Results (held-out test set)")

    # Gather columns
    columns = []
    mono = results["monolith"]
    mono_cost = get_model_cost(mono["model"])
    columns.append(("Monolith", mono["model"].split("/")[-1],
                     f"${mono_cost['input_cost']}" if mono_cost else "?",
                     mono["scores"], mono["latency"]))

    for model, data in results["students"].items():
        short = model.split("/")[-1]
        cost = get_model_cost(model)
        cost_str = f"${cost['input_cost']}" if cost else "?"
        columns.append(("Naive", short, cost_str,
                         data["naive"]["scores"], data["naive"]["latency"]))
        columns.append(("DSPy", short, cost_str,
                         data["optimized"]["scores"], data["optimized"]["latency"]))

    # Get metric names from first column
    metric_names = [k for k in columns[0][3].keys()]

    cw = 16
    n_cols = len(columns)

    # Headers
    print(f"\n  {'':.<20}", end="")
    for label, _, _, _, _ in columns:
        print(f" {label:>{cw}}", end="")
    print()

    print(f"  {'model':.<20}", end="")
    for _, short, _, _, _ in columns:
        display = short if len(short) <= cw else short[:cw-2] + ".."
        print(f" {display:>{cw}}", end="")
    print()

    print(f"  {'cost (in/1M)':.<20}", end="")
    for _, _, cost, _, _ in columns:
        print(f" {cost:>{cw}}", end="")
    print()

    print(f"  {'-' * (20 + (cw + 1) * n_cols)}")

    # Metric rows
    for key in metric_names:
        print(f"  {key:.<20}", end="")
        for _, _, _, scores, _ in columns:
            mean = scores[key]["mean"]
            std = scores[key]["std"]
            if std < 0.001:
                print(f" {mean:>{cw}.1%}", end="")
            else:
                cell = f"{mean:.1%}+/-{std:.1%}"
                print(f" {cell:>{cw}}", end="")
        print()

    # Latency row
    print(f"  {'latency (ms)':.<20}", end="")
    for _, _, _, _, lat in columns:
        lat_mean = lat["mean"]
        lat_std = lat["std"]
        if lat_std < 1:
            print(f" {lat_mean:>{cw},.0f}", end="")
        else:
            cell = f"{lat_mean:,.0f}+/-{lat_std:,.0f}"
            print(f" {cell:>{cw}}", end="")
    print()

    # -- AI Summary --
    if results.get("summary"):
        banner("AI-Generated Analysis")
        print(f"\n  {results['summary']}")

    # -- Save prompts --
    banner("Production Prompts")
    for model, data in results["students"].items():
        short = model.split("/")[-1]
        prompt_path = f"production_prompt_{short}.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(data["prompt"])
        print(f"  Saved: {prompt_path}")

    # -- Done --
    banner("DONE")
    print(f"\n  Task: {task_config.name}")
    print(f"  All scores: mean of {args.trials} trials, temperature=0, held-out test set.")
    print()


if __name__ == "__main__":
    main()
