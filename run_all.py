"""
run_all.py
----------
Orchestrates multiple experiment runs and produces a comparison table.

1. Define your experiment matrix via MODELS_TO_RUN and BASE_CONFIGS.
2. Run:  python run_all.py
3. Find results in ./experiments/<model_tag>__<run_name>/

After all runs finish, a comparison table is printed and saved to
./experiments/comparison.csv.

Flags:
  --only-benchmark   Skip inference, only (re)run benchmark on existing results
  --only-inference   Skip benchmark step
  --models           Comma-separated subset of model tags to run
                     e.g. --models llama31_8b,qwen3_8b
"""

from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

from experiment_config import RunConfig, PROMPT_VERSIONS
from run_experiment import run as run_inference
from benchmark import benchmark as run_benchmark


# ---------------------------------------------------------------------------
# MODEL REGISTRY
# Add or remove models here. Tags are used as directory prefixes.
# ---------------------------------------------------------------------------

MODEL_IDS: dict[str, str] = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25_7b":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_8b":   "Qwen/Qwen3-8B",
}

# Contexto máximo por modelo — respeta max_position_embeddings de cada uno
MODEL_MAX_CONTEXT: dict[str, int] = {
    "llama31_8b": 45000,   # soporta 128k
    "qwen25_7b":  30000,   # max_position_embeddings=32768, dejamos margen
    "qwen3_8b":   30000,   # mismo caso, 32k nativo
}


# Qwen3 needs /no_think to suppress chain-of-thought and return JSON directly.
# We register the variant here so experiment_config validates it correctly.
PROMPT_VERSIONS["v3_full_qwen3"] = PROMPT_VERSIONS["v3_full"] + "\n/no_think"

# Map each model tag to the prompt version it should use for "v3_full" runs.
# All other prompt versions are model-agnostic and shared across models.
MODEL_PROMPT_OVERRIDE: dict[str, str] = {
    "qwen3_8b": "v3_full_qwen3",   # disables thinking mode
}

# Models to include in this run — edit to run a subset without deleting configs.
MODELS_TO_RUN: list[str] = [
    "llama31_8b",
    "qwen25_7b",
    "qwen3_8b",
]


# ---------------------------------------------------------------------------
# BASE EXPERIMENT CONFIGS — model-agnostic
# Add new prompt/param variants here. model_id and output_dir are set below.
# ---------------------------------------------------------------------------

BASE_CONFIGS: list[dict] = [

    # ── Baseline: no few-shot, minimal prompt ─────────────────────────────
    dict(
        name="v1_no_fewshot",
        description="Baseline — minimal prompt, no few-shot examples",
        prompt_version="v1_baseline",
        use_few_shot=False,
        temperature=0.0,
        overlap_chars=800,
        max_context_tokens=45000,
    ),

    # ── Date rules only ───────────────────────────────────────────────────
    dict(
        name="v2_date_rules",
        description="Adds explicit date extraction rules, no few-shot",
        prompt_version="v2_with_date_rules",
        use_few_shot=False,
        temperature=0.0,
        overlap_chars=800,
        max_context_tokens=45000,
    ),

    # ── Full prompt + few-shot (production config) ────────────────────────
    dict(
        name="v3_full_fewshot",
        description="Full rules + few-shot examples — production config",
        prompt_version="v3_full",        # overridden for qwen3 via MODEL_PROMPT_OVERRIDE
        use_few_shot=True,
        temperature=0.0,
        overlap_chars=800,
        max_context_tokens=45000,
    ),

    # ── Full + few-shot, larger overlap ──────────────────────────────────
    dict(
        name="v3_overlap1600",
        description="Full + few-shot, overlap doubled to 1600 chars",
        prompt_version="v3_full",
        use_few_shot=True,
        temperature=0.0,
        overlap_chars=1600,
        max_context_tokens=45000,
    ),

    # ── Slight temperature to reduce over-conservatism on binary fields ───
    dict(
        name="v3_temp02",
        description="Full + few-shot, temperature=0.2",
        prompt_version="v3_full",
        use_few_shot=True,
        temperature=0.2,
        overlap_chars=800,
        max_context_tokens=45000,
    ),
]


# ---------------------------------------------------------------------------
# Build experiment matrix
# Iterates models first so all experiments for a model run together,
# minimizing VRAM reloads between runs.
# ---------------------------------------------------------------------------

def _resolve_prompt(base_prompt: str, model_tag: str) -> str:
    """
    Returns the model-specific prompt override for v3_full variants,
    or the original prompt_version for everything else.
    """
    if base_prompt == "v3_full" and model_tag in MODEL_PROMPT_OVERRIDE:
        return MODEL_PROMPT_OVERRIDE[model_tag]
    return base_prompt

EXPERIMENTS: list[RunConfig] = [
    RunConfig(
        **{
            **cfg,
            "name":        f"{model_tag}__{cfg['name']}",
            "description": f"[{model_tag}] {cfg['description']}",
            "prompt_version": _resolve_prompt(cfg["prompt_version"], model_tag),
            "model_id":    MODEL_IDS[model_tag],
            "output_dir":  f"./experiments/{model_tag}__{cfg['name']}/results",
            "max_context_tokens": MODEL_MAX_CONTEXT[model_tag],
        }
    )
    for model_tag in MODELS_TO_RUN
    for cfg in BASE_CONFIGS
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELDS = [
    "parties", "agreement_date", "effective_date", "expiration_date",
    "renewal_term", "notice_period_to_terminate_renewal", "governing_law",
    "anti_assignment", "audit_rights", "cap_on_liability",
    "termination_for_convenience", "liquidated_damages",
]


def load_benchmark_summary(results_dir: str) -> dict | None:
    """Read benchmark_detailed.csv and compute per-field accuracy + P/R/F1."""
    detailed_csv = Path(results_dir, "benchmark_detailed.csv")
    if not detailed_csv.exists():
        return None

    _ABSENT = {"not mentioned", "none", "null", ""}
    per_field: dict[str, dict] = {
        f: {"correct": 0, "total": 0,
            "tp": 0, "fp": 0, "fn": 0,
            "present_correct": 0, "present_total": 0}
        for f in FIELDS
    }
    unique_files: set[str] = set()

    with detailed_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row["field"]
            unique_files.add(row["file_key"])
            if field not in per_field:
                continue
            matched = row["match"] == "1"
            per_field[field]["total"] += 1
            if matched:
                per_field[field]["correct"] += 1

            gt   = (row.get("gt_norm")   or "").strip().lower()
            pred = (row.get("pred_norm") or "").strip().lower()
            gt_present   = gt   not in _ABSENT
            pred_present = pred not in _ABSENT

            if gt_present:
                per_field[field]["present_total"] += 1
                if matched:
                    per_field[field]["present_correct"] += 1
                    per_field[field]["tp"] += 1
                else:
                    per_field[field]["fn"] += 1
            elif pred_present:
                # model extracted a value where ground truth is absent → false positive
                per_field[field]["fp"] += 1
            # else: both absent → TN, not counted in P/R/F1

    summary = {}
    acc_values, prec_values, rec_values, f1_values = [], [], [], []
    for field, counts in per_field.items():
        # Accuracy: over all predictions (including absent-absent matches)
        acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
        summary[f"acc_{field}"] = round(acc, 4)
        acc_values.append(acc)

        # Precision / Recall / F1: only on cases where gt or pred is present
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        summary[f"prec_{field}"] = round(prec, 4)
        summary[f"rec_{field}"]  = round(rec,  4)
        summary[f"f1_{field}"]   = round(f1,   4)
        prec_values.append(prec)
        rec_values.append(rec)
        f1_values.append(f1)

    present_correct = sum(v["present_correct"] for v in per_field.values())
    present_total   = sum(v["present_total"]   for v in per_field.values())
    summary["micro_acc"]       = round(present_correct / present_total if present_total else 0.0, 4)
    summary["macro_acc"]       = round(sum(acc_values)  / len(acc_values),  4)
    summary["macro_precision"] = round(sum(prec_values) / len(prec_values), 4)
    summary["macro_recall"]    = round(sum(rec_values)  / len(rec_values),  4)
    summary["macro_f1"]        = round(sum(f1_values)   / len(f1_values),   4)
    summary["files_evaluated"] = len(unique_files)

    inference_summary_path = Path(results_dir) / "inference_summary.json"
    try:
        with inference_summary_path.open(encoding="utf-8") as f:
            inf = json.load(f)
        summary["elapsed_seconds"] = inf.get("elapsed_seconds", "")
    except (FileNotFoundError, json.JSONDecodeError):
        summary["elapsed_seconds"] = ""

    return summary


def print_comparison_table(rows: list[dict]) -> None:
    if not rows:
        print("No benchmark results to compare.")
        return

    col_width = 22
    header_cols = ["run_name", "files_evaluated", "macro_acc", "micro_acc", "elapsed_seconds"] + [f"acc_{f}" for f in FIELDS]
    header = "  ".join(c[:col_width].ljust(col_width) for c in header_cols)
    print("\n" + "=" * len(header))
    print("EXPERIMENT COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(col_width) for c in header_cols)
        print(line)
    print("=" * len(header))


def save_comparison_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n📄 Comparison saved to: {path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main(
    run_inference_step: bool = True,
    run_benchmark_step: bool = True,
    experiments: list[RunConfig] | None = None,
) -> None:
    if experiments is None:
        experiments = EXPERIMENTS

    comparison_rows: list[dict] = []
    current_model = None

    for config in experiments:
        # ── Log model transitions to track VRAM reloads ───────────────────
        if config.model_id != current_model:
            current_model = config.model_id
            print(f"\n{'='*60}")
            print(f"🔄 Loading model: {current_model}")
            print(f"{'='*60}")

        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT: {config.name}")
        print(f"# {config.description}")
        print(f"{'#'*60}")

        # ── Inference ─────────────────────────────────────────────────────
        if run_inference_step:
            try:
                run_inference(config)
            except Exception as e:
                print(f"❌ Inference failed for {config.name}: {e}")
                continue

        # ── Benchmark ─────────────────────────────────────────────────────
        if run_benchmark_step:
            detailed_csv = Path(config.output_dir, "benchmark_detailed.csv")
            try:
                run_benchmark(
                    results_dir=Path(config.output_dir),
                    ground_truth_csv=Path(config.ground_truth_csv),
                    output_csv=detailed_csv,
                )
            except Exception as e:
                print(f"❌ Benchmark failed for {config.name}: {e}")
                continue

        # ── Collect metrics ───────────────────────────────────────────────
        metrics = load_benchmark_summary(config.output_dir)
        if metrics:
            row = {"run_name": config.name, **metrics}
            comparison_rows.append(row)
            Path(config.output_dir, "benchmark_summary.json").write_text(
                json.dumps(row, indent=2)
            )

    # ── Final comparison ──────────────────────────────────────────────────
    print_comparison_table(comparison_rows)
    save_comparison_csv(comparison_rows, "./experiments/comparison.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all experiments and compare")
    parser.add_argument(
        "--only-benchmark",
        action="store_true",
        help="Skip inference, only (re)run benchmark on existing results",
    )
    parser.add_argument(
        "--only-inference",
        action="store_true",
        help="Skip benchmark step",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model tags to run (e.g. llama31_8b,qwen3_8b). "
             f"Available: {', '.join(MODEL_IDS.keys())}",
    )
    args = parser.parse_args()

    # Filter experiments if --models flag was passed
    experiments_to_run = EXPERIMENTS
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        unknown = requested - set(MODEL_IDS.keys())
        if unknown:
            print(f"❌ Unknown model tags: {unknown}. Available: {list(MODEL_IDS.keys())}")
            sys.exit(1)
        experiments_to_run = [e for e in EXPERIMENTS if any(e.name.startswith(m) for m in requested)]
        print(f"🎯 Running {len(experiments_to_run)} experiments for models: {requested}")

    main(
        run_inference_step=not args.only_benchmark,
        run_benchmark_step=not args.only_inference,
        experiments=experiments_to_run,
    )
