from __future__ import annotations

"""Evaluation wrapper using lm-eval harness.

Works for both base (non-instruct) and SFT-adapted models:
  - MC tasks (arc_challenge, arc_easy, hellaswag, winogrande, piqa, mmlu)
    use log-probability scoring — no instruction following required.
  - Generation tasks (gsm8k_cot_zeroshot) require some instruction following;
    best reserved for models after SFT.

Typical usage:

  # Phase-1: score the pretrained base model on MC benchmarks
  uv run eval_harness.py \\
    --base-model runs/pretrain_270m_v1 \\
    --output-json runs/pretrain_270m_v1/eval_harness.json

  # Phase-3: score SFT-adapted model (adapter on top of base)
  uv run eval_harness.py \\
    --base-model google/gemma-3-1b-it \\
    --adapter-path runs/gsm8k_1b_full_v1/adapter \\
    --tasks arc_challenge,arc_easy,hellaswag,winogrande,piqa,gsm8k_cot_zeroshot \\
    --output-json runs/gsm8k_1b_full_v1/eval_harness.json
"""

import argparse
import json
from pathlib import Path

import lm_eval
import torch

# Default benchmark suite for pretraining evaluation.
# All tasks use log-prob scoring — no generation needed.
DEFAULT_TASKS = "arc_challenge,arc_easy,hellaswag,winogrande,piqa"

# Metric keys lm-eval uses per task (in priority order).
_METRIC_PRIORITY = [
    "acc_norm,none",  # length-normalised accuracy (HellaSwag, ARC)
    "acc,none",       # plain accuracy (PIQA, WinoGrande, MMLU)
    "exact_match,flexible-extract",  # GSM8K
    "exact_match,strict-match",
]


def best_metric(task_result: dict) -> tuple[str, float] | tuple[None, None]:
    """Return the (metric_name, value) pair we care most about for a task."""
    for key in _METRIC_PRIORITY:
        if key in task_result:
            return key, task_result[key]
    # Fall back to first numeric value found
    for k, v in task_result.items():
        if isinstance(v, float):
            return k, v
    return None, None


def build_model_args(args: argparse.Namespace) -> str:
    parts = [f"pretrained={args.base_model}"]
    if args.adapter_path:
        parts.append(f"peft={args.adapter_path}")
    # Do NOT pass torch_dtype/dtype here — lm-eval's HFLM.__init__ has its own
    # dtype="auto" parameter which (after patching huggingface.py) correctly
    # maps to torch_dtype= in from_pretrained. Passing it again causes a
    # "multiple values for keyword argument 'torch_dtype'" error.
    parts.append("attn_implementation=eager")
    return ",".join(parts)


def run(args: argparse.Namespace) -> None:
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_args = build_model_args(args)
    num_fewshot = args.num_fewshot if args.num_fewshot >= 0 else None
    limit = args.limit if args.limit > 0 else None

    print(f"model : {args.base_model}")
    if args.adapter_path:
        print(f"adapter: {args.adapter_path}")
    print(f"tasks  : {tasks}")
    print(f"fewshot: {num_fewshot!r} (None = task default)")
    print(f"limit  : {limit!r}")
    print(f"model_args: {model_args}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = lm_eval.simple_evaluate(  # type: ignore[call-arg]
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        device=device,
        batch_size=args.batch_size,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned None — check task names and model args.")

    # Build a flat summary for easy comparison across runs.
    summary: dict[str, dict] = {}
    for task in tasks:
        task_result = results["results"].get(task, {})
        metric, value = best_metric(task_result)
        if metric is not None:
            summary[task] = {"metric": metric, "value": value}

    output = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "summary": summary,
        "results": results["results"],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    # Print summary table.
    print("\n--- results ---")
    col = max((len(t) for t in tasks), default=10)
    for task in tasks:
        if task in summary:
            s = summary[task]
            print(f"  {task:<{col}}  {s['metric']:<40}  {s['value']:.4f}")
        else:
            print(f"  {task:<{col}}  (no result)")
    print(f"\nsaved: {out_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lm-eval harness benchmarks on a base or SFT-adapted model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-270m",
        help="HuggingFace model ID or local path (pretrained checkpoint or hub model).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional PEFT adapter directory to load on top of --base-model.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=-1,
        help="Few-shot examples per task. -1 = use each task's default. "
             "0 = force zero-shot for all tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit evaluation to this many samples per task (0 = full eval).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="runs/eval_harness.json",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
