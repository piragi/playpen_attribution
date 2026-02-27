"""Evaluation wrapper using lm-eval harness.

Works for both base and LoRA SFT-adapted models. MC tasks use log-prob scoring.
For SFT models, pass --apply-chat-template to wrap prompts in the model's chat
template before scoring — this is required for accurate instruct model evaluation.

Typical usage:

  # Base model (no chat template)
  uv run eval_harness.py \\
    --base-model HuggingFaceTB/SmolLM2-1.7B \\
    --output-json runs/base_eval.json

  # LoRA SFT model (chat template required)
  uv run eval_harness.py \\
    --base-model HuggingFaceTB/SmolLM2-1.7B \\
    --adapter-path runs/smoltalk_v4/adapter \\
    --apply-chat-template \\
    --output-json runs/smoltalk_v4/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lm_eval
import torch
import transformers

from pipeline_common import (
    DEFAULT_BASE_MODEL,
    ensure_hf_home_env,
)

ensure_hf_home_env()

# lm-eval may pass dtype= to from_pretrained while some model classes only
# accept torch_dtype=. Normalize this at runtime so eval works consistently.
_orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def _from_pretrained_compat(*args, dtype=None, torch_dtype=None, **kwargs):
    return _orig_from_pretrained(*args, torch_dtype=torch_dtype if torch_dtype is not None else dtype, **kwargs)

transformers.AutoModelForCausalLM.from_pretrained = _from_pretrained_compat

# Default benchmark suite for SFT evaluation.
# Log-prob MC: arc_challenge, arc_easy, hellaswag, winogrande (, mmlu — commented: 57 subtasks, very slow)
# Generation: ifeval (rule-based instruction following), gsm8k (math reasoning, 5-shot)
DEFAULT_TASKS = "arc_challenge,arc_easy,hellaswag,winogrande,ifeval,gsm8k"  #,mmlu"
_CODE_EVAL_TASK_HINTS = ("humaneval", "mbpp", "code_eval")

# Metric keys lm-eval uses per task (in priority order).
_METRIC_PRIORITY = [
    "acc_norm,none",                    # length-normalised accuracy (HellaSwag, ARC)
    "acc,none",                         # plain accuracy (WinoGrande, MMLU)
    "prompt_level_strict_acc,none",     # IFEval
    "exact_match,flexible-extract",     # GSM8K
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


def maybe_enable_code_eval(tasks: list[str]) -> bool:
    """Enable code-eval safety switches only when code tasks are requested."""
    lowered = [t.lower() for t in tasks]
    needs_code_eval = any(
        any(hint in task for hint in _CODE_EVAL_TASK_HINTS)
        for task in lowered
    )
    if needs_code_eval and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("set HF_ALLOW_CODE_EVAL=1 (code evaluation tasks detected)")
    return needs_code_eval


def build_model_args(args: argparse.Namespace) -> str:
    parts = [f"pretrained={args.base_model}"]
    if args.adapter_path:
        parts.append(f"peft={args.adapter_path}")
        # Use the IT tokenizer saved alongside the adapter (has chat_template).
        parts.append(f"tokenizer={args.adapter_path}")
    # Do NOT pass torch_dtype/dtype here — lm-eval's HFLM.__init__ has its own
    # dtype="auto" parameter which (after patching huggingface.py) correctly
    # maps to torch_dtype= in from_pretrained. Passing it again causes a
    # "multiple values for keyword argument 'torch_dtype'" error.
    parts.append("attn_implementation=sdpa")
    return ",".join(parts)


def run(cfg: dict) -> None:
    """Run lm-eval from a config dict.

    Required keys: base_model, adapter_path, output_json
    Optional keys: tasks, eval_batch_size, apply_chat_template, num_fewshot, limit
    """
    args = argparse.Namespace(
        base_model=cfg["base_model"],
        adapter_path=cfg.get("adapter_path"),
        tasks=cfg.get("eval_tasks", DEFAULT_TASKS),
        num_fewshot=cfg.get("num_fewshot", -1),
        limit=cfg.get("limit", 0),
        batch_size=cfg.get("eval_batch_size", "8"),
        apply_chat_template=cfg.get("apply_chat_template", True),
        output_json=cfg["output_json"],
    )
    run_from_args(args)


def run_from_args(args: argparse.Namespace) -> None:
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    needs_code_eval = maybe_enable_code_eval(tasks)
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
    extra = {"apply_chat_template": True} if args.apply_chat_template else {}

    results = lm_eval.simple_evaluate(  # type: ignore[call-arg]
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        device=device,
        batch_size=args.batch_size,
        confirm_run_unsafe_code=needs_code_eval,
        **extra,
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
        "apply_chat_template": args.apply_chat_template,
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
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace base model ID or local path.",
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
        type=str,
        default="8",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        default=False,
        help="Wrap benchmark prompts in the model's chat template before scoring. "
             "Required for accurate instruct/SFT model evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="runs/eval_harness.json",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
