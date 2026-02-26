"""Run the end-to-end attribution pipeline.

One CONFIG dict controls every step. Comment out steps you've already run
and adjust the config freely — each step reads only the keys it needs.

Flow:
  1) build_sft_data
  2) finetune (base adapter)
  3) rebuild_attr_query
  4) score
  5) probe
  6) generate_continued_dataset
  7) finetune (all continuation arms)
  8) eval_harness (all continuation arms)

Usage:
    uv run run_experiments.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

import build_sft_data
import eval_harness
import finetune
import generate_continued_dataset
import probe
import rebuild_attr_query
import score


CONFIG = {
    # ── Shared ───────────────────────────────────────────────────────────────
    "run_dir": "runs/smoltalk_v5",
    "base_model": "HuggingFaceTB/SmolLM2-1.7B",
    "seed": 42,
    "exp_tag": "",  # optional suffix for continuation outputs (e.g. "ms1200_lr3e-4")

    # ── build_sft_data ───────────────────────────────────────────────────────
    "dataset_name": "HuggingFaceTB/smoltalk",
    "dataset_config": "smol-magpie-ultra",
    "max_length": 2048,
    "category_filter": {"math", "data-analysis"},
    "train_size": 5_000,
    "val_size": 500,
    "attr_pool_size": 0,
    "smoke_test": False,

    # ── finetune ─────────────────────────────────────────────────────────────
    "num_train_epochs": 2,
    "max_steps": 300,
    "learning_rate": 3e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "logging_steps": 20,
    "save_steps": 500,
    "save_total_limit": 2,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",

    # ── rebuild_attr_query ───────────────────────────────────────────────────
    "query_category": None,          # None | "math" | "data-analysis" | "all"
    "query_smol_size": 4096,
    "query_quality_min": {"good", "excellent"},

    # ── score ────────────────────────────────────────────────────────────────
    "score_output_dir": "runs/smoltalk_v5/scores_math_da",
    "query_split": "attr_query",
    "pool_split": "score_pool",
    "projection_dim": 32,
    "token_batch_size": 2048,
    "preconditioning_mode": "query",
    "mixing_coefficient": 0.99,
    "score_mode": "mean",
    "unit_normalize": True,
    "loss_reduction": "mean",
    "label_smoothing": 0.0,

    # ── probe ────────────────────────────────────────────────────────────────
    # scores_dir defaults to score_output_dir when not set per-probe
    "probes": [{"name": "math_da"}],
    "extraction_layer": 17,
    "probe_adapter_path": None,      # None → use {run_dir}/adapter; set to override
    "ridge_alpha": 100.0,
    "val_frac": 0.20,
    "probe_batch_size": 64,

    # ── generate_continued_dataset ───────────────────────────────────────────
    "pool_size": 30_000,
    "quality_size": 10_000,
    "random_size": 10_000,
    "gen_batch_size": 64,
    "gen_adapter_path": None,        # None → use {run_dir}/adapter; set to override

    # ── finetune / eval continuation arms (steps 6–8) ────────────────────────
    # Set finetune_seed to re-run steps 6–8 with different randomness (random arm
    # selection + LoRA training) without re-running steps 1–5. Adapters/evals are
    # tagged _s{finetune_seed} so previous results are never overwritten.
    "finetune_seed": [43],           # None → use seed; set e.g. 43, 44, 45 for multi-seed runs
    "skip_half_arms": False,         # skip *_50pct continuation arms
    "token_matched_controls_only": False,  # only random* + label_good_excellent* arms
    "arm_allowlist": None,           # e.g. {"quality_math_da", "random"}

    # ── eval_harness ─────────────────────────────────────────────────────────
    "run_eval": True,
    "eval_tasks": "arc_challenge,arc_easy,hellaswag,winogrande,ifeval,gsm8k",
    "eval_batch_size": "32",
}

SMOKE_CONFIG = {
    **CONFIG,
    "run_dir": "runs/smoke_test",
    "train_size": 64,
    "val_size": 20,
    "attr_pool_size": 50,
    "query_smol_size": 64,
    "pool_size": 200,
    "quality_size": 50,
    "random_size": 50,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "probe_batch_size": 8,
    "gen_batch_size": 8,
    "run_eval": False,
}


def _clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    cfg = dict(SMOKE_CONFIG if "--smoke" in sys.argv else CONFIG)
    if "--token-controls-only" in sys.argv:
        cfg["token_matched_controls_only"] = True
    run_dir = Path(cfg["run_dir"])

    # Default embedding extraction to the trained base adapter.
    base_adapter = str(run_dir / "adapter")
    if cfg.get("probe_adapter_path") is None:
        cfg["probe_adapter_path"] = base_adapter
    if cfg.get("gen_adapter_path") is None:
        cfg["gen_adapter_path"] = base_adapter

    # print("\n=== 1) build_sft_data ===")
    # build_sft_data.run(cfg)
# 
    # print("\n=== 2) finetune base adapter ===")
    # finetune.run(cfg)
    # _clear_gpu()
# 
    # print("\n=== 3) rebuild_attr_query ===")
    # rebuild_attr_query.run(cfg)
# 
    # print("\n=== 4) score ===")
    # score.run(cfg)
    # _clear_gpu()
# 
    # print("\n=== 5) probe ===")
    # probe.run(cfg)
    # _clear_gpu()

    seed_cfg = cfg.get("finetune_seed")
    finetune_seeds = (
        seed_cfg if isinstance(seed_cfg, list)
        else [cfg["seed"]] if seed_cfg is None
        else [seed_cfg]
    )
    exp_tag = str(cfg.get("exp_tag", "")).strip()
    exp_suffix = f"_{exp_tag}" if exp_tag else ""
    val_data = json.loads((run_dir / "manifest.json").read_text())["splits"]["val"]["path"]

    for finetune_seed in finetune_seeds:
        seed_tag = f"_s{finetune_seed}" if seed_cfg is not None else ""

        print(f"\n=== 6) generate_continued_dataset (seed={finetune_seed}) ===")
        generate_continued_dataset.run({**cfg, "seed": finetune_seed})
        _clear_gpu()

        print(f"\n=== 7) finetune continuation arms (seed={finetune_seed}) ===")
        cont_manifest = json.loads((run_dir / "continuation" / "continuation_manifest.json").read_text())
        arm_items = list(cont_manifest["arms"].items())
        if cfg.get("skip_half_arms"):
            arm_items = [(n, i) for n, i in arm_items if not n.endswith("_50pct")]
        if cfg.get("token_matched_controls_only"):
            arm_items = [
                (n, i) for n, i in arm_items
                if n.startswith("random") or n.startswith("label_good_excellent")
            ]
        if cfg.get("arm_allowlist"):
            allow = set(cfg["arm_allowlist"])
            arm_items = [(n, i) for n, i in arm_items if n in allow]

        for arm_name, arm_info in arm_items:
            print(f"\n  --- arm: {arm_name} ---")
            finetune.run({
                **cfg,
                "seed": finetune_seed,
                "train_data": arm_info["path"],
                "val_data": val_data,
                "output_dir": str(run_dir / f"adapter_{arm_name}{exp_suffix}{seed_tag}"),
            })
            _clear_gpu()

        print(f"\n=== 8) eval continuation arms (seed={finetune_seed}) ===")
        if cfg["run_eval"]:
            eval_dir = run_dir / "evals"
            eval_dir.mkdir(parents=True, exist_ok=True)
            for arm_name, _ in arm_items:
                eval_harness.run({
                    **cfg,
                    "adapter_path": str(run_dir / f"adapter_{arm_name}{exp_suffix}{seed_tag}"),
                    "output_json": str(eval_dir / f"{arm_name}{exp_suffix}{seed_tag}.json"),
                    "apply_chat_template": True,
                })

    print("\nDone.")


if __name__ == "__main__":
    main()
