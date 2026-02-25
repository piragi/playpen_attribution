"""Prepare SmolTalk data for LoRA SFT + Bergson attribution.

Loads smol-magpie-ultra, applies the SmolLM2-Instruct chat template with
prompt masking, filters to max_length tokens, then produces three splits:

  train/        SFT training data (Phase 1 model)
  val/          SFT validation data
  attr_pool/    held-out attribution pool (optional; scored in Phase 2)

Attribution query is built separately with rebuild_attr_query.py.

Each split stores:
  input_ids     List[int]   full tokenized conversation
  labels        List[int]   same but prompt tokens = -100
  length        int         number of tokens
  magpie_score  float       Magpie quality score (0.0 if unavailable)

A manifest.json ties the splits together for score.py.

Usage:
    uv run build_sft_data.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import argparse
from datasets import Dataset, load_dataset

from pipeline_common import (
    DEFAULT_BASE_MODEL,
    ensure_hf_home_env,
    get_magpie_score,
    load_tokenizer,
    mask_prompt,
)

ensure_hf_home_env()

CONFIG = {
    "dataset_name": "HuggingFaceTB/smoltalk",
    "dataset_config": "smol-magpie-ultra",  # in-distribution; quality labels preserved
    "base_model": DEFAULT_BASE_MODEL,
    "max_length": 2048,
    # Only stream rows from these categories (set to None for all categories).
    "category_filter": {"math", "data-analysis"},
    # Split sizes (rows, not tokens)
    "train_size": 5_000,
    "val_size": 500,
    # Set to 0 to score the train split itself (score_pool → train in manifest).
    "attr_pool_size": 0,
    "output_dir": "runs/smoltalk_v4",
    "seed": 42,
}

SMOKE_CONFIG = {
    **CONFIG,
    "train_size": 64,
    "val_size": 20,
    "attr_pool_size": 50,
    "output_dir": "runs/smoke_test",
}


# ---------------------------------------------------------------------------
# Build SmolTalk splits
# ---------------------------------------------------------------------------

def build_smoltalk_splits(tokenizer, cfg: dict) -> tuple[Dataset, Dataset, Dataset, int]:
    """Return (train, val, attr_pool) datasets and the raw row count consumed."""
    print(f"Loading {cfg['dataset_name']}/{cfg['dataset_config']} (streaming) ...")
    raw = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split="train", streaming=True)

    needed = cfg["train_size"] + cfg["val_size"] + cfg["attr_pool_size"]
    gather_target = needed * 2  # gather 2× so shuffle has room
    category_filter = cfg.get("category_filter")
    rows: list[dict] = []
    raw_rows_seen = 0

    for row in raw:
        raw_rows_seen += 1
        if category_filter and row.get("category", "") not in category_filter:
            continue
        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok = mask_prompt(messages, tokenizer, cfg["max_length"])
        if tok is None:
            continue
        tok["magpie_score"] = get_magpie_score(row)
        tok["quality"] = row.get("quality", "")
        tok["category"] = row.get("category", "")
        rows.append(tok)
        if len(rows) % 5000 == 0:
            print(f"  processed {len(rows):,} valid rows...", flush=True)
        if len(rows) >= gather_target:
            break

    print(f"  valid rows after filtering: {len(rows):,}")
    if len(rows) < needed:
        raise RuntimeError(
            f"Not enough valid rows: need {needed}, got {len(rows)}. "
            "Try reducing split sizes or max_length."
        )

    ds = Dataset.from_list(rows).shuffle(seed=cfg["seed"])
    train_ds  = ds.select(range(cfg["train_size"]))
    val_ds    = ds.select(range(cfg["train_size"], cfg["train_size"] + cfg["val_size"]))
    pool_ds   = ds.select(range(cfg["train_size"] + cfg["val_size"],
                                cfg["train_size"] + cfg["val_size"] + cfg["attr_pool_size"]))
    return train_ds, val_ds, pool_ds, raw_rows_seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny sizes for end-to-end pipeline validation.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = dict(SMOKE_CONFIG if args.smoke_test else CONFIG)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["base_model"])
    train_ds, val_ds, pool_ds, raw_rows_consumed = build_smoltalk_splits(tokenizer, cfg)

    splits_info: dict[str, dict] = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("attr_pool", pool_ds)]:
        if len(ds) == 0:
            print(f"  {name}: 0 rows — skipped")
            continue
        path = out / "data" / name
        if path.exists():
            shutil.rmtree(path)
        ds.save_to_disk(str(path))
        total_tokens = int(sum(ds["length"]))
        splits_info[name] = {"path": str(path), "rows": len(ds), "total_tokens": total_tokens}
        print(f"  {name}: {len(ds):,} rows, {total_tokens / 1e6:.1f}M tokens → {path}")

    # When attr_pool_size=0, score the train split directly.
    score_pool_info = splits_info.get("attr_pool", splits_info["train"])
    manifest = {
        "base_model": cfg["base_model"],
        "max_length": cfg["max_length"],
        "dataset": cfg["dataset_name"],
        "dataset_config": cfg["dataset_config"],
        "raw_rows_consumed": raw_rows_consumed,
        "splits": {
            "score_pool": score_pool_info,
            "train": splits_info["train"],
            "val": splits_info["val"],
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to: {manifest_path}")
    run_dir = str(out)
    print("\nNext steps:")
    print(f"  1. uv run finetune.py --train-data {run_dir}/data/train --val-data {run_dir}/data/val --output-dir {run_dir}/adapter")
    print(f"  2. uv run rebuild_attr_query.py")
    print(f"  3. uv run score.py --manifest {run_dir}/manifest.json --adapter-path {run_dir}/adapter")


if __name__ == "__main__":
    main()
