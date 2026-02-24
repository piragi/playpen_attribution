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
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from pipeline_common import DEFAULT_BASE_MODEL, ensure_hf_home_env, infer_instruct_tokenizer_model

ensure_hf_home_env()

_BASE_MODEL = DEFAULT_BASE_MODEL

CONFIG = {
    "dataset_name": "HuggingFaceTB/smoltalk",
    "dataset_config": "smol-magpie-ultra",  # in-distribution; quality labels preserved
    "base_model": _BASE_MODEL,  # model weights for training
    "tokenizer_model": infer_instruct_tokenizer_model(_BASE_MODEL),  # tokenizer for chat template
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> dict | None:
    """Tokenize a conversation, masking all non-assistant tokens in labels.

    Returns {"input_ids": [...], "labels": [...], "length": int}
    or None if the result exceeds max_length or is trivially short.
    """
    # Full conversation tokenized
    full_ids: list[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if len(full_ids) > max_length:
        return None

    # Build labels: start with all -100, then fill in assistant turns
    labels = [-100] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # Prefix up to (but not including) this assistant turn,
        # with add_generation_prompt=True so we find where the model starts.
        prefix_ids: list[int] = tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True
        )
        start = len(prefix_ids)

        # Prefix up to (and including) this assistant turn
        up_to_ids: list[int] = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )
        end = len(up_to_ids)
        end = min(end, len(full_ids))

        labels[start:end] = full_ids[start:end]

    # Require at least one supervised token
    if all(l == -100 for l in labels):
        return None

    return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


def _get_magpie_score(row: dict) -> float:
    """Best-effort extraction of Magpie quality score from a SmolTalk row."""
    for key in ("score", "quality_score", "magpie_score", "quality"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


# ---------------------------------------------------------------------------
# Build SmolTalk splits
# ---------------------------------------------------------------------------

def build_smoltalk_splits(tokenizer: Any, cfg: dict) -> tuple[Dataset, Dataset, Dataset, int]:
    """Return (train, val, attr_pool) datasets."""
    print(f"Loading {cfg['dataset_name']}/{cfg['dataset_config']} (streaming) ...")
    raw = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split="train", streaming=True)

    rng_seed = cfg["seed"]
    needed = cfg["train_size"] + cfg["val_size"] + cfg["attr_pool_size"]

    # Gather 2× needed so shuffle has room (keeps smoke-test fast too).
    gather_target = needed * 2
    rows: list[dict] = []
    raw_rows_seen = 0
    category_filter = cfg.get("category_filter")
    for row in raw:
        raw_rows_seen += 1
        if category_filter and row.get("category", "") not in category_filter:
            continue
        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, cfg["max_length"])
        if tok_out is None:
            continue
        tok_out["magpie_score"] = _get_magpie_score(row)
        # Store quality/category when available (smol-magpie-ultra preserves these labels).
        tok_out["quality"] = row.get("quality", "")
        tok_out["category"] = row.get("category", "")
        rows.append(tok_out)
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

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=rng_seed)

    train_ds = ds.select(range(cfg["train_size"]))
    val_ds = ds.select(range(cfg["train_size"], cfg["train_size"] + cfg["val_size"]))
    attr_pool_ds = ds.select(
        range(cfg["train_size"] + cfg["val_size"],
              cfg["train_size"] + cfg["val_size"] + cfg["attr_pool_size"])
    )

    return train_ds, val_ds, attr_pool_ds, raw_rows_seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SMOKE_CONFIG = {
    **CONFIG,
    "train_size": 64,
    "val_size": 20,
    "attr_pool_size": 50,
    "output_dir": "runs/smoke_test",
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny sizes for end-to-end pipeline validation.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    args = parser.parse_args()

    cfg = dict(SMOKE_CONFIG if args.smoke_test else CONFIG)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds, attr_pool_ds, raw_rows_consumed = build_smoltalk_splits(tokenizer, cfg)

    # Save splits
    splits_info: dict[str, dict] = {}
    for name, ds in [
        ("train", train_ds),
        ("val", val_ds),
        ("attr_pool", attr_pool_ds),
    ]:
        if len(ds) == 0:
            print(f"  {name}: 0 rows — skipped")
            continue
        path = out / "data" / name
        if path.exists():
            import shutil
            shutil.rmtree(path)
        ds.save_to_disk(str(path))
        total_tokens = int(sum(ds["length"]))
        splits_info[name] = {
            "path": str(path),
            "rows": len(ds),
            "total_tokens": total_tokens,
        }
        print(f"  {name}: {len(ds):,} rows, {total_tokens / 1e6:.1f}M tokens → {path}")

    # Write manifest for score.py
    # When attr_pool_size=0, score the train split directly.
    score_pool_info = splits_info.get("attr_pool", splits_info["train"])
    manifest = {
        "base_model": cfg["base_model"],
        "tokenizer_model": cfg["tokenizer_model"],
        "max_length": cfg["max_length"],
        "dataset": cfg["dataset_name"],
        "dataset_config": cfg["dataset_config"],
        "raw_rows_consumed": raw_rows_consumed,  # used by generate_continued_dataset.py to skip ahead
        "splits": {
            # score.py uses score_pool + attr_query
            "score_pool": score_pool_info,
            # full train/val for reference
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
    print(f"  2. uv run rebuild_attr_query.py   # builds {run_dir}/data/attr_query from smol-magpie-ultra")
    print(f"  3. uv run score.py --manifest {run_dir}/manifest.json --adapter-path {run_dir}/adapter")


if __name__ == "__main__":
    main()
