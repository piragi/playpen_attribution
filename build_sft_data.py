"""Prepare SmolTalk data for LoRA SFT + Bergson attribution.

Loads smol-smoltalk, applies Gemma chat template with prompt masking,
filters to max_length tokens, then produces three splits:

  train/        SFT training data (Phase 1 model)
  val/          SFT validation data
  attr_pool/    held-out attribution pool (scored in Phase 2)

Plus a benchmark attribution query set:
  attr_query/   ARC-Challenge + WinoGrande formatted as instruct examples

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

CONFIG = {
    "dataset_name": "HuggingFaceTB/smoltalk",
    "dataset_config": "smol-magpie-ultra",  # in-distribution; quality labels preserved
    "base_model": "HuggingFaceTB/SmolLM2-1.7B",      # model weights for training
    "tokenizer_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # tokenizer for chat template (same vocab)
    "max_length": 2048,
    # Split sizes (rows, not tokens)
    "train_size": 1_000,
    "val_size": 2_000,
    "attr_pool_size": 15_000,
    # Attribution query composition (handled by rebuild_attr_query.py for v2)
    "query_smol_size": 0,     # disabled: rebuild_attr_query.py builds attr_query separately
    "query_gsm8k_size": 0,
    "query_arc_size": 0,
    "output_dir": "runs/smoltalk_v2",
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
    for row in raw:
        raw_rows_seen += 1
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
# Build benchmark attribution query
# ---------------------------------------------------------------------------

def _arc_row_to_instruct(row: dict, tokenizer: Any, max_length: int) -> dict | None:
    """Convert one ARC-Challenge row to an instruct-formatted attribution example."""
    choices = row["choices"]
    labels_list = choices["label"]
    texts = choices["text"]
    answer_key = row["answerKey"]

    options = "\n".join(f"{l}) {t}" for l, t in zip(labels_list, texts))
    user_text = f"Answer the following question with just the letter of the correct option.\n\nQuestion: {row['question']}\n\n{options}"
    assistant_text = answer_key

    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    tok_out = _mask_prompt(messages, tokenizer, max_length)
    return tok_out


def _gsm8k_row_to_instruct(row: dict, tokenizer: Any, max_length: int) -> dict | None:
    """Convert one GSM8K row to an instruct example.

    The answer field contains the full step-by-step solution ending with '#### <number>',
    which is exactly the reasoning trace we want as the supervised completion.
    """
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    tok_out = _mask_prompt(messages, tokenizer, max_length)
    return tok_out


def build_attr_query(tokenizer: Any, cfg: dict, raw_rows_consumed: int) -> Dataset:
    """Build attribution query: smol-magpie-ultra (fresh rows) + GSM8K + ARC-Challenge.

    smol-magpie-ultra: high-quality instruction data — the core quality signal.
      Streamed from SmolTalk 'all' past raw_rows_consumed, guaranteeing no overlap
      with train/val/pool splits.
    GSM8K train: multi-step reasoning traces — selects for reasoning capability.
    ARC-Challenge train: factual grounding — keeps some short Q&A signal.
    """
    rows: list[dict] = []

    # --- smol-magpie-ultra: stream fresh rows past the consumed offset ---
    n_smol = cfg["query_smol_size"]
    print(f"Building attribution query: smol-magpie-ultra (skipping {raw_rows_consumed:,} rows) ...")
    raw = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split="train", streaming=True)
    raw_seen = 0
    smol_rows: list[dict] = []
    for row in raw:
        raw_seen += 1
        if raw_seen <= raw_rows_consumed:
            if raw_seen % 100_000 == 0:
                print(f"  skipping... {raw_seen:,}/{raw_rows_consumed:,}", flush=True)
            continue
        if row.get("source") != "smol-magpie-ultra":
            continue
        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, cfg["max_length"])
        if tok_out is None:
            continue
        tok_out["magpie_score"] = _get_magpie_score(row)
        smol_rows.append(tok_out)
        if len(smol_rows) >= n_smol:
            break
    if len(smol_rows) < n_smol:
        print(f"  WARNING: only found {len(smol_rows)} smol-magpie-ultra rows (wanted {n_smol})")
    print(f"  {len(smol_rows)} smol-magpie-ultra examples")
    rows.extend(smol_rows)

    # --- GSM8K: multi-step reasoning traces (optional) ---
    gsm_rows: list[dict] = []
    n_gsm = cfg["query_gsm8k_size"]
    if n_gsm > 0:
        print(f"Building attribution query: GSM8K train ({n_gsm} examples) ...")
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for row in gsm:
            tok_out = _gsm8k_row_to_instruct(row, tokenizer, cfg["max_length"])
            if tok_out is not None:
                tok_out["magpie_score"] = 0.0
                gsm_rows.append(tok_out)
            if len(gsm_rows) >= n_gsm:
                break
        print(f"  {len(gsm_rows)} GSM8K examples")
        rows.extend(gsm_rows)

    # --- ARC-Challenge: factual grounding (optional) ---
    arc_rows: list[dict] = []
    n_arc = cfg["query_arc_size"]
    if n_arc > 0:
        print(f"Building attribution query: ARC-Challenge train ({n_arc} examples) ...")
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for row in arc:
            tok_out = _arc_row_to_instruct(row, tokenizer, cfg["max_length"])
            if tok_out is not None:
                tok_out["magpie_score"] = 0.0
                arc_rows.append(tok_out)
            if len(arc_rows) >= n_arc:
                break
        print(f"  {len(arc_rows)} ARC-Challenge examples")
        rows.extend(arc_rows)

    total = len(rows)
    print(f"  attr_query total: {total} examples ({len(smol_rows)} smol + {len(gsm_rows)} GSM8K + {len(arc_rows)} ARC)")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SMOKE_CONFIG = {
    **CONFIG,
    "train_size": 64,
    "val_size": 20,
    "attr_pool_size": 50,
    "query_smol_size": 40,
    "query_gsm8k_size": 0,
    "query_arc_size": 0,
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
    query_ds = build_attr_query(tokenizer, cfg, raw_rows_consumed)

    # Save splits
    splits_info: dict[str, dict] = {}
    for name, ds in [
        ("train", train_ds),
        ("val", val_ds),
        ("attr_pool", attr_pool_ds),
        ("attr_query", query_ds),
    ]:
        if len(ds) == 0:
            print(f"  {name}: 0 rows — skipped (rebuild_attr_query.py will build this)")
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
    manifest = {
        "base_model": cfg["base_model"],
        "tokenizer_model": cfg["tokenizer_model"],
        "max_length": cfg["max_length"],
        "dataset": cfg["dataset_name"],
        "dataset_config": cfg["dataset_config"],
        "raw_rows_consumed": raw_rows_consumed,  # used by build_continuation_data.py to skip ahead
        "splits": {
            # score.py uses score_pool + attr_query
            "score_pool": splits_info["attr_pool"],
            # attr_query populated by rebuild_attr_query.py (may not exist yet)
            **( {"attr_query": splits_info["attr_query"]} if "attr_query" in splits_info else {} ),
            # full train/val for reference
            "train": splits_info["train"],
            "val": splits_info["val"],
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to: {manifest_path}")
    print("\nNext steps:")
    print("  1. uv run finetune.py            # Phase 1: LoRA SFT on train split")
    print("  2. uv run eval_harness.py ...     # Gate check on instruct benchmarks")
    print("  3. uv run score.py --adapter-path runs/smoltalk_v1/adapter  # Attribution")


if __name__ == "__main__":
    main()
