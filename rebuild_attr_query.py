"""Rebuild attr_query from smol-magpie-ultra (quality-filtered).

Loads smol-magpie-ultra directly (quality labels preserved), filters for
quality in quality_min, and takes the first query_smol_size rows as attr_query.

Run this when you want to change the attribution query without rebuilding everything:

    uv run rebuild_attr_query.py

Then rerun score.py with the updated attr_query:

    uv run score.py --adapter-path runs/smoltalk_v1/adapter \
        --output-dir runs/smoltalk_v1/scores
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt

CONFIG = {
    "manifest_path": "runs/smoltalk_v4/manifest.json",
    "query_smol_size": 4096,
    "quality_min": {"good", "excellent"},
    # Set to None to accept all categories, or a set of category strings to filter.
    # Override via --category CLI arg.
    "category_filter": {"math", "data-analysis"},
    "seed": 42,
}

# Canonical category name → manifest key suffix (avoids hyphens in keys).
_CATEGORY_KEY = {
    "math": "math",
    "data-analysis": "da",
    "reasoning": "reasoning",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild attr_query, optionally filtered to one category.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=list(_CATEGORY_KEY.keys()) + ["all"],
        help="Category to filter to (math, data-analysis, reasoning). "
             "Omit or pass 'all' for no category filter.",
    )
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.category and args.category != "all":
        cfg["category_filter"] = {args.category}

    manifest = json.loads(Path(cfg["manifest_path"]).read_text())

    max_length = manifest["max_length"]

    tokenizer_model = manifest.get("tokenizer_model")  # None if not in manifest
    base_model = manifest.get("base_model", "google/gemma-3-1b-pt")
    if not tokenizer_model or tokenizer_model == base_model:
        tokenizer_model = (base_model[:-3] + "-it") if base_model.endswith("-pt") else (base_model + "-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = manifest["dataset"]  # "HuggingFaceTB/smoltalk"
    quality_ok = cfg["quality_min"]
    n_target = cfg["query_smol_size"]

    # Skip rows already consumed by build_sft_data.py to prevent attr_query/attr_pool overlap.
    # Only relevant when dataset_config == "smol-magpie-ultra" (both stream the same source).
    skip_rows = 0
    if manifest.get("dataset_config") == "smol-magpie-ultra":
        skip_rows = manifest.get("raw_rows_consumed", 0)
        if skip_rows:
            print(f"  skipping first {skip_rows:,} raw rows (consumed by build_sft_data.py)")

    category_filter = cfg.get("category_filter")  # None → accept all
    print(f"Loading {dataset_name}/smol-magpie-ultra ...")
    print(f"  quality filter:   {quality_ok}")
    print(f"  category filter:  {sorted(category_filter) if category_filter else 'all'}")
    smol_raw = load_dataset(dataset_name, "smol-magpie-ultra", split="train", streaming=True)

    quality_dist: dict[str, int] = {}
    category_dist: dict[str, int] = {}
    candidate_rows: list[dict] = []
    raw_seen = 0

    for row in smol_raw:
        raw_seen += 1
        if raw_seen <= skip_rows:
            continue
        quality = row.get("quality", "")
        category = row.get("category", "")
        quality_dist[quality] = quality_dist.get(quality, 0) + 1

        if quality not in quality_ok:
            continue
        if category_filter and category not in category_filter:
            continue

        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, max_length)
        if tok_out is None:
            continue

        tok_out["magpie_score"] = _get_magpie_score(row)
        tok_out["category"] = category
        category_dist[category] = category_dist.get(category, 0) + 1
        candidate_rows.append(tok_out)

        if len(candidate_rows) >= n_target:
            break

    print(f"\nQuality distribution seen:")
    for q, c in sorted(quality_dist.items()):
        mark = "✓" if q in quality_ok else " "
        print(f"  {mark} {q or '(empty)'}: {c:,}")
    print(f"\nCategory breakdown of collected rows:")
    for cat, c in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"    {cat:<22} {c:>5,}  ({c/len(candidate_rows)*100:.1f}%)")
    print(f"Collected {len(candidate_rows):,} rows")

    if len(candidate_rows) == 0:
        raise RuntimeError(f"No rows passed quality filter {quality_ok}")

    query_ds = Dataset.from_list(candidate_rows)

    # Determine manifest split key and output path based on category filter.
    if category_filter and len(category_filter) == 1:
        cat = next(iter(category_filter))
        key_suffix = _CATEGORY_KEY.get(cat, cat.replace("-", "_"))
        split_key = f"attr_query_{key_suffix}"
    else:
        split_key = "attr_query"

    default_path = str(Path(cfg["manifest_path"]).parent / "data" / split_key)
    out_path = Path(manifest.get("splits", {}).get(split_key, {}).get("path", default_path))
    if out_path.exists():
        shutil.rmtree(out_path)
    query_ds.save_to_disk(str(out_path))

    manifest.setdefault("splits", {})[split_key] = {
        "path": str(out_path),
        "rows": len(query_ds),
        "total_tokens": int(sum(query_ds["length"])),
    }
    manifest[f"{split_key}_source"] = (
        f"smol-magpie-ultra quality>={sorted(quality_ok)}"
        + (f" category={sorted(category_filter)}" if category_filter else "")
    )
    Path(cfg["manifest_path"]).write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved {len(query_ds)} rows → {out_path}  (manifest key: {split_key})")
    run_dir = str(Path(cfg["manifest_path"]).parent)
    print("\nNext: rerun score.py to recompute attribution with the new query:")
    print(f"  uv run score.py --adapter-path {run_dir}/adapter "
          f"--query-split {split_key} "
          f"--output-dir {run_dir}/scores_{split_key.removeprefix('attr_query_') if split_key != 'attr_query' else 'math_da'}")


if __name__ == "__main__":
    main()
