"""Rebuild attr_query from smol-magpie-ultra (quality-filtered).

Loads smol-magpie-ultra, filters for quality in quality_min, and saves the
first query_smol_size rows as the attribution query split.

Run after build_sft_data.py when you want to change the query without
rebuilding everything:

    uv run rebuild_attr_query.py

Then rerun score.py with the updated query:

    uv run score.py --manifest <run_dir>/manifest.json \
        --adapter-path <run_dir>/adapter \
        --query-split attr_query \
        --output-dir <run_dir>/scores_math_da
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    "run_dir": "runs/smoltalk_v4",
    "query_smol_size": 4096,
    "query_quality_min": {"good", "excellent"},
    "category_filter": {"math", "data-analysis"},
}

# Category name → manifest key suffix (only needed where the name isn't a valid key as-is)
_CATEGORY_KEY = {
    "data-analysis": "da",
}


def run(cfg: dict) -> None:
    run_dir = Path(cfg["run_dir"])
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base_model = manifest.get("base_model", DEFAULT_BASE_MODEL)
    tokenizer = load_tokenizer(base_model)

    # Skip rows already consumed by build_sft_data.py to prevent overlap.
    skip_rows = 0
    if manifest.get("dataset_config") == "smol-magpie-ultra":
        skip_rows = manifest.get("raw_rows_consumed", 0)
        if skip_rows:
            print(f"  skipping first {skip_rows:,} raw rows (consumed by build_sft_data.py)")

    quality_ok = cfg.get("query_quality_min", {"good", "excellent"})
    category_filter = cfg.get("category_filter")
    # Allow a single-category override via query_category key.
    if cfg.get("query_category") and cfg["query_category"] != "all":
        category_filter = {cfg["query_category"]}

    print(f"Loading {manifest['dataset']}/smol-magpie-ultra ...")
    print(f"  quality filter:  {quality_ok}")
    print(f"  category filter: {sorted(category_filter) if category_filter else 'all'}")

    raw = load_dataset(manifest["dataset"], "smol-magpie-ultra", split="train", streaming=True)
    quality_dist: dict[str, int] = {}
    category_dist: dict[str, int] = {}
    candidate_rows: list[dict] = []

    for i, row in enumerate(raw):
        if i < skip_rows:
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
        tok = mask_prompt(messages, tokenizer, manifest["max_length"])
        if tok is None:
            continue

        tok["magpie_score"] = get_magpie_score(row)
        tok["category"] = category
        category_dist[category] = category_dist.get(category, 0) + 1
        candidate_rows.append(tok)

        if len(candidate_rows) >= cfg.get("query_smol_size", 4096):
            break

    if not candidate_rows:
        raise RuntimeError(f"No rows passed quality filter {quality_ok}")

    print(f"\nQuality distribution seen:")
    for q, c in sorted(quality_dist.items()):
        mark = "✓" if q in quality_ok else " "
        print(f"  {mark} {q or '(empty)'}: {c:,}")
    print(f"\nCategory breakdown of collected rows:")
    for cat, c in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"    {cat:<22} {c:>5,}  ({c / len(candidate_rows) * 100:.1f}%)")
    print(f"Collected {len(candidate_rows):,} rows")

    query_ds = Dataset.from_list(candidate_rows)

    # Determine manifest split key and output path.
    if category_filter and len(category_filter) == 1:
        cat = next(iter(category_filter))
        split_key = f"attr_query_{_CATEGORY_KEY.get(cat, cat.replace('-', '_'))}"
    else:
        split_key = "attr_query"

    default_path = str(run_dir / "data" / split_key)
    out_path = Path(manifest.get("splits", {}).get(split_key, {}).get("path", default_path))
    if out_path.exists():
        shutil.rmtree(out_path)
    query_ds.save_to_disk(str(out_path))

    manifest.setdefault("splits", {})[split_key] = {
        "path": str(out_path),
        "rows": len(query_ds),
        "total_tokens": int(sum(query_ds["length"])),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {len(query_ds)} rows → {out_path}  (manifest key: {split_key})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild attr_query, optionally filtered to one category.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=sorted(CONFIG["category_filter"]) + ["all"],
        help="Category to filter to. Omit or pass 'all' for no category filter.",
    )
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.category:
        cfg["query_category"] = args.category
    run(cfg)


if __name__ == "__main__":
    main()
