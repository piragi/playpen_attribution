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


def _category_quotas(
    cat_rows: dict[str, list[dict]],
    n: int,
) -> dict[str, int]:
    """Per-category row quotas summing to n, proportional to observed category sizes."""
    total = sum(len(rows) for rows in cat_rows.values())
    if n <= 0 or total <= 0:
        return {cat: 0 for cat in cat_rows}
    raw = {cat: n * len(rows) / total for cat, rows in cat_rows.items()}
    quotas = {cat: int(q) for cat, q in raw.items()}
    deficit = n - sum(quotas.values())
    for cat in sorted(raw, key=lambda c: raw[c] - int(raw[c]), reverse=True)[:deficit]:
        quotas[cat] += 1
    return quotas


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
    total_n = int(cfg.get("query_smol_size", 4096))
    collect_target = total_n * 2
    buckets: dict[str, list[dict]] = {cat: [] for cat in sorted(category_filter)} if category_filter else {}
    quality_dist: dict[str, int] = {}
    rows_seen_after_skip = 0

    for i, row in enumerate(raw):
        if i < skip_rows:
            continue
        rows_seen_after_skip += 1
        quality = row.get("quality", "")
        category = row.get("category", "")
        quality_dist[quality] = quality_dist.get(quality, 0) + 1

        if quality not in quality_ok:
            continue
        bucket_key = category if category_filter else (category or "(empty)")
        if category_filter and bucket_key not in buckets:
            continue

        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok = mask_prompt(messages, tokenizer, manifest["max_length"])
        if tok is None:
            continue

        tok["magpie_score"] = get_magpie_score(row)
        tok["quality"] = quality
        tok["category"] = category
        buckets.setdefault(bucket_key, [])
        buckets[bucket_key].append(tok)

        if sum(len(v) for v in buckets.values()) >= collect_target:
            break

    categories = sorted(buckets)
    total_available = sum(len(v) for v in buckets.values())
    target_n = min(total_n, total_available)
    quotas = _category_quotas(buckets, target_n)

    candidate_rows = [row for cat in categories for row in buckets[cat][:quotas[cat]]]

    if not candidate_rows:
        raise RuntimeError(f"No rows passed quality filter {quality_ok}")

    n = len(candidate_rows)
    magpie_scores = [r["magpie_score"] for r in candidate_rows if r["magpie_score"] is not None]

    print(f"\nQuality distribution seen (all rows):")
    for q, c in sorted(quality_dist.items()):
        mark = "✓" if q in quality_ok else " "
        print(f"  {mark} {q or '(empty)'}: {c:,}")
    print(f"\nCategory targets vs available:")
    for cat in categories:
        target = quotas.get(cat, 0)
        avail = len(buckets[cat])
        print(f"    {cat:<22} target={target:>5,}  available={avail:>5,}")
    print(f"\nQuery set label distribution ({n:,} rows collected):")
    query_quality_dist = {}
    for r in candidate_rows:
        query_quality_dist[r["quality"]] = query_quality_dist.get(r["quality"], 0) + 1
    for q, c in sorted(query_quality_dist.items(), key=lambda x: -x[1]):
        print(f"    {q:<12} {c:>5,}  ({c / n * 100:.1f}%)")
    print(f"\nCategory breakdown:")
    chosen_counts: dict[str, int] = {}
    for r in candidate_rows:
        chosen_counts[r["category"] or "(empty)"] = chosen_counts.get(r["category"] or "(empty)", 0) + 1
    for cat in categories:
        c = chosen_counts.get(cat, 0)
        print(f"    {cat:<22} {c:>5,}  ({c / n * 100:.1f}%)")
    if magpie_scores:
        import numpy as np
        ms = np.array(magpie_scores)
        print(f"\nMagpie score (numeric):  mean={ms.mean():.3f}  std={ms.std():.3f}  "
              f"min={ms.min():.3f}  p25={np.percentile(ms,25):.3f}  "
              f"p75={np.percentile(ms,75):.3f}  max={ms.max():.3f}")

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
    manifest["attr_query_raw_rows_consumed"] = skip_rows + rows_seen_after_skip
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {len(query_ds)} rows → {out_path}  (manifest key: {split_key})")
    print(f"attr_query_raw_rows_consumed={manifest['attr_query_raw_rows_consumed']:,}")


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
