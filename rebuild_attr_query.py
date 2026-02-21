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

import json
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt

CONFIG = {
    "manifest_path": "runs/smoltalk_v1/manifest.json",
    "query_smol_size": 1024,
    "quality_min": {"good", "excellent"},
    "seed": 42,
}


def main() -> None:
    cfg = CONFIG
    manifest = json.loads(Path(cfg["manifest_path"]).read_text())

    max_length = manifest["max_length"]

    tokenizer_model = manifest.get("tokenizer_model", "google/gemma-3-1b-it")
    base_model = manifest.get("base_model", "google/gemma-3-1b-pt")
    if not tokenizer_model or tokenizer_model == base_model:
        tokenizer_model = base_model.replace("-pt", "") + "-it"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = manifest["dataset"]  # "HuggingFaceTB/smoltalk"
    quality_ok = cfg["quality_min"]
    n_target = cfg["query_smol_size"]

    print(f"Loading {dataset_name}/smol-magpie-ultra ...")
    print(f"  quality filter: {quality_ok}")
    smol_raw = load_dataset(dataset_name, "smol-magpie-ultra", split="train", streaming=True)

    quality_dist: dict[str, int] = {}
    candidate_rows: list[dict] = []

    for row in smol_raw:
        quality = row.get("quality", "")
        quality_dist[quality] = quality_dist.get(quality, 0) + 1

        if quality not in quality_ok:
            continue

        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, max_length)
        if tok_out is None:
            continue

        tok_out["magpie_score"] = _get_magpie_score(row)
        candidate_rows.append(tok_out)

        if len(candidate_rows) >= n_target:
            break

    print(f"\nQuality distribution seen so far:")
    for q, c in sorted(quality_dist.items()):
        mark = "✓" if q in quality_ok else " "
        print(f"  {mark} {q or '(empty)'}: {c:,}")
    print(f"Collected {len(candidate_rows):,} rows")

    if len(candidate_rows) == 0:
        raise RuntimeError(f"No rows passed quality filter {quality_ok}")

    query_ds = Dataset.from_list(candidate_rows)
    out_path = Path(manifest["splits"]["attr_query"]["path"])
    if out_path.exists():
        shutil.rmtree(out_path)
    query_ds.save_to_disk(str(out_path))

    manifest["splits"]["attr_query"] = {
        "path": str(out_path),
        "rows": len(query_ds),
        "total_tokens": int(sum(query_ds["length"])),
    }
    manifest["attr_query_source"] = f"smol-magpie-ultra quality>={sorted(quality_ok)}"
    Path(cfg["manifest_path"]).write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved {len(query_ds)} attr_query rows → {out_path}")
    print("\nNext: rerun score.py to recompute attribution with the new query:")
    print("  uv run score.py --adapter-path runs/smoltalk_v1/adapter "
          "--output-dir runs/smoltalk_v1/scores")


if __name__ == "__main__":
    main()
