"""Score new SmolTalk examples with the residual stream probe and produce ablation arms.

Streams SmolTalk past the rows consumed by build_sft_data.py, scores each
example with the probe(s) trained in probe.py, and produces training-ready
datasets for finetune.py:

  continuation/quality_{probe}/        — top-N by probe score
  continuation/quality_{probe}_50pct/  — top-N/2 by probe score
  continuation/random/                 — uniform random baseline
  continuation/label_good_excellent/   — top-N by LLM quality label (control)

Run AFTER probe.py:

    uv run generate_continued_dataset.py
"""
from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset

from pipeline_common import (
    ensure_hf_home_env,
    get_magpie_score,
    last_response_token_positions,
    load_model_with_hook,
    load_tokenizer,
    mask_prompt,
    pad_tokenized_batch,
    pool_hidden_at_positions,
    resolve_device_dtype,
)

ensure_hf_home_env()

CONFIG = {
    "run_dir": "runs/smoltalk_v4",
    # probe filenames default to probe_{name}.pkl when not set
    "probes": [
        {"name": "math_da"},
    ],
    "extraction_layer": 17,         # must match probe.py
    "gen_adapter_path": None,       # set to extract from adapter instead of base model
    "category_filter": {"math", "data-analysis"},
    "pool_size": 30_000,
    "quality_size": 10_000,         # rows in the full quality arm (and label arm)
    "random_size": 10_000,
    "gen_batch_size": 64,
    "seed": 42,
}


# ── Core functions ────────────────────────────────────────────────────────────

def score_batch(batch_rows, probes, model, captured, device) -> np.ndarray:
    """Forward pass → probe scores for a batch of tokenized rows.

    Returns a (len(batch), n_probes) array.
    """
    ids_t, lbl_t = pad_tokenized_batch(
        [r["input_ids"] for r in batch_rows],
        [r["labels"] for r in batch_rows],
        device=device,
    )
    with torch.inference_mode():
        model(input_ids=ids_t)
    pooled = pool_hidden_at_positions(
        captured["acts"], last_response_token_positions(lbl_t)
    ).float().cpu().numpy()
    return np.array([
        [float(p["probe"].predict(pooled[i].reshape(1, -1))[0]) for p in probes]
        for i in range(len(batch_rows))
    ])


def stream_and_score(cfg, manifest, probes, model, captured, device) -> tuple[list[dict], np.ndarray]:
    """Stream SmolTalk, tokenize, and score each example with all probes.

    Skips rows already consumed by build_sft_data.py to prevent data leakage.
    Returns (pool_rows, scores_matrix) where scores_matrix is (n, n_probes).
    """
    tokenizer = load_tokenizer(manifest["base_model"])
    skip_rows = manifest.get("raw_rows_consumed") or _estimate_skip_rows(manifest)
    category_filter = cfg.get("category_filter")

    raw = load_dataset(manifest["dataset"], manifest["dataset_config"], split="train", streaming=True)
    pool_rows: list[dict] = []
    all_scores: list[list[float]] = []
    buffer: list[dict] = []

    def flush() -> None:
        scores = score_batch(buffer, probes, model, captured, device)
        pool_rows.extend(buffer)
        all_scores.extend(scores.tolist())
        buffer.clear()

    print(f"Streaming {manifest['dataset']}, skipping {skip_rows:,} rows ...")
    for i, row in enumerate(raw):
        if i < skip_rows:
            if i > 0 and i % 50_000 == 0:
                print(f"  skipping... {i:,}/{skip_rows:,}", flush=True)
            continue
        if category_filter and row.get("category", "") not in category_filter:
            continue
        messages = row.get("messages") or row.get("conversations") or []
        tok = mask_prompt(messages, tokenizer, manifest["max_length"])
        if tok is None:
            continue
        tok["magpie_score"] = get_magpie_score(row)
        tok["quality"]   = row.get("quality", "")
        tok["category"]  = row.get("category", "")
        buffer.append(tok)
        if len(buffer) >= cfg["batch_size"]:
            flush()
            if len(pool_rows) % 5_000 == 0:
                limit_str = f"{cfg['pool_size']:,}" if cfg.get("pool_size") else "∞"
                print(f"  scored {len(pool_rows):,}/{limit_str}", flush=True)
        if cfg.get("pool_size") and len(pool_rows) >= cfg["pool_size"]:
            break

    if buffer:
        flush()

    pool_rows = pool_rows[:cfg.get("pool_size")]
    scores    = np.array(all_scores[:cfg.get("pool_size")], dtype=np.float32)
    print(f"Collected {len(pool_rows):,} examples")
    return pool_rows, scores


def select_quality(scores: np.ndarray, n: int) -> list[int]:
    """Return sorted indices of the top-n rows by probe score."""
    return sorted(int(i) for i in np.argsort(scores)[::-1][:n].tolist())


def select_random(n: int, exclude: set[int], pool_size: int, seed: int) -> list[int]:
    """Return n uniformly random indices, excluding the given set."""
    candidates = [i for i in range(pool_size) if i not in exclude]
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(candidates, size=n, replace=False).tolist())


def select_label(pool_rows: list[dict], qualities: set[str], n: int) -> list[int]:
    """Return indices of the top-n rows ranked by LLM quality label."""
    rank = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}
    candidates = sorted(
        [i for i, r in enumerate(pool_rows) if r.get("quality", "").strip().lower() in qualities],
        key=lambda i: rank.get(pool_rows[i].get("quality", ""), -1),
        reverse=True,
    )
    return sorted(candidates[:n])


def save_arm(pool_rows: list[dict], indices: list[int], out_dir: Path, name: str) -> dict:
    """Save a dataset arm and return its manifest entry."""
    rows = [pool_rows[i] for i in indices]
    path = out_dir / name
    Dataset.from_list(rows).save_to_disk(str(path))
    n_tok = sum(r["length"] for r in rows)
    print(f"  {name}: {len(rows):,} rows, {n_tok / 1e6:.1f}M tokens → {path}")
    return {"path": str(path), "rows": len(rows), "total_tokens": n_tok}


def _estimate_skip_rows(manifest: dict) -> int:
    """Estimate raw rows consumed when raw_rows_consumed is missing from manifest."""
    needed = (
        manifest["splits"]["train"]["rows"]
        + manifest["splits"]["val"]["rows"]
        + manifest["splits"]["score_pool"]["rows"]
    )
    # gather_target = 2 * needed; ~63.5% pass rate; 10% safety margin
    estimated = int((needed * 2) / 0.635 * 1.10)
    print(f"WARNING: manifest missing raw_rows_consumed — estimated skip={estimated:,}")
    return estimated


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    run_dir = Path(cfg["run_dir"])
    device, dtype = resolve_device_dtype()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    base_model = manifest["base_model"]

    probe_dir = run_dir / "probe"
    probes = []
    for spec in cfg["probes"]:
        filename = spec.get("filename") or f"probe_{spec['name']}.pkl"
        path = probe_dir / filename
        with path.open("rb") as f:
            probes.append({"name": spec["name"], "probe": pickle.load(f)})
    print(f"Loaded probes: {[p['name'] for p in probes]}")

    model, captured = load_model_with_hook(
        base_model, cfg.get("gen_adapter_path"), cfg.get("extraction_layer", 17), dtype, device
    )

    # Override batch_size key for stream_and_score compatibility
    stream_cfg = {**cfg, "batch_size": cfg.get("gen_batch_size", 64)}
    pool_rows, scores_matrix = stream_and_score(stream_cfg, manifest, probes, model, captured, device)
    n_pool = len(pool_rows)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    out_dir = run_dir / "continuation"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, dict] = {}
    all_quality_idx: set[int] = set()

    for j, p in enumerate(probes):
        scores = scores_matrix[:, j]
        name   = p["name"]

        quality_idx    = select_quality(scores, cfg["quality_size"])
        quality_50_idx = select_quality(scores, cfg["quality_size"] // 2)
        all_quality_idx.update(quality_idx)

        arms[f"quality_{name}"]       = save_arm(pool_rows, quality_idx,    out_dir, f"quality_{name}")
        arms[f"quality_{name}_50pct"] = save_arm(pool_rows, quality_50_idx, out_dir, f"quality_{name}_50pct")

    random_idx = select_random(cfg.get("random_size", 10_000), all_quality_idx, n_pool, cfg.get("seed", 42))
    arms["random"] = save_arm(pool_rows, random_idx, out_dir, "random")

    random_50_idx = select_random(cfg.get("random_size", 10_000) // 2, all_quality_idx, n_pool, cfg.get("seed", 42))
    arms["random_50pct"] = save_arm(pool_rows, random_50_idx, out_dir, "random_50pct")

    label_idx = select_label(pool_rows, {"good", "excellent"}, cfg["quality_size"])
    arms["label_good_excellent"] = save_arm(pool_rows, label_idx, out_dir, "label_good_excellent")

    label_50_idx = select_label(pool_rows, {"good", "excellent"}, cfg["quality_size"] // 2)
    arms["label_good_excellent_50pct"] = save_arm(pool_rows, label_50_idx, out_dir, "label_good_excellent_50pct")

    cont_manifest = {
        "base_model": base_model,
        "max_length": manifest["max_length"],
        "extraction_layer": cfg.get("extraction_layer", 17),
        "probes": [
            {"name": s["name"], "path": str(probe_dir / (s.get("filename") or f"probe_{s['name']}.pkl"))}
            for s in cfg["probes"]
        ],
        "pool_size": n_pool,
        "arms": arms,
    }
    cont_path = out_dir / "continuation_manifest.json"
    cont_path.write_text(json.dumps(cont_manifest, indent=2))
    print(f"\nManifest → {cont_path}")


def main() -> None:
    run(CONFIG)


if __name__ == "__main__":
    main()
