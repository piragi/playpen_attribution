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
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm.auto import tqdm

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

def _score_cache_signature(cfg: dict, manifest: dict, probe_paths: list[Path]) -> dict:
    category_filter = cfg.get("category_filter")
    category_vals = sorted(category_filter) if category_filter else None
    probe_files = []
    for p in probe_paths:
        st = p.stat()
        probe_files.append({"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {
        "dataset": manifest["dataset"],
        "dataset_config": manifest["dataset_config"],
        "raw_rows_consumed": manifest.get("raw_rows_consumed"),
        "max_length": manifest["max_length"],
        "base_model": manifest["base_model"],
        "gen_adapter_path": cfg.get("gen_adapter_path"),
        "extraction_layer": cfg.get("extraction_layer", 17),
        "category_filter": category_vals,
        "pool_size": cfg.get("pool_size"),
        "probes": probe_files,
    }


def _load_score_cache(cache_dir: Path, signature: dict) -> tuple[list[dict], np.ndarray] | None:
    meta_path = cache_dir / "score_cache_meta.json"
    rows_path = cache_dir / "pool_rows"
    scores_path = cache_dir / "scores.npy"
    if not (meta_path.exists() and rows_path.exists() and scores_path.exists()):
        return None
    try:
        cached_meta = json.loads(meta_path.read_text())
        if cached_meta != signature:
            return None
        rows = load_from_disk(str(rows_path)).to_list()
        scores = np.load(str(scores_path))
        if len(rows) != len(scores):
            return None
        return rows, scores
    except Exception as e:
        print(f"WARNING: failed to load score cache ({e}); rescoring.")
        return None


def _save_score_cache(cache_dir: Path, signature: dict, rows: list[dict], scores: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_path = cache_dir / "pool_rows"
    if rows_path.exists():
        shutil.rmtree(rows_path)
    Dataset.from_list(rows).save_to_disk(str(rows_path))
    np.save(str(cache_dir / "scores.npy"), scores.astype(np.float32))
    (cache_dir / "score_cache_meta.json").write_text(json.dumps(signature, indent=2))

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
    skip_pbar = tqdm(total=skip_rows, desc="Skip raw rows", unit="row") if skip_rows > 0 else None
    score_pbar = tqdm(total=cfg.get("pool_size"), desc="Score kept rows", unit="row")

    def flush() -> None:
        n = len(buffer)
        scored_before = len(pool_rows)
        scores = score_batch(buffer, probes, model, captured, device)
        pool_rows.extend(buffer)
        all_scores.extend(scores.tolist())
        buffer.clear()
        if cfg.get("pool_size"):
            remaining = max(0, cfg["pool_size"] - scored_before)
            score_pbar.update(min(n, remaining))
        else:
            score_pbar.update(n)

    print(f"Streaming {manifest['dataset']}, skipping {skip_rows:,} rows ...")
    for i, row in enumerate(raw):
        if i < skip_rows:
            if skip_pbar is not None:
                skip_pbar.update(1)
            continue
        if skip_pbar is not None:
            skip_pbar.close()
            skip_pbar = None
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
        if cfg.get("pool_size") and len(pool_rows) >= cfg["pool_size"]:
            break

    if buffer:
        flush()
    if skip_pbar is not None:
        skip_pbar.close()
    score_pbar.close()

    pool_rows = pool_rows[:cfg.get("pool_size")]
    scores    = np.array(all_scores[:cfg.get("pool_size")], dtype=np.float32)
    print(f"Collected {len(pool_rows):,} examples")
    return pool_rows, scores


def _group_by_category(pool_rows: list[dict]) -> dict[str, list[int]]:
    """Map category → pool indices. Rows without a category go into ''."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(pool_rows):
        groups.setdefault(r.get("category") or "", []).append(i)
    return groups


def _category_quotas(cat_groups: dict[str, list[int]], n: int) -> dict[str, int]:
    """Per-category row quotas summing to n, proportional to group sizes."""
    total = sum(len(idxs) for idxs in cat_groups.values())
    raw = {cat: n * len(idxs) / total for cat, idxs in cat_groups.items()}
    quotas = {cat: int(q) for cat, q in raw.items()}
    deficit = n - sum(quotas.values())
    for cat in sorted(raw, key=lambda c: raw[c] - int(raw[c]), reverse=True)[:deficit]:
        quotas[cat] += 1
    return quotas


def select_quality(scores: np.ndarray, cat_groups: dict[str, list[int]], n: int) -> list[int]:
    """Return sorted indices of top-n rows by probe score, stratified by category."""
    quotas = _category_quotas(cat_groups, n)
    selected: list[int] = []
    for cat, quota in quotas.items():
        top = sorted(cat_groups[cat], key=lambda i: float(scores[i]), reverse=True)[:quota]
        selected.extend(top)
    return sorted(selected)


def select_random(cat_groups: dict[str, list[int]], n: int, exclude: set[int], seed: int) -> list[int]:
    """Return n uniformly random indices stratified by category, excluding the given set."""
    quotas = _category_quotas(cat_groups, n)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cat, quota in quotas.items():
        candidates = [i for i in cat_groups[cat] if i not in exclude]
        chosen = rng.choice(candidates, size=min(quota, len(candidates)), replace=False)
        selected.extend(int(i) for i in chosen)
    return sorted(selected)


def select_label(pool_rows: list[dict], cat_groups: dict[str, list[int]], qualities: set[str], n: int) -> list[int]:
    """Return indices of top-n rows by LLM quality label, stratified by category."""
    rank = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}
    quotas = _category_quotas(cat_groups, n)
    selected: list[int] = []
    for cat, quota in quotas.items():
        candidates = sorted(
            [i for i in cat_groups[cat]
             if pool_rows[i].get("quality", "").strip().lower() in qualities],
            key=lambda i: rank.get(pool_rows[i].get("quality", ""), -1),
            reverse=True,
        )
        selected.extend(candidates[:quota])
    return sorted(selected)


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
    probe_specs = cfg["probes"]
    probe_paths = [probe_dir / (s.get("filename") or f"probe_{s['name']}.pkl") for s in probe_specs]
    probe_names = [s["name"] for s in probe_specs]
    cache_dir = run_dir / "continuation" / "cache"
    cache_sig = _score_cache_signature(cfg, manifest, probe_paths)
    cached = _load_score_cache(cache_dir, cache_sig)

    if cached is not None:
        pool_rows, scores_matrix = cached
        print(f"Loaded score cache: {len(pool_rows):,} rows from {cache_dir}")
    else:
        probes = []
        for spec in probe_specs:
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

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        _save_score_cache(cache_dir, cache_sig, pool_rows, scores_matrix)
        print(f"Saved score cache: {len(pool_rows):,} rows to {cache_dir}")

    n_pool = len(pool_rows)
    if scores_matrix.ndim == 1:
        scores_matrix = scores_matrix.reshape(-1, 1)
    if scores_matrix.shape[1] != len(probe_names):
        raise RuntimeError(
            f"Probe/score shape mismatch: scores have {scores_matrix.shape[1]} columns, "
            f"but config has {len(probe_names)} probes."
        )

    out_dir = run_dir / "continuation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_groups = _group_by_category(pool_rows)
    arms: dict[str, dict] = {}
    all_quality_idx: set[int] = set()

    for j, name in enumerate(probe_names):
        scores = scores_matrix[:, j]

        quality_idx    = select_quality(scores, cat_groups, cfg["quality_size"])
        quality_50_idx = select_quality(scores, cat_groups, cfg["quality_size"] // 2)
        all_quality_idx.update(quality_idx)

        arms[f"quality_{name}"]       = save_arm(pool_rows, quality_idx,    out_dir, f"quality_{name}")
        arms[f"quality_{name}_50pct"] = save_arm(pool_rows, quality_50_idx, out_dir, f"quality_{name}_50pct")

    random_idx = select_random(cat_groups, cfg.get("random_size", 10_000), all_quality_idx, cfg.get("seed", 42))
    arms["random"] = save_arm(pool_rows, random_idx, out_dir, "random")

    random_50_idx = select_random(cat_groups, cfg.get("random_size", 10_000) // 2, all_quality_idx, cfg.get("seed", 42))
    arms["random_50pct"] = save_arm(pool_rows, random_50_idx, out_dir, "random_50pct")

    label_idx = select_label(pool_rows, cat_groups, {"good", "excellent"}, cfg["quality_size"])
    arms["label_good_excellent"] = save_arm(pool_rows, label_idx, out_dir, "label_good_excellent")

    label_50_idx = select_label(pool_rows, cat_groups, {"good", "excellent"}, cfg["quality_size"] // 2)
    arms["label_good_excellent_50pct"] = save_arm(pool_rows, label_50_idx, out_dir, "label_good_excellent_50pct")

    cont_manifest = {
        "base_model": base_model,
        "max_length": manifest["max_length"],
        "extraction_layer": cfg.get("extraction_layer", 17),
        "probes": [
            {"name": s["name"], "path": str(probe_dir / (s.get("filename") or f"probe_{s['name']}.pkl"))}
            for s in probe_specs
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
