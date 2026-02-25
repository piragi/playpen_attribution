"""Train a Ridge Regression probe to predict attribution scores from the residual stream.

Runs a forward pass on the score pool, extracts the hidden state at
`extraction_layer` at the last response token position, and fits a Ridge
Regression to the Bergson attribution scores from score.py.

Reports R² and Pearson r on a held-out 20% split. Saves probe_<name>.pkl
and caches the embedding matrix for reuse across probes.

Run AFTER score.py:

    uv run score.py --manifest runs/smoltalk_v4/manifest.json \\
        --adapter-path runs/smoltalk_v4/adapter \\
        --output-dir runs/smoltalk_v4/scores_math_da
    uv run probe.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from pipeline_common import (
    ensure_hf_home_env,
    last_response_token_positions,
    load_model_with_hook,
    pad_tokenized_batch,
    pool_hidden_at_positions,
    resolve_device_dtype,
)

ensure_hf_home_env()

CONFIG = {
    "manifest_path": "runs/smoltalk_v4/manifest.json",
    "output_dir": "runs/smoltalk_v4/probe",
    # One probe is trained per entry; embeddings are extracted once and reused.
    "probes": [
        {"name": "math_da", "scores_dir": "runs/smoltalk_v4/scores_math_da"},
    ],
    "extraction_layer": 17,
    "adapter_path": None,   # set to extract from adapter instead of base model
    "ridge_alpha": 100.0,
    "val_frac": 0.20,
    "seed": 42,
    "batch_size": 64,
}


# ── Core functions ────────────────────────────────────────────────────────────

def extract_embeddings(pool_ds, model, captured, device, batch_size) -> np.ndarray:
    """Extract residual stream at the last response token for every pool example.

    Pools at the final assistant token (last index where labels != -100).
    For a decoder-only model this is the most principled pooling strategy:
    causal attention means the final token has attended to everything before it,
    giving a compressed summary of the full context — identical to how
    InstructGPT-style reward models pool before their linear scoring head.
    """
    n = len(pool_ds)
    all_embeddings: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = pool_ds.select(range(start, min(start + batch_size, n)))
            ids_t, lbl_t = pad_tokenized_batch(batch["input_ids"], batch["labels"], device=device)
            model(input_ids=ids_t)
            pooled = pool_hidden_at_positions(captured["acts"], last_response_token_positions(lbl_t))
            all_embeddings.append(pooled.float().cpu().numpy())
            if start > 0 and start % (batch_size * 50) == 0:
                print(f"  {start + batch_size:,}/{n:,} processed ...", flush=True)

    return np.concatenate(all_embeddings, axis=0)


def load_scores(scores_dir: str, n_pool: int) -> np.ndarray:
    """Load row_diagnostics.jsonl and return a (n_pool,) score array."""
    diag_path = Path(scores_dir) / "row_diagnostics.jsonl"
    if not diag_path.exists():
        raise FileNotFoundError(
            f"Scores not found: {diag_path}\n"
            f"Run: uv run score.py --query-split <query_split> --output-dir {scores_dir}"
        )
    records = sorted(
        [json.loads(line) for line in diag_path.read_text().splitlines() if line.strip()],
        key=lambda r: r["index"],
    )
    scores = np.array([r["score"] for r in records], dtype=np.float32)
    if len(scores) != n_pool:
        raise ValueError(f"Pool size {n_pool} != scores size {len(scores)} in {scores_dir}.")
    return scores


def fit_probe(embeddings, scores_y, cfg, probe_name, out_dir, base_model) -> None:
    """Fit and save one Ridge probe."""
    idx = np.arange(len(embeddings))
    train_idx, val_idx = train_test_split(idx, test_size=cfg["val_frac"], random_state=cfg["seed"])

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = scores_y[train_idx], scores_y[val_idx]

    print(f"\n--- Probe: {probe_name} ---")
    print(f"Fitting Ridge(alpha={cfg['ridge_alpha']}) on {len(train_idx):,} examples ...")
    probe = Ridge(alpha=cfg["ridge_alpha"])
    probe.fit(X_train, y_train)

    r2     = float(probe.score(X_val, y_val))
    r, p   = pearsonr(y_val.astype(float), probe.predict(X_val).astype(float))
    r2_tr  = float(probe.score(X_train, y_train))
    r_tr,_ = pearsonr(y_train.astype(float), probe.predict(X_train).astype(float))

    print(f"Validation (n={len(val_idx):,}):")
    print(f"  R²        = {r2:.4f}")
    print(f"  Pearson r = {r:.4f}  (p={p:.2e})")
    print(f"Train (sanity): R²={r2_tr:.4f}  r={r_tr:.4f}")
    if r2 < 0.05:
        print(f"  WARNING: R² < 0.05 for probe '{probe_name}'.")

    probe_path = out_dir / f"probe_{probe_name}.pkl"
    with probe_path.open("wb") as f:
        pickle.dump(probe, f)

    meta = {
        "base_model": base_model,
        "probe_name": probe_name,
        "extraction_layer": cfg["extraction_layer"],
        "ridge_alpha": cfg["ridge_alpha"],
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_r2": r2,
        "val_pearson_r": float(r),
        "train_r2": r2_tr,
    }
    (out_dir / f"probe_meta_{probe_name}.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved probe → {probe_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = CONFIG
    np.random.seed(cfg["seed"])

    device, dtype = resolve_device_dtype()
    print(f"Device: {device}, dtype: {dtype}")

    manifest = json.loads(Path(cfg["manifest_path"]).read_text())
    base_model = manifest["base_model"]

    pool_ds = load_from_disk(str(Path(manifest["splits"]["score_pool"]["path"])))
    print(f"Pool: {len(pool_ds):,} examples")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings once, reuse for all probes.
    # Cache is keyed by (base_model, adapter_path, layer) so stale caches are
    # avoided when you change the embedding source.
    fingerprint = hashlib.sha1(
        f"{base_model}|{cfg.get('adapter_path')}|{cfg['extraction_layer']}".encode()
    ).hexdigest()[:8]
    emb_path = out_dir / f"pool_embeddings_{fingerprint}.npy"

    if emb_path.exists():
        print(f"\nReusing cached embeddings from {emb_path}")
        embeddings = np.load(str(emb_path))
    else:
        model, captured = load_model_with_hook(
            base_model, cfg.get("adapter_path"), cfg["extraction_layer"], dtype, device
        )
        print(f"\nExtracting residual stream embeddings ...")
        embeddings = extract_embeddings(pool_ds, model, captured, device, cfg["batch_size"])
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        np.save(str(emb_path), embeddings)
        print(f"Saved embeddings → {emb_path}")

    print(f"Embeddings: {embeddings.shape}  (dtype={embeddings.dtype})")

    # Train one probe per entry in CONFIG['probes'].
    for spec in cfg["probes"]:
        scores_y = load_scores(spec["scores_dir"], len(pool_ds))
        print(f"\nScores [{spec['name']}]: range [{scores_y.min():.4f}, {scores_y.max():.4f}]  mean={scores_y.mean():.4f}")
        fit_probe(embeddings, scores_y, cfg, spec["name"], out_dir, base_model)

    print("\nNext: uv run generate_continued_dataset.py")


if __name__ == "__main__":
    main()
