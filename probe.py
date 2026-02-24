"""Train a Ridge Regression probe to predict Bergson attribution scores from residual stream.

Runs the base model forward pass on the score pool, extracts the
residual stream at `extraction_layer` over response token positions only
(labels != -100), and fits a Ridge Regression to the continuous Bergson attribution scores.

Reports R² and Pearson r on a held-out 20% split. Saves probe_<name>.pkl and
the full embedding matrix (reused by generate_continued_dataset.py).

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
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from peft import PeftModel
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM

from pipeline_common import (
    ATTN_IMPLEMENTATION,
    ensure_hf_home_env,
    get_transformer_layers_for_hook,
    last_response_token_positions,
    pad_tokenized_batch,
    pool_hidden_at_positions,
    resolve_device_dtype,
)

ensure_hf_home_env()

CONFIG = {
    "manifest_path": "runs/smoltalk_v4/manifest.json",
    "output_dir": "runs/smoltalk_v4/probe",
    # List of (name, scores_dir) pairs — one probe is trained per entry.
    # Embeddings are extracted once and reused for all probes.
    # Set to None to fall back to single-probe mode using scores_dir below.
    "probes": [
        {"name": "math_da", "scores_dir": "runs/smoltalk_v4/scores_math_da"},
    ],
    # Single-probe fallback (used when probes is None).
    "scores_dir": "runs/smoltalk_v4/scores_math_da",
    # Ablation: set to True to use binary quality labels instead of Bergson scores.
    "use_quality_labels": False,
    # Layer to extract from (0-indexed).
    "extraction_layer": 17,
    "ridge_alpha": 100.0,
    "val_frac": 0.20,
    "seed": 42,
    "batch_size": 64,
    "device": "auto",
    # Embedding source model for residual extraction:
    # - "base": use manifest base_model (default)
    # - "adapter": load manifest base_model + this LoRA adapter
    "embedding_source_mode": "base",
    "embedding_adapter_path": None,
}


def resolve_embedding_source(cfg: dict, base_model: str) -> dict[str, Any]:
    """Resolve embedding source config into a normalized descriptor."""
    mode = str(cfg.get("embedding_source_mode", "base")).strip().lower()
    if mode not in {"base", "adapter"}:
        raise ValueError(f"Invalid embedding_source_mode '{mode}'. Expected 'base' or 'adapter'.")

    adapter_path = cfg.get("embedding_adapter_path")
    if mode == "adapter":
        if not adapter_path:
            raise ValueError("embedding_source_mode='adapter' requires embedding_adapter_path.")
        adapter_path = str(Path(adapter_path))
    else:
        adapter_path = None

    source = {
        "mode": mode,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "extraction_layer": int(cfg["extraction_layer"]),
    }
    fingerprint = hashlib.sha1(json.dumps(source, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    source["fingerprint"] = fingerprint
    return source


def extract_embeddings(
    pool_ds,
    model: torch.nn.Module,
    captured: dict,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Extract residual stream at the last response token position for every pool example.

    Uses the last position where labels != -100 (final assistant token). For a
    decoder-only model this is the most principled pooling strategy: causal attention
    means the final token has attended to every preceding token, so its hidden state
    is already a compressed summary of the full context. This is identical to how
    InstructGPT-style reward models pool before the linear scoring head.
    """
    n = len(pool_ds)
    all_embeddings: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = pool_ds.select(range(start, end))
            ids_t, lbl_t = pad_tokenized_batch(
                batch["input_ids"],
                batch["labels"],
                input_pad_token_id=0,
                label_pad_token_id=-100,
                device=device,
            )

            model(input_ids=ids_t)
            hidden = captured["acts"]  # (batch, seq_len, d_model)
            last_resp_idx = last_response_token_positions(lbl_t, label_pad_token_id=-100)
            pooled = pool_hidden_at_positions(hidden, last_resp_idx)

            all_embeddings.append(pooled.float().cpu().numpy())

            if start % (batch_size * 50) == 0 and start > 0:
                print(f"  {end:,}/{n:,} processed ...", flush=True)

    return np.concatenate(all_embeddings, axis=0)   # (n, d_model)


def load_scores_from_dir(scores_dir: str, n_pool: int) -> np.ndarray:
    """Load row_diagnostics.jsonl and return a (n_pool,) score array."""
    diag_path = Path(scores_dir) / "row_diagnostics.jsonl"
    if not diag_path.exists():
        raise FileNotFoundError(
            f"Scores not found: {diag_path}\n"
            f"Run: uv run score.py --query-split <query_split> --output-dir {scores_dir}"
        )
    records = [json.loads(l) for l in diag_path.read_text().splitlines() if l.strip()]
    records.sort(key=lambda r: r["index"])
    scores = np.array([r["score"] for r in records], dtype=np.float32)
    if n_pool != len(scores):
        raise ValueError(f"Pool size {n_pool} != scores size {len(scores)} in {scores_dir}.")
    return scores


def fit_probe(
    embeddings: np.ndarray,
    scores_y: np.ndarray,
    cfg: dict,
    probe_name: str,
    out_dir: Path,
    scores_source: str,
    base_model: str,
    embeddings_path: str,
    embedding_source: dict[str, Any],
) -> None:
    """Fit and save one Ridge probe."""
    idx = np.arange(len(embeddings))
    train_idx, val_idx = train_test_split(idx, test_size=cfg["val_frac"], random_state=cfg["seed"])

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = scores_y[train_idx], scores_y[val_idx]

    print(f"\n--- Probe: {probe_name} ---")
    print(f"Fitting Ridge(alpha={cfg['ridge_alpha']}) on {len(train_idx):,} examples ...")
    probe = Ridge(alpha=cfg["ridge_alpha"])
    probe.fit(X_train, y_train)

    y_pred_val = probe.predict(X_val)
    r2 = float(probe.score(X_val, y_val))
    r, p = pearsonr(y_val.astype(float), y_pred_val.astype(float))
    y_pred_train = probe.predict(X_train)
    r2_train = float(probe.score(X_train, y_train))
    r_train, _ = pearsonr(y_train.astype(float), y_pred_train.astype(float))

    print(f"Validation (n={len(val_idx):,}):")
    print(f"  R²        = {r2:.4f}")
    print(f"  Pearson r = {r:.4f}  (p={p:.2e})")
    print(f"Train (sanity): R²={r2_train:.4f}  r={r_train:.4f}")
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
        "mode": "attribution_scores",
        "val_r2": r2,
        "val_pearson_r": float(r),
        "train_r2": r2_train,
        "embeddings_path": embeddings_path,
        "embedding_source": embedding_source,
        "scores_source": scores_source,
    }
    (out_dir / f"probe_meta_{probe_name}.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved probe → {probe_path}")


def main() -> None:
    cfg = CONFIG
    np.random.seed(cfg["seed"])

    device, dtype = resolve_device_dtype(cfg["device"])
    print(f"Device: {device}, dtype: {dtype}")

    manifest = json.loads(Path(cfg["manifest_path"]).read_text())
    base_model = manifest["base_model"]
    embedding_source = resolve_embedding_source(cfg, base_model)
    print(
        "Embedding source: "
        f"{embedding_source['mode']}"
        + (f" ({embedding_source['adapter_path']})" if embedding_source["adapter_path"] else "")
    )

    pool_path = Path(manifest["splits"]["score_pool"]["path"])
    pool_ds = load_from_disk(str(pool_path))
    print(f"Pool: {len(pool_ds):,} examples")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract embeddings once (reused for all probes) ---
    emb_path = out_dir / f"pool_embeddings_{embedding_source['fingerprint']}.npy"
    emb_meta_path = out_dir / f"pool_embeddings_{embedding_source['fingerprint']}.json"
    if emb_path.exists():
        print(f"\nReusing cached embeddings from {emb_path}")
        embeddings = np.load(str(emb_path))
        print(f"Embeddings: {embeddings.shape}  (dtype={embeddings.dtype})")
    else:
        print(f"\nLoading embedding model from {embedding_source['mode']} source ...")
        captured: dict[str, torch.Tensor] = {}

        def hook_fn(_module: Any, _input: Any, output: Any):
            captured["acts"] = output[0] if isinstance(output, tuple) else output

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        if embedding_source["mode"] == "adapter":
            model = PeftModel.from_pretrained(
                base,
                embedding_source["adapter_path"],
                is_trainable=False,
                autocast_adapter_dtype=False,
            )
        else:
            model = base
        model.eval()
        model.config.use_cache = False
        layers = get_transformer_layers_for_hook(model)
        layers[cfg["extraction_layer"]].register_forward_hook(hook_fn)
        print(f"Hook registered at layer {cfg['extraction_layer']}")

        print(f"\nExtracting residual stream embeddings ...")
        embeddings = extract_embeddings(pool_ds, model, captured, device, cfg["batch_size"])
        print(f"Embeddings: {embeddings.shape}  (dtype={embeddings.dtype})")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        np.save(str(emb_path), embeddings)
        emb_meta_path.write_text(json.dumps({
            "embeddings_path": str(emb_path),
            "embedding_source": embedding_source,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }, indent=2))
        print(f"Saved embeddings → {emb_path}")

    # --- Train probes ---
    use_quality_labels = cfg.get("use_quality_labels", False)
    probe_specs = cfg.get("probes")  # list[{"name": str, "scores_dir": str}] or None

    if use_quality_labels:
        if "quality" not in pool_ds.column_names:
            raise ValueError("Pool dataset has no 'quality' column.")
        quality_ok = {"good", "excellent"}
        scores_y = np.array(
            [1.0 if q in quality_ok else 0.0 for q in pool_ds["quality"]], dtype=np.float32
        )
        pos = int(scores_y.sum())
        print(f"Quality labels: {pos:,} positive, {len(scores_y)-pos:,} negative")
        fit_probe(
            embeddings,
            scores_y,
            cfg,
            "quality_labels",
            out_dir,
            "quality_labels",
            base_model,
            str(emb_path),
            embedding_source,
        )

    elif probe_specs:
        for spec in probe_specs:
            name, scores_dir = spec["name"], spec["scores_dir"]
            scores_y = load_scores_from_dir(scores_dir, len(pool_ds))
            print(f"\nScores [{name}]: range [{scores_y.min():.4f}, {scores_y.max():.4f}]  mean={scores_y.mean():.4f}")
            fit_probe(
                embeddings,
                scores_y,
                cfg,
                name,
                out_dir,
                str(Path(scores_dir) / "row_diagnostics.jsonl"),
                base_model,
                str(emb_path),
                embedding_source,
            )

    else:
        scores_y = load_scores_from_dir(cfg["scores_dir"], len(pool_ds))
        print(f"Loaded {len(scores_y):,} attribution scores")
        print(f"  range [{scores_y.min():.4f}, {scores_y.max():.4f}]  mean={scores_y.mean():.4f}")
        fit_probe(
            embeddings,
            scores_y,
            cfg,
            "probe",
            out_dir,
            str(Path(cfg["scores_dir"]) / "row_diagnostics.jsonl"),
            base_model,
            str(emb_path),
            embedding_source,
        )
    print("\nNext: uv run generate_continued_dataset.py")


if __name__ == "__main__":
    main()
