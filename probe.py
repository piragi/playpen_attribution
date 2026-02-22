"""Train a Ridge Regression probe to predict Bergson attribution scores from residual stream.

Runs the base model (gemma-3-1b-pt) forward pass on the 50k attr_pool, extracts the
mean-pooled residual stream at `extraction_layer` over response token positions only
(labels != -100), and fits a Ridge Regression to the continuous Bergson attribution scores.

Reports R² and Pearson r on a held-out 20% split. Saves probe.pkl and the full embedding
matrix (reused by generate_continued_dataset.py to avoid re-extracting).

Run AFTER score.py:

    uv run score.py --adapter-path runs/smoltalk_v1/adapter \\
        --output-dir runs/smoltalk_v1/scores
    uv run probe.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM

CONFIG = {
    "manifest_path": "runs/smoltalk_v1/manifest.json",
    "scores_dir": "runs/smoltalk_v1/scores",
    "output_dir": "runs/smoltalk_v1/probe",
    # Layer to extract from (0-indexed). GemmaScope chose layer 17 for gemma-3-1b-pt;
    # start there. Try 14 if R² is low.
    "extraction_layer": 17,
    "ridge_alpha": 1.0,
    "val_frac": 0.20,
    "seed": 42,
    "batch_size": 16,
    "device": "auto",
}


def resolve_device(device_arg: str) -> tuple[str, torch.dtype]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cpu", torch.float32
    if device_arg == "cuda":
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device_arg, torch.float32


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

            # Pad to max length within this mini-batch
            max_len = max(len(ids) for ids in batch["input_ids"])
            ids_padded, lbl_padded = [], []
            for ids, lbls in zip(batch["input_ids"], batch["labels"]):
                pad = max_len - len(ids)
                ids_padded.append(ids + [0] * pad)
                lbl_padded.append(lbls + [-100] * pad)

            ids_t = torch.tensor(ids_padded, dtype=torch.long, device=device)
            lbl_t = torch.tensor(lbl_padded, dtype=torch.long)   # keep on CPU for index finding

            model(input_ids=ids_t)
            hidden = captured["acts"]  # (batch, seq_len, d_model)

            # Last response token: last position where labels != -100
            # If somehow no response tokens exist, fall back to the final padded position.
            batch_size_actual = lbl_t.shape[0]
            last_resp_idx = torch.zeros(batch_size_actual, dtype=torch.long)
            for i in range(batch_size_actual):
                resp_positions = (lbl_t[i] != -100).nonzero(as_tuple=True)[0]
                if len(resp_positions) > 0:
                    last_resp_idx[i] = resp_positions[-1]
                else:
                    last_resp_idx[i] = lbl_t.shape[1] - 1

            # Gather the hidden state at each example's last response position
            idx = last_resp_idx.to(hidden.device).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, 1, hidden.shape[-1])           # (batch, 1, d_model)
            pooled = hidden.gather(dim=1, index=idx).squeeze(1) # (batch, d_model)

            all_embeddings.append(pooled.float().cpu().numpy())

            if start % (batch_size * 50) == 0 and start > 0:
                print(f"  {end:,}/{n:,} processed ...", flush=True)

    return np.concatenate(all_embeddings, axis=0)   # (n, d_model)


def main() -> None:
    cfg = CONFIG
    np.random.seed(cfg["seed"])

    device, dtype = resolve_device(cfg["device"])
    print(f"Device: {device}, dtype: {dtype}")

    manifest = json.loads(Path(cfg["manifest_path"]).read_text())
    base_model = manifest["base_model"]

    # --- Load Bergson scores ---
    diag_path = Path(cfg["scores_dir"]) / "row_diagnostics.jsonl"
    if not diag_path.exists():
        raise FileNotFoundError(
            f"Scores not found: {diag_path}\n"
            "Run score.py first:\n"
            "  uv run score.py --adapter-path runs/smoltalk_v1/adapter "
            "--output-dir runs/smoltalk_v1/scores"
        )
    records = [json.loads(l) for l in diag_path.read_text().splitlines() if l.strip()]
    records.sort(key=lambda r: r["index"])
    scores_y = np.array([r["score"] for r in records], dtype=np.float32)
    print(f"Loaded {len(scores_y):,} attribution scores")
    print(f"  range [{scores_y.min():.4f}, {scores_y.max():.4f}]  mean={scores_y.mean():.4f}")

    # --- Load pool dataset ---
    pool_path = Path(manifest["splits"]["score_pool"]["path"])
    pool_ds = load_from_disk(str(pool_path))
    if len(pool_ds) != len(scores_y):
        raise ValueError(
            f"Pool size {len(pool_ds)} != scores size {len(scores_y)}. "
            "Re-run score.py against the current attr_pool."
        )
    print(f"Pool: {len(pool_ds):,} examples")

    # --- Load base model + hook ---
    print(f"\nLoading {base_model} ...")
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _input, output):
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, attn_implementation="eager"
    )
    model.eval()
    model.config.use_cache = False
    model.model.layers[cfg["extraction_layer"]].register_forward_hook(hook_fn)
    print(f"Hook registered at layer {cfg['extraction_layer']}")

    # --- Extract embeddings ---
    print(f"\nExtracting residual stream embeddings ...")
    embeddings = extract_embeddings(pool_ds, model, captured, device, cfg["batch_size"])
    print(f"Embeddings: {embeddings.shape}  (dtype={embeddings.dtype})")

    # Free GPU before sklearn
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Persist embeddings for downstream reuse ---
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "pool_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"Saved embeddings → {emb_path}")

    # --- Train Ridge probe ---
    idx = np.arange(len(embeddings))
    train_idx, val_idx = train_test_split(idx, test_size=cfg["val_frac"], random_state=cfg["seed"])

    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = scores_y[train_idx], scores_y[val_idx]

    print(f"\nFitting Ridge(alpha={cfg['ridge_alpha']}) on {len(train_idx):,} examples ...")
    probe = Ridge(alpha=cfg["ridge_alpha"])
    probe.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred_val = probe.predict(X_val)
    r2 = float(probe.score(X_val, y_val))
    r, p = pearsonr(y_val.astype(float), y_pred_val.astype(float))

    y_pred_train = probe.predict(X_train)
    r2_train = float(probe.score(X_train, y_train))
    r_train, _ = pearsonr(y_train.astype(float), y_pred_train.astype(float))

    print(f"\nValidation (n={len(val_idx):,}):")
    print(f"  R²        = {r2:.4f}")
    print(f"  Pearson r = {r:.4f}  (p={p:.2e})")
    print(f"Train (sanity): R²={r2_train:.4f}  r={r_train:.4f}")

    if r2 < 0.05:
        print(
            "\nWARNING: R² < 0.05 — probe is not learning the attribution direction well.\n"
            "  Try a different extraction_layer (e.g., 14) or check that score.py used\n"
            "  the current attr_query."
        )

    # --- Save probe ---
    probe_path = out_dir / "probe.pkl"
    with probe_path.open("wb") as f:
        pickle.dump(probe, f)
    print(f"\nSaved probe → {probe_path}")

    meta = {
        "base_model": base_model,
        "extraction_layer": cfg["extraction_layer"],
        "ridge_alpha": cfg["ridge_alpha"],
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_r2": r2,
        "val_pearson_r": float(r),
        "train_r2": r2_train,
        "embeddings_path": str(emb_path),
        "scores_source": str(diag_path),
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata → {out_dir / 'probe_meta.json'}")
    print("\nNext: uv run generate_continued_dataset.py")


if __name__ == "__main__":
    main()
