from __future__ import annotations

"""Extract SAE features from top/bottom attribution-ranked FineWeb chunks.

Reads row_diagnostics.jsonl from a completed score.py run, labels the top and
bottom `top_frac`/`bottom_frac` of the pool by attribution score, then runs
each labeled chunk through the pretrained model + GemmaScope SAE to extract
sparse top-K features and global activation statistics.

Output (saved to output_dir/):
  {sae_id}.npz           — feat_ids (N,K), feat_vals (N,K), masks (N,K),
                            global_stats (N,10), labels (N,), stat_names
  examples.jsonl         — per-row metadata: row_id, pool_index, score, label
  summary.json           — config + run statistics
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sae_lens import SAE
from transformers import AutoModelForCausalLM

CONFIG = {
    "base_model": "google/gemma-3-1b-pt",
    "sae_release": "gemma-scope-2-1b-pt-res",  # verify exact release name in sae_lens
    "sae_id": "layer_12_width_16k_l0_small",   # verify sae_id for 1B
    "layer_idx": 12,
    "diagnostics_path": "runs/smoltalk_v1/scores/row_diagnostics.jsonl",
    "pool_path": "runs/smoltalk_v1/data/attr_pool",
    "output_dir": "runs/smoltalk_v1/sae_features/layer12_width16k",
    "top_frac": 0.10,
    "bottom_frac": 0.10,
    "topk_features": 256,
    "device": "auto",
}

STAT_NAMES = [
    "seq_len",
    "active_entry_fraction",
    "active_feature_fraction",
    "mean_active_activation",
    "std_active_activation",
    "max_active_activation",
    "token_active_mean",
    "token_active_std",
    "token_active_cv",
    "feature_concentration_hhi",
]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str) -> tuple[str, torch.dtype]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cpu", torch.float32
    if device_arg == "cuda":
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device_arg, torch.float32


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels(
    diagnostics_path: str,
    top_frac: float,
    bottom_frac: float,
) -> list[dict]:
    """Load attribution scores and label top/bottom fraction.

    Returns a list of dicts with keys: pool_index, row_id, score, label.
    Middle fraction is discarded.
    """
    records: list[dict] = []
    with open(diagnostics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({
                "pool_index": int(rec["index"]),
                "row_id": str(rec["row_id"]),
                "score": float(rec["score"]),
            })

    n = len(records)
    scores = np.array([r["score"] for r in records], dtype=np.float32)
    order = np.argsort(scores)  # ascending

    n_top = int(round(n * top_frac))
    n_bottom = int(round(n * bottom_frac))

    bottom_indices = set(order[:n_bottom].tolist())
    top_indices = set(order[n - n_top:].tolist())

    labeled: list[dict] = []
    for i, rec in enumerate(records):
        if i in top_indices:
            labeled.append({**rec, "label": 1})
        elif i in bottom_indices:
            labeled.append({**rec, "label": 0})

    print(
        f"Loaded {n:,} records → {sum(1 for r in labeled if r['label']==1):,} top "
        f"+ {sum(1 for r in labeled if r['label']==0):,} bottom = {len(labeled):,} labeled"
    )
    return labeled


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_topk(
    sae_acts: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract top-K features by mean activation across the sequence.

    Args:
        sae_acts: (1, seq_len, d_sae) — JumpReLU SAE output
        k: number of features to extract

    Returns:
        feat_ids:  (k,) int64  — feature indices (padded with 0)
        feat_vals: (k,) float32 — mean activation (padded with 0)
        mask:      (k,) bool    — True where real features
    """
    mean_acts = sae_acts.squeeze(0).clamp(min=0).mean(dim=0)  # (d_sae,)
    n_active = int((mean_acts > 0).sum().item())
    actual_k = min(k, n_active)

    feat_ids = torch.zeros(k, dtype=torch.long)
    feat_vals = torch.zeros(k, dtype=torch.float32)
    mask = torch.zeros(k, dtype=torch.bool)

    if actual_k > 0:
        topk = torch.topk(mean_acts, k=actual_k)
        feat_ids[:actual_k] = topk.indices
        feat_vals[:actual_k] = topk.values
        mask[:actual_k] = True

    return feat_ids, feat_vals, mask


def compute_global_stats(sae_acts: torch.Tensor) -> np.ndarray:
    """Compute 10 global statistics from SAE activations.

    Args:
        sae_acts: (1, seq_len, d_sae)

    Returns:
        stats: (10,) float32 matching STAT_NAMES order
    """
    acts = sae_acts.squeeze(0).to(torch.float32)  # (seq_len, d_sae)
    active = acts > 0

    seq_len = float(acts.shape[0])
    total_entries = float(active.numel())
    total_active = float(active.sum().item())

    active_entry_fraction = total_active / max(total_entries, 1.0)
    active_feature_fraction = float((active.any(dim=0)).float().mean().item())

    token_active_counts = active.sum(dim=1).float()  # (seq_len,)
    token_active_mean = float(token_active_counts.mean().item())
    token_active_std = float(token_active_counts.std(unbiased=False).item())
    token_active_cv = token_active_std / (token_active_mean + 1e-8)

    if total_active > 0:
        active_vals = acts[active]
        mean_active = float(active_vals.mean().item())
        std_active = float(active_vals.std(unbiased=False).item())
        max_active = float(active_vals.max().item())
    else:
        mean_active = std_active = max_active = 0.0

    feature_counts = active.float().sum(dim=0)  # (d_sae,)
    feat_sum = float(feature_counts.sum().item())
    if feat_sum > 0:
        p = feature_counts / feat_sum
        hhi = float((p * p).sum().item())
    else:
        hhi = 0.0

    return np.array([
        seq_len,
        active_entry_fraction,
        active_feature_fraction,
        mean_active,
        std_active,
        max_active,
        token_active_mean,
        token_active_std,
        token_active_cv,
        hhi,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = resolve_device(cfg["device"])
    print(f"Device: {device}, dtype: {dtype}")

    # --- Labels ---
    labeled = load_labels(cfg["diagnostics_path"], cfg["top_frac"], cfg["bottom_frac"])

    # --- Pool dataset ---
    print(f"Loading pool from {cfg['pool_path']} ...")
    pool_ds = load_from_disk(cfg["pool_path"])
    print(f"  Pool size: {len(pool_ds):,} chunks")

    # --- Model ---
    print(f"Loading model from {cfg['base_model']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    # Register hook to capture residual stream after layer layer_idx
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _input, output):
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    hook = model.model.layers[cfg["layer_idx"]].register_forward_hook(hook_fn)  # type: ignore[union-attr,index]

    # --- SAE ---
    print(f"Loading SAE {cfg['sae_release']} / {cfg['sae_id']} ...")
    sae_obj: SAE = SAE.from_pretrained(  # type: ignore[assignment]
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )
    sae_obj.eval()
    d_sae = int(sae_obj.cfg.d_sae)  # type: ignore[union-attr]
    k = cfg["topk_features"]
    print(f"  SAE d_sae={d_sae}, extracting top-{k} features")

    # --- Allocate output arrays ---
    n = len(labeled)
    all_feat_ids = np.zeros((n, k), dtype=np.int32)
    all_feat_vals = np.zeros((n, k), dtype=np.float32)
    all_masks = np.zeros((n, k), dtype=np.bool_)
    all_stats = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)
    all_labels = np.zeros(n, dtype=np.int32)

    # --- Process ---
    for i, rec in enumerate(labeled):
        pool_idx = rec["pool_index"]
        row = pool_ds[pool_idx]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device).unsqueeze(0)

        with torch.inference_mode():
            model(input_ids=input_ids)
            acts = captured["acts"]  # (1, seq_len, d_model)
            # Move acts to SAE device if needed
            sae_acts = sae_obj.encode(acts.to(next(sae_obj.parameters()).device))  # (1, seq_len, d_sae)

        feat_ids, feat_vals, mask = extract_topk(sae_acts.cpu(), k)
        stats = compute_global_stats(sae_acts.cpu())

        all_feat_ids[i] = feat_ids.numpy().astype(np.int32)
        all_feat_vals[i] = feat_vals.numpy()
        all_masks[i] = mask.numpy()
        all_stats[i] = stats
        all_labels[i] = rec["label"]

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  processed {i + 1}/{n}", end="\r", flush=True)

    hook.remove()
    print(f"\nDone. Processed {n} examples.")

    # --- Save NPZ ---
    npz_path = output_dir / f"{cfg['sae_id']}.npz"
    np.savez_compressed(
        npz_path,
        feat_ids=all_feat_ids,
        feat_vals=all_feat_vals,
        masks=all_masks,
        global_stats=all_stats,
        labels=all_labels,
        stat_names=np.array(STAT_NAMES, dtype="<U64"),
    )
    print(f"Saved features → {npz_path}")
    print(f"  feat_ids: {all_feat_ids.shape}, feat_vals: {all_feat_vals.shape}")
    print(f"  labels: {all_labels.sum()} positive, {(all_labels==0).sum()} negative")

    # --- Save examples.jsonl ---
    examples_path = output_dir / "examples.jsonl"
    with examples_path.open("w") as f:
        for i, rec in enumerate(labeled):
            f.write(json.dumps({
                "array_index": i,
                "pool_index": rec["pool_index"],
                "row_id": rec["row_id"],
                "score": rec["score"],
                "label": rec["label"],
            }) + "\n")
    print(f"Saved metadata → {examples_path}")

    # --- Save summary ---
    summary = {
        "config": cfg,
        "n_total": n,
        "n_positive": int(all_labels.sum()),
        "n_negative": int((all_labels == 0).sum()),
        "d_sae": d_sae,
        "topk_features": k,
        "npz_path": str(npz_path),
        "examples_path": str(examples_path),
        "stat_names": STAT_NAMES,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
