"""Apply trained SAE classifier to new SmolTalk examples → quality vs random datasets.

Reads the manifest from build_sft_data.py, streams the *next* portion of SmolTalk
(skipping all rows already consumed by build_sft_data.py), scores each example using
the SAE fingerprint classifier, and produces two HF datasets:

  continuation/quality/  — top-N by classifier score
  continuation/random/   — random-N from the same scored pool (controlled comparison)

Both are pre-tokenized (input_ids + labels, same format as train split) and ready
for finetune.py.

Usage:
    uv run build_continuation_data.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt
from train_bidir_classifier import SAEFingerprint

CONFIG = {
    "manifest_path": "runs/smoltalk_v1/manifest.json",
    # Classifier: K must match the best-K chosen from train_bidir_classifier.py ablation
    "classifier_k": 256,
    "classifier_dir": "runs/smoltalk_v1/sae_classifier",
    # SAE — update sae_release/sae_id to match sae_analysis.py CONFIG
    "sae_release": "gemma-scope-2-1b-pt-res",  # verify exact name in sae_lens
    "sae_id": "layer_12_width_16k_l0_small",   # verify for 1B
    "layer_idx": 12,
    # Classifier architecture — must match train_bidir_classifier.py CONFIG
    "d_sae": 16384,
    "d_embed": 64,
    "d_hidden": 128,
    # How many new SmolTalk rows to score with the classifier
    "pool_size": 50_000,
    # Size of each arm (quality and random draw from the same pool)
    "continuation_size": 20_000,
    "output_dir": "runs/smoltalk_v1/continuation",
    "seed": 42,
    "device": "auto",
}


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
# SAE feature extraction (single example)
# ---------------------------------------------------------------------------

def extract_sae_features(
    input_ids: list[int],
    model: Any,
    sae: SAE,
    captured: dict,
    device: str,
    dtype: torch.dtype,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feat_ids, feat_vals, mask) arrays of shape (k,) for one example."""
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=ids_t)
    acts = captured["acts"]  # (1, seq_len, d_model)

    # Mean-pool over sequence positions (matches sae_analysis.py approach)
    acts_mean = acts.mean(dim=1)  # (1, d_model)

    with torch.inference_mode():
        sae_out = sae.encode(acts_mean.to(dtype))  # (1, d_sae)

    activations = sae_out[0].float().cpu().numpy()  # (d_sae,)

    # Top-k by activation value
    nonzero_idx = np.nonzero(activations)[0]
    if len(nonzero_idx) == 0:
        feat_ids = np.zeros(k, dtype=np.int32)
        feat_vals = np.zeros(k, dtype=np.float32)
        mask = np.zeros(k, dtype=bool)
        return feat_ids, feat_vals, mask

    top_idx = nonzero_idx[np.argsort(activations[nonzero_idx])[::-1][:k]]
    n_active = len(top_idx)

    feat_ids = np.zeros(k, dtype=np.int32)
    feat_vals = np.zeros(k, dtype=np.float32)
    mask = np.zeros(k, dtype=bool)
    feat_ids[:n_active] = top_idx
    feat_vals[:n_active] = activations[top_idx]
    mask[:n_active] = True

    return feat_ids, feat_vals, mask


# ---------------------------------------------------------------------------
# Classifier scoring
# ---------------------------------------------------------------------------

def score_examples(
    feat_ids_list: list[np.ndarray],
    feat_vals_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    classifier: SAEFingerprint,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Run classifier over all examples in batches. Returns logits (N,)."""
    n = len(feat_ids_list)
    all_logits: list[float] = []
    classifier.eval()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids = torch.from_numpy(np.stack(feat_ids_list[start:end]).astype(np.int64)).to(device)
            vals = torch.from_numpy(np.stack(feat_vals_list[start:end]).astype(np.float32)).to(device)
            masks = torch.from_numpy(np.stack(mask_list[start:end]).astype(bool)).to(device)
            logits = classifier(ids, vals, masks).cpu().numpy()
            all_logits.extend(logits.tolist())
    return np.array(all_logits, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG
    np.random.seed(cfg["seed"])

    device, dtype = resolve_device(cfg["device"])
    print(f"Device: {device}, dtype: {dtype}")

    # --- Manifest ---
    manifest = json.loads(Path(cfg["manifest_path"]).read_text())
    base_model = manifest["base_model"]
    max_length = manifest["max_length"]
    dataset_name = manifest["dataset"]
    dataset_config = manifest["dataset_config"]

    # How many raw rows to skip — saved by build_sft_data.py
    skip_rows = manifest.get("raw_rows_consumed")
    if skip_rows is None:
        # Fallback: estimate from known pass rate and gather_target
        # gather_target = 2 * (train + val + pool). At ~63.5% pass rate:
        # skip ≈ gather_target / 0.635. Round up generously.
        print("WARNING: manifest missing raw_rows_consumed. Estimating skip from pass rate.")
        needed = (
            manifest["splits"]["train"]["size"]
            + manifest["splits"]["val"]["size"]
            + manifest["splits"]["score_pool"]["size"]
        )
        skip_rows = int((needed * 2) / 0.635 * 1.10)  # 10% safety margin
    print(f"Skipping first {skip_rows:,} raw SmolTalk rows (already used by build_sft_data.py)")

    # --- IT tokenizer (from base model name) ---
    it_model = base_model.replace("-pt", "") + "-it"
    tokenizer = AutoTokenizer.from_pretrained(it_model)

    # --- Base model + SAE hook ---
    print(f"Loading base model: {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, attn_implementation="eager"
    )
    model.eval()
    model.config.use_cache = False

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    model.model.layers[cfg["layer_idx"]].register_forward_hook(hook_fn)  # type: ignore[union-attr,index]

    print(f"Loading SAE {cfg['sae_release']} / {cfg['sae_id']} ...")
    sae_obj: SAE = SAE.from_pretrained(  # type: ignore[assignment]
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )
    sae_obj.eval()
    k = cfg["classifier_k"]

    # --- Classifier ---
    classifier_path = Path(cfg["classifier_dir"]) / f"K{k}" / "best_model.pt"
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier not found: {classifier_path}\n"
            "Run train_bidir_classifier.py first and check the best K."
        )
    print(f"Loading classifier from {classifier_path} ...")
    classifier = SAEFingerprint(
        d_sae=cfg["d_sae"],
        d_embed=cfg["d_embed"],
        d_hidden=cfg["d_hidden"],
        dropout=0.0,  # no dropout at inference
    )
    classifier.load_state_dict(torch.load(str(classifier_path), map_location=device))
    classifier = classifier.to(device)
    classifier.eval()

    # --- Stream SmolTalk, skip already-consumed rows ---
    print(f"Streaming SmolTalk {dataset_config}, skipping {skip_rows:,} rows ...")
    raw = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    pool_rows: list[dict] = []       # tokenized rows (input_ids, labels, magpie_score)
    pool_feat_ids: list[np.ndarray] = []
    pool_feat_vals: list[np.ndarray] = []
    pool_masks: list[np.ndarray] = []

    raw_seen = 0
    for row in raw:
        raw_seen += 1
        if raw_seen <= skip_rows:
            if raw_seen % 100_000 == 0:
                print(f"  skipping... {raw_seen:,}/{skip_rows:,}", flush=True)
            continue

        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, max_length)
        if tok_out is None:
            continue
        tok_out["magpie_score"] = _get_magpie_score(row)

        feat_ids, feat_vals, mask = extract_sae_features(
            tok_out["input_ids"], model, sae_obj, captured, device, dtype, k
        )

        pool_rows.append(tok_out)
        pool_feat_ids.append(feat_ids)
        pool_feat_vals.append(feat_vals)
        pool_masks.append(mask)

        if len(pool_rows) % 1000 == 0:
            print(f"  scored {len(pool_rows):,}/{cfg['pool_size']:,}", flush=True)
        if len(pool_rows) >= cfg["pool_size"]:
            break

    n_pool = len(pool_rows)
    if n_pool < cfg["continuation_size"]:
        raise RuntimeError(
            f"Only {n_pool} valid rows collected; need at least {cfg['continuation_size']}. "
            "Reduce continuation_size or increase pool_size."
        )
    print(f"Collected {n_pool:,} examples. Running classifier ...")

    # --- Score pool with classifier ---
    scores = score_examples(pool_feat_ids, pool_feat_vals, pool_masks, classifier, device)
    print(f"  score range: [{scores.min():.3f}, {scores.max():.3f}]  mean={scores.mean():.3f}")

    # --- Quality arm: top-N by score, saved in original stream order ---
    cont_size = cfg["continuation_size"]
    top_n_idx = np.argsort(scores)[::-1][:cont_size]
    quality_idx = sorted(top_n_idx.tolist())  # sort by original position → natural stream order
    quality_rows = [pool_rows[i] for i in quality_idx]
    print(f"\nQuality arm: top-{cont_size} by classifier score (saved in stream order)")
    print(f"  score range: [{scores[quality_idx].min():.3f}, {scores[quality_idx].max():.3f}]")

    # --- Random arm: just the next N valid examples from the stream (no scoring) ---
    # The streaming iterator `raw` advanced past the pool — continue from where it left off.
    print(f"\nCollecting random arm: next {cont_size} valid examples from stream ...")
    random_rows: list[dict] = []
    for row in raw:
        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, max_length)
        if tok_out is None:
            continue
        tok_out["magpie_score"] = _get_magpie_score(row)
        random_rows.append(tok_out)
        if len(random_rows) % 1000 == 0:
            print(f"  collected {len(random_rows):,}/{cont_size:,}", flush=True)
        if len(random_rows) >= cont_size:
            break

    if len(random_rows) < cont_size:
        raise RuntimeError(
            f"Only {len(random_rows)} examples for random arm; need {cont_size}. "
            "SmolTalk may be exhausted — reduce continuation_size."
        )
    print(f"Random arm: {len(random_rows)} next-in-stream examples (no selection)")

    # --- Save ---
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_arm(rows: list[dict], name: str) -> dict:
        ds = Dataset.from_list(rows)
        path = out_dir / name
        ds.save_to_disk(str(path))
        n_tokens = sum(r["length"] for r in rows)
        print(f"  {name}: {len(ds):,} rows, {n_tokens / 1e6:.1f}M tokens → {path}")
        return {"path": str(path), "size": len(ds), "n_tokens": n_tokens}

    print("\nSaving datasets ...")
    quality_info = save_arm(quality_rows, "quality")
    random_info = save_arm(random_rows, "random")

    scores_info = {
        "pool_mean": float(scores.mean()),
        "pool_min": float(scores.min()),
        "pool_max": float(scores.max()),
        "quality_mean": float(scores[quality_idx].mean()),
        "quality_min": float(scores[quality_idx].min()),
        "quality_max": float(scores[quality_idx].max()),
        # random arm is unscored — it's the uncurated next-in-stream baseline
    }

    cont_manifest = {
        "base_model": base_model,
        "max_length": max_length,
        "classifier_k": k,
        "classifier_path": str(classifier_path),
        "pool_size": n_pool,
        "continuation_size": cont_size,
        "arms": {
            "quality": quality_info,
            "random": random_info,
        },
        "score_stats": scores_info,
    }
    manifest_path = out_dir / "continuation_manifest.json"
    manifest_path.write_text(json.dumps(cont_manifest, indent=2))
    print(f"\nManifest written to: {manifest_path}")

    print("\nNext steps:")
    print(f"  uv run finetune.py --train-data {quality_info['path']} "
          f"--resume-adapter runs/smoltalk_v1/adapter "
          f"--output-dir runs/smoltalk_v1/adapter_quality")
    print(f"  uv run finetune.py --train-data {random_info['path']} "
          f"--resume-adapter runs/smoltalk_v1/adapter "
          f"--output-dir runs/smoltalk_v1/adapter_random")


if __name__ == "__main__":
    main()
