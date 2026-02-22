"""Score new SmolTalk examples with the residual stream probe and produce quality + random arms.

Streams the next unselected block of SmolTalk (past the rows consumed by build_sft_data.py),
runs each example through the base model, extracts the layer-17 residual stream, and scores
it with the probe trained in probe.py. Produces two pre-tokenized HF datasets:

  continuation/quality/  — top 10% by probe score
  continuation/random/   — random 10k drawn from the non-quality remainder

Both are ready for finetune.py (input_ids + labels + length columns).

Usage (run AFTER probe.py):

    uv run generate_continued_dataset.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt

CONFIG = {
    "manifest_path": "runs/smoltalk_v1/manifest.json",
    "probe_dir": "runs/smoltalk_v1/probe",
    "output_dir": "runs/smoltalk_v1/continuation",
    "extraction_layer": 17,    # must match probe.py
    "pool_size": 200_000,      # new SmolTalk rows to score
    "quality_size": 50_000,    # top-N by probe score → quality arm
    "random_size": 50_000,     # random-N from the non-quality remainder → baseline
    "seed": 42,
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


def get_embedding(
    input_ids: list[int],
    labels: list[int],
    model: Any,
    captured: dict,
    device: str,
) -> np.ndarray:
    """Single forward pass → residual stream at the last response token position.

    For a decoder-only model the last response token has attended to every preceding
    token via causal attention, making it the most information-dense single position.
    This matches the pooling strategy in probe.py and standard reward model heads.
    """
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=ids_t)
    hidden = captured["acts"].squeeze(0).float().cpu()   # (seq_len, d_model)

    lbl_t = torch.tensor(labels, dtype=torch.long)
    resp_positions = (lbl_t != -100).nonzero(as_tuple=True)[0]
    last_idx = int(resp_positions[-1]) if len(resp_positions) > 0 else len(labels) - 1
    return hidden[last_idx].numpy()


def estimate_skip_rows(manifest: dict) -> int:
    """Estimate raw rows consumed when raw_rows_consumed is missing from manifest."""
    needed = (
        manifest["splits"]["train"]["rows"]
        + manifest["splits"]["val"]["rows"]
        + manifest["splits"]["score_pool"]["rows"]
    )
    # gather_target = 2 * needed; ~63.5% pass rate; 10% safety margin
    return int((needed * 2) / 0.635 * 1.10)


def main() -> None:
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])

    device, dtype = resolve_device(cfg["device"])
    print(f"Device: {device}, dtype: {dtype}")

    manifest = json.loads(Path(cfg["manifest_path"]).read_text())
    base_model = manifest["base_model"]
    max_length = manifest["max_length"]
    dataset_name = manifest["dataset"]
    dataset_config = manifest["dataset_config"]

    skip_rows = manifest.get("raw_rows_consumed")
    if skip_rows is None:
        skip_rows = estimate_skip_rows(manifest)
        print(f"WARNING: manifest missing raw_rows_consumed — estimated skip={skip_rows:,}")
    else:
        print(f"Skipping first {skip_rows:,} raw SmolTalk rows (used by build_sft_data.py)")

    # --- Load probe ---
    probe_path = Path(cfg["probe_dir"]) / "probe.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}\nRun probe.py first.")
    with probe_path.open("rb") as f:
        probe = pickle.load(f)
    print(f"Loaded probe from {probe_path}")

    # --- IT tokenizer ---
    it_model = base_model.replace("-pt", "") + "-it"
    tokenizer = AutoTokenizer.from_pretrained(it_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model + hook ---
    print(f"\nLoading {base_model} ...")
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, attn_implementation="eager"
    )
    model.eval()
    model.config.use_cache = False
    model.model.layers[cfg["extraction_layer"]].register_forward_hook(hook_fn)
    print(f"Hook at layer {cfg['extraction_layer']}")

    # --- Stream and score ---
    print(f"\nStreaming {dataset_name}/{dataset_config}, skipping {skip_rows:,} rows ...")
    raw = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    pool_rows: list[dict] = []
    pool_scores: list[float] = []
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

        emb = get_embedding(tok_out["input_ids"], tok_out["labels"], model, captured, device)
        score = float(probe.predict(emb.reshape(1, -1))[0])

        pool_rows.append(tok_out)
        pool_scores.append(score)

        if len(pool_rows) % 1_000 == 0:
            print(f"  scored {len(pool_rows):,}/{cfg['pool_size']:,}", flush=True)
        if len(pool_rows) >= cfg["pool_size"]:
            break

    n_pool = len(pool_rows)
    scores_arr = np.array(pool_scores, dtype=np.float32)
    print(f"\nCollected and scored {n_pool:,} examples")
    print(f"  score range: [{scores_arr.min():.4f}, {scores_arr.max():.4f}]  mean={scores_arr.mean():.4f}")

    quality_size = cfg["quality_size"]
    random_size = cfg["random_size"]
    if n_pool < quality_size + random_size:
        raise RuntimeError(
            f"Only {n_pool} examples; need at least {quality_size + random_size}. "
            "Reduce quality_size/random_size or increase pool_size."
        )

    # Quality arm: top-N in stream order (sort back so examples are chronologically ordered)
    top_idx = np.argsort(scores_arr)[::-1][:quality_size]
    quality_idx = sorted(top_idx.tolist())
    quality_rows = [pool_rows[i] for i in quality_idx]
    threshold = float(scores_arr[top_idx[-1]])
    print(f"\nQuality arm: top-{quality_size} (score >= {threshold:.4f})")
    print(f"  score range: [{scores_arr[quality_idx].min():.4f}, {scores_arr[quality_idx].max():.4f}]")

    # Random arm: drawn from the non-quality remainder for a controlled comparison
    non_quality_idx = np.setdiff1d(np.arange(n_pool), top_idx)
    rand_chosen = sorted(rng.choice(non_quality_idx, size=random_size, replace=False).tolist())
    random_rows = [pool_rows[i] for i in rand_chosen]
    print(f"Random arm: {random_size} sampled from {len(non_quality_idx):,} non-quality examples")

    # --- Save ---
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_arm(rows: list[dict], name: str) -> dict:
        ds = Dataset.from_list(rows)
        path = out_dir / name
        ds.save_to_disk(str(path))
        n_tok = sum(r["length"] for r in rows)
        print(f"  {name}: {len(ds):,} rows, {n_tok / 1e6:.1f}M tokens → {path}")
        return {"path": str(path), "rows": len(ds), "total_tokens": n_tok}

    print("\nSaving ...")
    quality_info = save_arm(quality_rows, "quality")
    random_info = save_arm(random_rows, "random")

    cont_manifest = {
        "base_model": base_model,
        "max_length": max_length,
        "extraction_layer": cfg["extraction_layer"],
        "probe_path": str(probe_path),
        "pool_size": n_pool,
        "quality_size": quality_size,
        "random_size": random_size,
        "score_stats": {
            "pool_mean": float(scores_arr.mean()),
            "pool_min": float(scores_arr.min()),
            "pool_max": float(scores_arr.max()),
            "quality_mean": float(scores_arr[quality_idx].mean()),
            "quality_threshold": threshold,
        },
        "arms": {
            "quality": quality_info,
            "random": random_info,
        },
    }
    cont_path = out_dir / "continuation_manifest.json"
    cont_path.write_text(json.dumps(cont_manifest, indent=2))
    print(f"\nManifest → {cont_path}")

    print("\nNext steps:")
    print(f"  uv run finetune.py --train-data {quality_info['path']} "
          f"--resume-adapter runs/smoltalk_v1/adapter "
          f"--output-dir runs/smoltalk_v1/adapter_quality")
    print(f"  uv run finetune.py --train-data {random_info['path']} "
          f"--resume-adapter runs/smoltalk_v1/adapter "
          f"--output-dir runs/smoltalk_v1/adapter_random")


if __name__ == "__main__":
    main()
