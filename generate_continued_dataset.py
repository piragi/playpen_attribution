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
import os
import pickle
from pathlib import Path

# Ensure transformers/datasets find the local model cache.
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt

CONFIG = {
    "manifest_path": "runs/smoltalk_v4/manifest.json",
    "probe_dir": "runs/smoltalk_v4/probe",
    # List of probe files to load. One quality arm is built per probe.
    # Set to None to fall back to single-probe mode using probe_filename.
    "probes": [
        {"name": "math_da", "filename": "probe_math_da.pkl"},
    ],
    "probe_filename": "probe.pkl",   # single-probe fallback
    "output_dir": "runs/smoltalk_v4/continuation",
    "extraction_layer": 17,    # must match probe.py
    # Only stream examples from these categories (set to None for all categories).
    "category_filter": {"math", "data-analysis"},
    "pool_size": 30_000,       # new SmolTalk rows to score with probe
    "quality_size": 10_000,    # top-N by each probe score → quality arm per probe
    "random_size": 10_000,     # length-stratified random from non-quality remainder
    "batch_size": 64,
    "seed": 42,
    "device": "auto",
}


def resolve_model_path(model_id: str) -> str:
    """Return local snapshot path if cached, otherwise return model_id for hub download."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / "hub"
    # HF cache layout: models--{org}--{name}/snapshots/<hash>/
    slug = "models--" + model_id.replace("/", "--")
    snapshots_dir = cache_dir / slug / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[-1])
    return model_id


def resolve_device(device_arg: str) -> tuple[str, torch.dtype]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cpu", torch.float32
    if device_arg == "cuda":
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device_arg, torch.float32


def score_batch(
    batch_rows: list[dict],
    probe: Any,
    model: Any,
    captured: dict,
    device: str,
) -> list[float]:
    """Batched forward pass → probe scores for a list of tokenized rows.

    Pads to the max length within the batch, runs one forward pass, then gathers
    the last response token (last index where labels != -100) for each example.
    Identical pooling strategy to probe.py and standard reward model heads.
    """
    max_len = max(len(r["input_ids"]) for r in batch_rows)
    ids_padded, lbl_padded = [], []
    for r in batch_rows:
        pad = max_len - len(r["input_ids"])
        ids_padded.append(r["input_ids"] + [0] * pad)
        lbl_padded.append(r["labels"] + [-100] * pad)

    ids_t = torch.tensor(ids_padded, dtype=torch.long, device=device)
    lbl_t = torch.tensor(lbl_padded, dtype=torch.long)  # CPU for index finding

    with torch.inference_mode():
        model(input_ids=ids_t)
    hidden = captured["acts"]  # (batch, seq_len, d_model)

    # Last response token per example
    last_resp_idx = torch.zeros(len(batch_rows), dtype=torch.long)
    for i in range(len(batch_rows)):
        resp_positions = (lbl_t[i] != -100).nonzero(as_tuple=True)[0]
        last_resp_idx[i] = resp_positions[-1] if len(resp_positions) > 0 else lbl_t.shape[1] - 1

    idx = last_resp_idx.to(hidden.device).unsqueeze(-1).unsqueeze(-1)
    idx = idx.expand(-1, 1, hidden.shape[-1])
    pooled = hidden.gather(dim=1, index=idx).squeeze(1).float().cpu().numpy()  # (batch, d_model)

    return [float(probe.predict(pooled[i].reshape(1, -1))[0]) for i in range(len(batch_rows))]


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

    # --- Load probe(s) ---
    probe_specs = cfg.get("probes")
    if probe_specs:
        probes = []
        for spec in probe_specs:
            p = Path(cfg["probe_dir"]) / spec["filename"]
            if not p.exists():
                raise FileNotFoundError(f"Probe not found: {p}\nRun probe.py first.")
            with p.open("rb") as f:
                probes.append({"name": spec["name"], "probe": pickle.load(f)})
        print(f"Loaded {len(probes)} probes: {[p['name'] for p in probes]}")
    else:
        probe_path = Path(cfg["probe_dir"]) / cfg.get("probe_filename", "probe.pkl")
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe not found: {probe_path}\nRun probe.py first.")
        with probe_path.open("rb") as f:
            probes = [{"name": "probe", "probe": pickle.load(f)}]
        print(f"Loaded probe from {probe_path}")

    # --- IT tokenizer ---
    it_model = (base_model[:-3] + "-it") if base_model.endswith("-pt") else (base_model + "-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(it_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model + hook ---
    model_path = resolve_model_path(base_model)
    print(f"\nLoading {base_model} (path: {model_path}) ...")
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()
    model.config.use_cache = False
    model.model.layers[cfg["extraction_layer"]].register_forward_hook(hook_fn)
    print(f"Hook at layer {cfg['extraction_layer']}")

    # --- Stream and score ---
    print(f"\nStreaming {dataset_name}/{dataset_config}, skipping {skip_rows:,} rows ...")
    raw = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    category_filter = cfg.get("category_filter")  # set of category strings or None
    pool_rows: list[dict] = []
    # pool_probe_scores[i] = list of scores, one per probe
    pool_probe_scores: list[list[float]] = []
    raw_seen = 0
    batch_size = cfg["batch_size"]
    buffer: list[dict] = []

    def flush_buffer() -> None:
        # Score each buffer row with all probes simultaneously.
        per_probe = [
            score_batch(buffer, p["probe"], model, captured, device)
            for p in probes
        ]
        pool_rows.extend(buffer)
        for i in range(len(buffer)):
            pool_probe_scores.append([per_probe[j][i] for j in range(len(probes))])
        buffer.clear()

    for row in raw:
        raw_seen += 1
        if raw_seen <= skip_rows:
            if raw_seen % 100_000 == 0:
                print(f"  skipping... {raw_seen:,}/{skip_rows:,}", flush=True)
            continue

        if category_filter and row.get("category", "") not in category_filter:
            continue

        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        tok_out = _mask_prompt(messages, tokenizer, max_length)
        if tok_out is None:
            continue
        tok_out["magpie_score"] = _get_magpie_score(row)
        tok_out["quality"] = row.get("quality", "")
        tok_out["category"] = row.get("category", "")

        buffer.append(tok_out)
        if len(buffer) >= batch_size:
            flush_buffer()
            if len(pool_rows) % 1_000 == 0:
                print(f"  scored {len(pool_rows):,}/{cfg['pool_size']:,}", flush=True)
        if len(pool_rows) >= cfg["pool_size"]:
            break

    if buffer:
        flush_buffer()

    n_pool = len(pool_rows)
    scores_matrix = np.array(pool_probe_scores, dtype=np.float32)  # (n_pool, n_probes)
    print(f"\nCollected and scored {n_pool:,} examples with {len(probes)} probe(s)")

    quality_size = cfg["quality_size"]
    random_size = cfg["random_size"]

    # Build one quality arm per probe (top-quality_size by that probe's scores).
    quality_idx_per_probe: list[list[int]] = []
    for j, p in enumerate(probes):
        col = scores_matrix[:, j]
        top_idx = np.argsort(col)[::-1][:quality_size]
        quality_idx_per_probe.append(sorted(top_idx.tolist()))
        threshold = float(col[top_idx[-1]])
        print(f"  Quality arm [{p['name']}]: top-{quality_size}  threshold={threshold:.4f}  "
              f"range [{col[top_idx].min():.4f}, {col[top_idx].max():.4f}]")

    # Random arm: length-stratified sample from examples not in ANY quality arm.
    all_quality = set(i for idxs in quality_idx_per_probe for i in idxs)
    remaining = [i for i in range(n_pool) if i not in all_quality]
    lengths = np.array([pool_rows[i]["length"] for i in remaining], dtype=np.float32)
    # Stratify by length quintile
    edges = np.unique(np.quantile(lengths, np.linspace(0, 1, 6)))
    if len(edges) > 2:
        bins = np.searchsorted(edges[1:-1], lengths, side="right")
    else:
        bins = np.zeros(len(remaining), dtype=np.int32)
    rng = np.random.default_rng(cfg["seed"])
    from collections import Counter
    bin_counts = Counter(int(b) for b in bins)
    total_remaining = len(remaining)
    random_idx: list[int] = []
    for b, count in bin_counts.items():
        share = max(1, round(random_size * count / total_remaining))
        pool_b = [remaining[k] for k in range(len(remaining)) if int(bins[k]) == b]
        take = min(share, len(pool_b))
        chosen = rng.choice(pool_b, size=take, replace=False).tolist()
        random_idx.extend(int(i) for i in chosen)
    # Trim or fill to exactly random_size
    if len(random_idx) > random_size:
        random_idx = random_idx[:random_size]
    elif len(random_idx) < random_size:
        leftover = [i for i in remaining if i not in set(random_idx)]
        need = min(random_size - len(random_idx), len(leftover))
        random_idx.extend(int(i) for i in rng.choice(leftover, size=need, replace=False))
    random_rows = [pool_rows[i] for i in sorted(random_idx)]
    print(f"  Random arm: {len(random_rows):,} length-stratified examples")

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
    arms_info: dict[str, dict] = {}
    for j, p in enumerate(probes):
        arm_name = f"quality_{p['name']}"
        arm_rows = [pool_rows[i] for i in quality_idx_per_probe[j]]
        arms_info[arm_name] = save_arm(arm_rows, arm_name)
    arms_info["random"] = save_arm(random_rows, "random")

    cont_manifest = {
        "base_model": base_model,
        "max_length": max_length,
        "extraction_layer": cfg["extraction_layer"],
        "probes": [{"name": p["name"], "path": str(Path(cfg["probe_dir"]) / spec["filename"])}
                   for p, spec in zip(probes, (cfg.get("probes") or [{"filename": cfg.get("probe_filename", "probe.pkl")}]))],
        "pool_size": n_pool,
        "quality_size": quality_size,
        "random_size": random_size,
        "category_filter": sorted(category_filter) if category_filter else None,
        "arms": arms_info,
    }
    cont_path = out_dir / "continuation_manifest.json"
    cont_path.write_text(json.dumps(cont_manifest, indent=2))
    print(f"\nManifest → {cont_path}")

    run_dir = "runs/smoltalk_v3_math"
    print("\nNext steps:")
    for arm_name in arms_info:
        if arm_name == "random":
            continue
        print(f"  uv run finetune.py --base-model HuggingFaceTB/SmolLM2-1.7B "
              f"--train-data {arms_info[arm_name]['path']} "
              f"--val-data {run_dir}/data/val "
              f"--output-dir {run_dir}/adapter_{arm_name}")
    print(f"  uv run finetune.py --base-model HuggingFaceTB/SmolLM2-1.7B "
          f"--train-data {arms_info['random']['path']} "
          f"--val-data {run_dir}/data/val "
          f"--output-dir {run_dir}/adapter_random")


if __name__ == "__main__":
    main()
