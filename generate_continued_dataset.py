"""Score new SmolTalk examples with the residual stream probe and produce quality + random baselines.

Streams the next unselected block of SmolTalk (past the rows consumed by build_sft_data.py),
runs each example through the base model, extracts the layer-17 residual stream, and scores
it with the probe trained in probe.py. Produces one quality arm plus two random baselines:

  continuation/quality_<probe>/                  — top-N by probe score
  continuation/random_token_match(_<probe>)/     — random rows with matched token budget
  continuation/random_token_cat_match(_<probe>)/ — random rows with matched token budget and category mix

Both are ready for finetune.py (input_ids + labels + length columns).

Usage (run AFTER probe.py):

    uv run generate_continued_dataset.py
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt
from pipeline_common import (
    ATTN_IMPLEMENTATION,
    ensure_hf_home_env,
    infer_instruct_tokenizer_model,
    last_response_token_positions,
    pad_tokenized_batch,
    pool_hidden_at_positions,
    resolve_device_dtype,
)

ensure_hf_home_env()

CONFIG = {
    "manifest_path": "runs/smoltalk_v4/manifest.json",
    "probe_dir": "runs/smoltalk_v4/probe",
    # One quality arm family is built per probe.
    "probes": [
        {"name": "math_da", "filename": "probe_math_da.pkl"},
    ],
    "output_dir": "runs/smoltalk_v4/continuation",
    "extraction_layer": 17,    # must match probe.py
    # Only stream examples from these categories (set to None for all categories).
    "category_filter": {"math", "data-analysis"},
    "pool_size": 30_000,       # new SmolTalk rows to score with probe
    "quality_size": 10_000,    # top-N by each probe score → reference quality arm per probe
    "quality_scales": [1.0, 0.8, 0.5],  # emit additional quality arms as fractions of quality_size
    "random_size": 10_000,     # rows in each random baseline arm
    "batch_size": 64,
    "seed": 42,
    "device": "auto",
}


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
    ids_t, lbl_t = pad_tokenized_batch(
        [r["input_ids"] for r in batch_rows],
        [r["labels"] for r in batch_rows],
        input_pad_token_id=0,
        label_pad_token_id=-100,
        device=device,
    )

    with torch.inference_mode():
        model(input_ids=ids_t)
    hidden = captured["acts"]  # (batch, seq_len, d_model)
    last_resp_idx = last_response_token_positions(lbl_t, label_pad_token_id=-100)
    pooled = pool_hidden_at_positions(hidden, last_resp_idx).float().cpu().numpy()  # (batch, d_model)

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


def _subset_token_sum(indices: np.ndarray, lengths: np.ndarray) -> int:
    return int(lengths[indices].sum()) if len(indices) else 0


def _max_token_sum(candidate_idx: np.ndarray, n_select: int, lengths: np.ndarray) -> int:
    if n_select <= 0:
        return 0
    if len(candidate_idx) < n_select:
        raise ValueError(f"Need {n_select} candidates, only have {len(candidate_idx)}.")
    vals = lengths[candidate_idx]
    # Sum of the n_select largest lengths.
    largest = np.partition(vals, len(vals) - n_select)[-n_select:]
    return int(largest.sum())


def _scaled_counts(base_counts: Counter[str], target_total: int) -> dict[str, int]:
    """Scale category counts to target_total while preserving proportions."""
    if target_total <= 0:
        return {}
    total = sum(base_counts.values())
    if total <= 0:
        return {}
    raw = {k: (v * target_total) / total for k, v in base_counts.items()}
    out = {k: int(np.floor(x)) for k, x in raw.items()}
    used = sum(out.values())
    remainder = target_total - used
    if remainder > 0:
        frac_sorted = sorted(raw.keys(), key=lambda k: (raw[k] - out[k]), reverse=True)
        for k in frac_sorted[:remainder]:
            out[k] += 1
    return out


def _normalize_quality_scales(raw_scales: Any) -> list[float]:
    """Validate and normalize quality scales into unique values in (0, 1]."""
    scales = [1.0] if raw_scales is None else list(raw_scales)
    if not scales:
        scales = [1.0]
    normalized: list[float] = []
    seen: set[float] = set()
    for s in scales:
        v = round(float(s), 6)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Invalid quality scale {s}; expected 0 < scale <= 1.")
        if v not in seen:
            normalized.append(v)
            seen.add(v)
    if 1.0 not in seen:
        normalized.insert(0, 1.0)
    return normalized


def _quality_arm_name(probe_name: str, scale: float) -> str:
    pct = int(round(scale * 100))
    return f"quality_{probe_name}" if pct == 100 else f"quality_{probe_name}_{pct}pct"


def _sample_random_with_token_target(
    candidate_idx: list[int],
    lengths: np.ndarray,
    target_tokens: int,
    rng: np.random.Generator,
    *,
    n_select: int,
    categories: list[str] | None = None,
    target_group_counts: dict[str, int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    candidates = np.array(candidate_idx, dtype=np.int32)
    if len(candidates) < n_select:
        raise ValueError(f"Need {n_select} random rows, but only {len(candidates)} remain.")

    if target_group_counts is None:
        group_counts = {"__all__": int(n_select)}
    else:
        group_counts = {str(k): int(v) for k, v in target_group_counts.items() if int(v) > 0}
        if sum(group_counts.values()) != n_select:
            raise ValueError("target_group_counts must sum to n_select.")
        if categories is None:
            raise ValueError("categories are required when target_group_counts is set.")

    group_pools: dict[str, np.ndarray] = {}
    group_selected: dict[str, np.ndarray] = {}
    group_unselected: dict[str, np.ndarray] = {}
    for group, need in group_counts.items():
        pool = (
            candidates
            if group == "__all__"
            else np.array([i for i in candidates if categories[int(i)] == group], dtype=np.int32)
        )
        if len(pool) < need:
            raise ValueError(
                f"Cannot sample group '{group}': need {need}, only {len(pool)} candidates available."
            )
        pick_pos = rng.choice(len(pool), size=need, replace=False)
        mask = np.ones(len(pool), dtype=bool)
        mask[pick_pos] = False
        group_pools[group] = pool
        group_selected[group] = pool[pick_pos].astype(np.int32)
        group_unselected[group] = pool[mask].astype(np.int32)

    selected = np.concatenate(list(group_selected.values())).astype(np.int32)
    tokens_before = _subset_token_sum(selected, lengths)
    swaps_applied = 0

    if tokens_before < target_tokens:
        needed = int(target_tokens - tokens_before)
        moves: list[tuple[int, int, int]] = []
        for group, sel in group_selected.items():
            uns = group_unselected[group]
            if len(sel) == 0 or len(uns) == 0:
                continue
            sel_sorted = sel[np.argsort(lengths[sel])]
            uns_sorted = uns[np.argsort(lengths[uns])[::-1]]
            for s, u in zip(sel_sorted.tolist(), uns_sorted.tolist()):
                gain = int(lengths[int(u)] - lengths[int(s)])
                if gain > 0:
                    moves.append((gain, int(s), int(u)))

        # Keep perturbation minimal: prefer small gains first.
        moves.sort(key=lambda x: x[0])
        selected_set = set(int(i) for i in selected.tolist())
        gained = 0
        for gain, s, u in moves:
            if s not in selected_set or u in selected_set:
                continue
            selected_set.remove(s)
            selected_set.add(u)
            swaps_applied += 1
            gained += int(gain)
            if gained >= needed:
                break
        selected = np.array(sorted(selected_set), dtype=np.int32)

    selected_tokens = _subset_token_sum(selected, lengths)
    max_possible_tokens = int(
        sum(_max_token_sum(group_pools[g], group_counts[g], lengths) for g in group_counts)
    )

    info: dict[str, Any] = {
        "target_tokens": int(target_tokens),
        "selected_tokens": int(selected_tokens),
        "token_delta": int(selected_tokens - target_tokens),
        "met_target_tokens": bool(selected_tokens >= target_tokens),
        "max_possible_tokens": int(max_possible_tokens),
        "selection_method": "random_plus_simple_swaps",
        "initial_selected_tokens": int(tokens_before),
        "swaps_applied": int(swaps_applied),
    }
    if target_group_counts is not None and categories is not None:
        selected_counts = Counter(categories[int(i)] for i in selected.tolist())
        info["target_category_counts"] = {k: int(v) for k, v in sorted(group_counts.items())}
        info["selected_category_counts"] = {k: int(v) for k, v in sorted(selected_counts.items())}

    return sorted(int(i) for i in selected.tolist()), info


def main() -> None:
    cfg = CONFIG
    device, dtype = resolve_device_dtype(cfg["device"])
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
    probe_specs = list(cfg.get("probes") or [])
    if not probe_specs:
        raise ValueError("CONFIG['probes'] must contain at least one probe entry.")
    probes = []
    for spec in probe_specs:
        probe_path = Path(cfg["probe_dir"]) / spec["filename"]
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe not found: {probe_path}\nRun probe.py first.")
        with probe_path.open("rb") as f:
            probes.append({"name": spec["name"], "probe": pickle.load(f)})
    print(f"Loaded {len(probes)} probes: {[p['name'] for p in probes]}")

    # --- IT tokenizer ---
    it_model = infer_instruct_tokenizer_model(base_model)
    tokenizer = AutoTokenizer.from_pretrained(it_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model + hook ---
    model_path = base_model
    print(f"\nLoading {base_model} (path: {model_path}) ...")
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, attn_implementation=ATTN_IMPLEMENTATION
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

    # Batching can overrun the target by up to batch_size; clamp to exact pool_size.
    if len(pool_rows) > cfg["pool_size"]:
        pool_rows = pool_rows[: cfg["pool_size"]]
        pool_probe_scores = pool_probe_scores[: cfg["pool_size"]]

    n_pool = len(pool_rows)
    scores_matrix = np.array(pool_probe_scores, dtype=np.float32)  # (n_pool, n_probes)
    print(f"\nCollected and scored {n_pool:,} examples with {len(probes)} probe(s)")

    quality_size = int(cfg["quality_size"])
    random_size = int(cfg["random_size"])
    quality_scales = _normalize_quality_scales(cfg.get("quality_scales"))
    all_lengths = np.array([int(r["length"]) for r in pool_rows], dtype=np.int64)
    all_categories = [str(r.get("category", "")) for r in pool_rows]

    print(
        "Quality scales: "
        + ", ".join(f"{s:.2f}" for s in quality_scales)
        + f"  (random baselines fixed at {random_size:,} rows)"
    )

    # Build quality arms per probe: reference top-quality_size plus scaled subsets.
    quality_idx_by_arm: dict[str, list[int]] = {}
    quality_arm_meta: dict[str, dict[str, Any]] = {}
    quality_arm_order: list[str] = []
    base_quality_idx_per_probe: list[list[int]] = []
    for j, p in enumerate(probes):
        col = scores_matrix[:, j]
        ranked_idx = np.argsort(col)[::-1]
        base_idx = ranked_idx[:quality_size].astype(np.int32)
        if len(base_idx) == 0:
            raise ValueError("quality_size produced an empty quality arm.")
        base_quality_idx_per_probe.append([int(i) for i in base_idx.tolist()])

        threshold = float(col[int(base_idx[-1])])
        print(
            f"  Quality reference [{p['name']}]: top-{quality_size}  "
            f"threshold={threshold:.4f}  range [{col[base_idx].min():.4f}, {col[base_idx].max():.4f}]"
        )

        for scale in quality_scales:
            scaled_n = max(1, min(quality_size, int(round(quality_size * scale))))
            scaled_idx = base_idx[:scaled_n]
            arm_name = _quality_arm_name(p["name"], scale)
            quality_idx_by_arm[arm_name] = sorted(int(i) for i in scaled_idx.tolist())
            quality_arm_order.append(arm_name)
            scale_threshold = float(col[int(scaled_idx[-1])])
            quality_arm_meta[arm_name] = {
                "probe": p["name"],
                "scale": float(scale),
                "rows": int(scaled_n),
                "threshold": float(scale_threshold),
                "reference_quality_size": int(quality_size),
            }
            if abs(scale - 1.0) > 1e-8:
                print(
                    f"    {arm_name}: {scaled_n:,} rows ({int(round(scale * 100))}%) "
                    f"threshold={scale_threshold:.4f}"
                )

    # Build two fixed-size random baselines per probe, matched to the 100% quality arm:
    # 1) token-matched (random_size rows)
    # 2) token + category matched (random_size rows)
    random_idx_by_arm: dict[str, list[int]] = {}
    random_match_meta: dict[str, dict[str, Any]] = {}
    random_arm_order: list[str] = []
    for j, p in enumerate(probes):
        quality_name = _quality_arm_name(p["name"], 1.0)
        quality_idx = base_quality_idx_per_probe[j]
        quality_set = set(quality_idx)
        candidates = [i for i in range(n_pool) if i not in quality_set]
        target_tokens = int(all_lengths[np.array(quality_idx, dtype=np.int32)].sum())

        # Use separate deterministic RNG streams per arm to avoid accidental coupling.
        base_seed = int(cfg["seed"]) + (j * 1000)
        token_rng = np.random.default_rng(base_seed + 1)
        cat_rng = np.random.default_rng(base_seed + 2)

        token_idx, token_meta = _sample_random_with_token_target(
            candidate_idx=candidates,
            lengths=all_lengths,
            target_tokens=target_tokens,
            rng=token_rng,
            n_select=random_size,
        )
        target_counts = _scaled_counts(Counter(all_categories[i] for i in quality_idx), random_size)
        cat_idx, cat_meta = _sample_random_with_token_target(
            candidate_idx=candidates,
            lengths=all_lengths,
            target_tokens=target_tokens,
            rng=cat_rng,
            n_select=random_size,
            categories=all_categories,
            target_group_counts=target_counts,
        )

        suffix = "" if len(probes) == 1 else f"_{p['name']}"
        token_arm = f"random_token_match{suffix}"
        cat_arm = f"random_token_cat_match{suffix}"

        token_meta = dict(token_meta)
        token_meta["matched_to_quality_arm"] = quality_name
        cat_meta = dict(cat_meta)
        cat_meta["matched_to_quality_arm"] = quality_name

        random_idx_by_arm[token_arm] = token_idx
        random_idx_by_arm[cat_arm] = cat_idx
        random_match_meta[token_arm] = token_meta
        random_match_meta[cat_arm] = cat_meta
        random_arm_order.extend([token_arm, cat_arm])

        print(
            f"  Random arm [{token_arm}]: {len(token_idx):,} rows, "
            f"tokens={token_meta['selected_tokens']:,} target={target_tokens:,} "
            f"delta={int(token_meta['token_delta']):+,}"
        )
        print(
            f"  Random arm [{cat_arm}]: {len(cat_idx):,} rows, "
            f"tokens={cat_meta['selected_tokens']:,} target={target_tokens:,} "
            f"delta={int(cat_meta['token_delta']):+,}"
        )
        if not token_meta["met_target_tokens"]:
            print(
                f"    WARNING: {token_arm} could not reach target tokens "
                f"(max={token_meta['max_possible_tokens']:,})."
            )
        if not cat_meta["met_target_tokens"]:
            print(
                f"    WARNING: {cat_arm} could not reach target tokens under category constraints "
                f"(max={cat_meta['max_possible_tokens']:,})."
            )

    # --- Save ---
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_arm(rows: list[dict], name: str) -> dict:
        ds = Dataset.from_list(rows)
        path = out_dir / name
        ds.save_to_disk(str(path))
        n_tok = sum(r["length"] for r in rows)
        cat_counts = Counter(str(r.get("category", "")) for r in rows)
        print(f"  {name}: {len(ds):,} rows, {n_tok / 1e6:.1f}M tokens → {path}")
        return {
            "path": str(path),
            "rows": len(ds),
            "total_tokens": n_tok,
            "category_counts": {k: int(v) for k, v in sorted(cat_counts.items())},
        }

    print("\nSaving ...")
    arms_info: dict[str, dict] = {}
    for arm_name in quality_arm_order:
        arm_rows = [pool_rows[i] for i in quality_idx_by_arm[arm_name]]
        arm_info = save_arm(arm_rows, arm_name)
        arm_info["selection"] = quality_arm_meta[arm_name]
        arms_info[arm_name] = arm_info
    for arm_name in random_arm_order:
        idxs = random_idx_by_arm[arm_name]
        arm_rows = [pool_rows[i] for i in idxs]
        arm_info = save_arm(arm_rows, arm_name)
        arm_info["match"] = random_match_meta[arm_name]
        arms_info[arm_name] = arm_info

    cont_manifest = {
        "base_model": base_model,
        "max_length": max_length,
        "extraction_layer": cfg["extraction_layer"],
        "probes": [
            {
                "name": spec["name"],
                "path": str(Path(cfg["probe_dir"]) / spec["filename"]),
            }
            for spec in probe_specs
        ],
        "pool_size": n_pool,
        "quality_size": quality_size,
        "quality_scales": quality_scales,
        "random_size": random_size,
        "category_filter": sorted(category_filter) if category_filter else None,
        "arms": arms_info,
    }
    cont_path = out_dir / "continuation_manifest.json"
    cont_path.write_text(json.dumps(cont_manifest, indent=2))
    print(f"\nManifest → {cont_path}")

    run_dir = str(Path(cfg["manifest_path"]).parent)
    val_data = manifest["splits"]["val"]["path"]
    print("\nNext steps:")
    for arm_name in list(quality_arm_order) + list(random_arm_order):
        print(f"  uv run finetune.py --base-model {base_model} "
              f"--train-data {arms_info[arm_name]['path']} "
              f"--val-data {val_data} "
              f"--output-dir {run_dir}/adapter_{arm_name}")


if __name__ == "__main__":
    main()
