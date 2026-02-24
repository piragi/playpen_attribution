"""Score new SmolTalk examples with the residual stream probe and produce ablation arms.

Streams the next unselected block of SmolTalk (past the rows consumed by build_sft_data.py),
runs each example through the base model, extracts the layer-17 residual stream, and scores
it with the probe trained in probe.py. Produces:

  continuation/quality_<probe>/                  — top-N by probe score
  continuation/random_token_match(_<probe>)/     — random rows with matched token budget
  continuation/random_token_cat_match(_<probe>)/ — random rows with matched token budget and category mix
  continuation/label_*                           — optional LLM-judge label-only baseline(s)

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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_sft_data import _get_magpie_score, _mask_prompt
from pipeline_common import (
    ATTN_IMPLEMENTATION,
    ensure_hf_home_env,
    get_transformer_layers_for_hook,
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
    # Optional label-only baseline arms (LLM-judge quality field from SmolTalk rows).
    # Each arm ranks rows whose `quality` is in `qualities` and keeps the best ones.
    # `size=None` defaults to quality_size; token budget is matched to `token_match_probe`
    # (or first probe when token_match_probe=None).
    "label_arms": [
        {
            "name": "label_good_excellent",
            "qualities": {"good", "excellent"},
            "size": None,
            "token_match_probe": None,
        },
    ],
    "batch_size": 64,
    "seed": 42,
    "device": "auto",
    # Embedding source for residual extraction:
    # - "probe_meta": auto-use source recorded in probe_meta_<name>.json (recommended)
    # - "base": always use manifest base_model
    # - "adapter": use manifest base_model + embedding_adapter_path
    "embedding_source_mode": "probe_meta",
    "embedding_adapter_path": None,
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


def resolve_embedding_source(cfg: dict, base_model: str, probe_specs: list[dict[str, str]]) -> dict[str, Any]:
    """Resolve embedding model source for continuation scoring."""
    mode = str(cfg.get("embedding_source_mode", "probe_meta")).strip().lower()
    if mode not in {"probe_meta", "base", "adapter"}:
        raise ValueError(
            f"Invalid embedding_source_mode '{mode}'. "
            "Expected one of: probe_meta, base, adapter."
        )

    if mode == "base":
        return {"mode": "base", "base_model": base_model, "adapter_path": None}

    if mode == "adapter":
        adapter_path = cfg.get("embedding_adapter_path")
        if not adapter_path:
            raise ValueError("embedding_source_mode='adapter' requires embedding_adapter_path.")
        return {
            "mode": "adapter",
            "base_model": base_model,
            "adapter_path": str(Path(adapter_path)),
        }

    # mode == probe_meta
    sources: list[tuple[str, str | None, str, int | None]] = []
    for spec in probe_specs:
        meta_path = Path(cfg["probe_dir"]) / f"probe_meta_{spec['name']}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        source = meta.get("embedding_source")
        if not source:
            continue
        src_mode = str(source.get("mode", "base")).strip().lower()
        src_adapter = source.get("adapter_path")
        src_layer = meta.get("extraction_layer")
        sources.append((src_mode, str(src_adapter) if src_adapter else None, str(meta_path), src_layer))

    if not sources:
        print("WARNING: probe meta has no embedding_source info; defaulting to base embeddings.")
        return {"mode": "base", "base_model": base_model, "adapter_path": None}

    first_mode, first_adapter, first_meta, first_layer = sources[0]
    for src_mode, src_adapter, meta_path, src_layer in sources[1:]:
        if src_mode != first_mode or src_adapter != first_adapter or src_layer != first_layer:
            raise ValueError(
                "Probe metas disagree on embedding source:\n"
                f"  first: mode={first_mode} adapter={first_adapter} layer={first_layer} ({first_meta})\n"
                f"  other: mode={src_mode} adapter={src_adapter} layer={src_layer} ({meta_path})"
            )
    return {
        "mode": first_mode,
        "base_model": base_model,
        "adapter_path": first_adapter,
        "probe_extraction_layer": first_layer,
        "from_probe_meta": True,
    }


def _sample_random_with_token_target(
    candidate_idx: list[int],
    lengths: np.ndarray,
    target_tokens: int,
    rng: np.random.Generator,
    *,
    n_select: int,
    categories: list[str] | None = None,
    target_group_counts: dict[str, int] | None = None,
    initial_selected_idx: list[int] | None = None,
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
        group_pools[group] = pool

    group_selected: dict[str, np.ndarray] = {}
    group_unselected: dict[str, np.ndarray] = {}
    if initial_selected_idx is None:
        for group, need in group_counts.items():
            pool = group_pools[group]
            pick_pos = rng.choice(len(pool), size=need, replace=False)
            mask = np.ones(len(pool), dtype=bool)
            mask[pick_pos] = False
            group_selected[group] = pool[pick_pos].astype(np.int32)
            group_unselected[group] = pool[mask].astype(np.int32)
    else:
        initial = np.array([int(i) for i in initial_selected_idx], dtype=np.int32)
        if len(initial) != n_select:
            raise ValueError(
                "initial_selected_idx must have exactly n_select rows: "
                f"got {len(initial)} vs expected {n_select}."
            )
        if len(set(initial.tolist())) != n_select:
            raise ValueError("initial_selected_idx must not contain duplicates.")
        candidate_set = set(int(i) for i in candidates.tolist())
        if any(int(i) not in candidate_set for i in initial.tolist()):
            raise ValueError("initial_selected_idx contains rows not present in candidate_idx.")

        selected_set = set(int(i) for i in initial.tolist())
        for group, need in group_counts.items():
            if group == "__all__":
                sel = np.array(sorted(selected_set), dtype=np.int32)
            else:
                sel = np.array(
                    sorted(i for i in selected_set if categories[int(i)] == group),
                    dtype=np.int32,
                )
            if len(sel) != need:
                raise ValueError(
                    f"initial_selected_idx has {len(sel)} rows for group '{group}', expected {need}."
                )
            pool = group_pools[group]
            unselected = np.array([int(i) for i in pool.tolist() if int(i) not in selected_set], dtype=np.int32)
            group_selected[group] = sel
            group_unselected[group] = unselected

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
    embedding_source = resolve_embedding_source(cfg, base_model, probe_specs)
    probe_layer = embedding_source.get("probe_extraction_layer")
    if embedding_source.get("from_probe_meta") and probe_layer is not None:
        cfg_layer = int(cfg["extraction_layer"])
        if int(probe_layer) != cfg_layer:
            raise ValueError(
                "Probe meta extraction layer mismatch: "
                f"probe_meta={probe_layer} vs generate_continued_dataset.py config={cfg_layer}. "
                "Set CONFIG['extraction_layer'] to match probe.py or regenerate the probe."
            )
    print(
        "Embedding source for continuation scoring: "
        f"{embedding_source['mode']}"
        + (f" ({embedding_source['adapter_path']})" if embedding_source.get("adapter_path") else "")
    )

    # --- IT tokenizer ---
    it_model = infer_instruct_tokenizer_model(base_model)
    tokenizer = AutoTokenizer.from_pretrained(it_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Embedding model + hook ---
    print(f"\nLoading base model for embedding extraction: {base_model} ...")
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, attn_implementation=ATTN_IMPLEMENTATION
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
    base_quality_idx_by_probe: dict[str, list[int]] = {}
    quality_ref_tokens_by_probe: dict[str, int] = {}
    for j, p in enumerate(probes):
        col = scores_matrix[:, j]
        ranked_idx = np.argsort(col)[::-1]
        base_idx = ranked_idx[:quality_size].astype(np.int32)
        if len(base_idx) == 0:
            raise ValueError("quality_size produced an empty quality arm.")
        base_quality_idx_by_probe[p["name"]] = [int(i) for i in base_idx.tolist()]
        quality_ref_tokens_by_probe[p["name"]] = int(all_lengths[base_idx].sum())

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
        quality_idx = base_quality_idx_by_probe[p["name"]]
        quality_set = set(quality_idx)
        candidates = [i for i in range(n_pool) if i not in quality_set]
        target_tokens = quality_ref_tokens_by_probe[p["name"]]

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

    # Optional label-only baselines from SmolTalk LLM-judge quality labels.
    label_idx_by_arm: dict[str, list[int]] = {}
    label_meta_by_arm: dict[str, dict[str, Any]] = {}
    label_arm_order: list[str] = []
    label_specs = list(cfg.get("label_arms") or [])
    if label_specs:
        default_probe_name = probes[0]["name"]
        row_quality = [str(r.get("quality", "")).strip().lower() for r in pool_rows]
        row_magpie = np.array([float(r.get("magpie_score", 0.0)) for r in pool_rows], dtype=np.float32)
        quality_rank = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}
        for arm_i, spec in enumerate(label_specs):
            arm_name = str(spec.get("name", "")).strip()
            if not arm_name:
                raise ValueError("Each entry in CONFIG['label_arms'] needs a non-empty 'name'.")
            if arm_name in quality_idx_by_arm or arm_name in random_idx_by_arm:
                raise ValueError(f"Label arm name collides with existing arm: {arm_name}")

            qualities = {str(q).strip().lower() for q in (spec.get("qualities") or []) if str(q).strip()}
            if not qualities:
                raise ValueError(f"Label arm '{arm_name}' has empty qualities; provide at least one label value.")

            requested_rows = quality_size if spec.get("size") is None else int(spec["size"])
            if requested_rows <= 0:
                raise ValueError(f"Label arm '{arm_name}' has non-positive size: {requested_rows}.")

            candidates = [i for i, q in enumerate(row_quality) if q in qualities]
            available_rows = len(candidates)
            if available_rows == 0:
                print(
                    f"    WARNING: label arm '{arm_name}' has zero candidates for "
                    f"qualities={sorted(qualities)}; skipping this arm."
                )
                continue
            selected_rows = min(requested_rows, available_rows)
            shortfall_rows = max(0, requested_rows - available_rows)

            # Deterministic "best available" ranking:
            # 1) quality label rank (excellent > good > fair > poor)
            # 2) magpie_score
            # 3) length
            ranked_candidates = sorted(
                candidates,
                key=lambda i: (
                    quality_rank.get(row_quality[i], -1),
                    float(row_magpie[i]),
                    int(all_lengths[i]),
                    -int(i),
                ),
                reverse=True,
            )
            base_selected = ranked_candidates[:selected_rows]

            match_probe = spec.get("token_match_probe")
            if match_probe is None:
                match_probe = default_probe_name
            if match_probe:
                match_probe = str(match_probe)
                if match_probe not in quality_ref_tokens_by_probe:
                    known = ", ".join(sorted(quality_ref_tokens_by_probe))
                    raise ValueError(
                        f"Label arm '{arm_name}' references unknown token_match_probe '{match_probe}'. "
                        f"Known probes: {known}"
                    )
                target_tokens = int(quality_ref_tokens_by_probe[match_probe])
            else:
                target_tokens = None

            if target_tokens is None:
                label_idx = sorted(int(i) for i in base_selected)
                selected_tokens = _subset_token_sum(np.array(label_idx, dtype=np.int32), all_lengths)
                label_meta: dict[str, Any] = {
                    "selection_method": "label_top_ranked",
                    "qualities": sorted(qualities),
                    "requested_rows": int(requested_rows),
                    "available_rows": int(available_rows),
                    "rows": int(selected_rows),
                    "shortfall_rows": int(shortfall_rows),
                    "selected_tokens": int(selected_tokens),
                    "token_match_probe": None,
                }
            else:
                # Preserve the top-ranked label mix, then apply minimal swaps within each label bucket
                # to better match token budget.
                target_quality_counts = Counter(row_quality[i] for i in base_selected)
                label_idx, label_meta = _sample_random_with_token_target(
                    candidate_idx=candidates,
                    lengths=all_lengths,
                    target_tokens=target_tokens,
                    rng=np.random.default_rng(int(cfg["seed"]) + 10_000 + arm_i),
                    n_select=selected_rows,
                    categories=row_quality,
                    target_group_counts={k: int(v) for k, v in target_quality_counts.items()},
                    initial_selected_idx=[int(i) for i in base_selected],
                )
                label_meta = dict(label_meta)
                label_meta["selection_method"] = "label_top_ranked_plus_simple_swaps"
                label_meta["qualities"] = sorted(qualities)
                label_meta["requested_rows"] = int(requested_rows)
                label_meta["available_rows"] = int(available_rows)
                label_meta["rows"] = int(selected_rows)
                label_meta["shortfall_rows"] = int(shortfall_rows)
                label_meta["token_match_probe"] = match_probe

            label_idx_by_arm[arm_name] = label_idx
            label_meta_by_arm[arm_name] = label_meta
            label_arm_order.append(arm_name)
            print(
                f"  Label arm [{arm_name}]: {len(label_idx):,} rows, "
                f"qualities={sorted(qualities)}, requested={requested_rows:,}, "
                f"available={available_rows:,}, selected={selected_rows:,}, "
                f"tokens={label_meta['selected_tokens']:,}"
                + (
                    f", target={label_meta['target_tokens']:,}, delta={int(label_meta['token_delta']):+,}"
                    if "target_tokens" in label_meta
                    else ""
                )
            )
            if shortfall_rows > 0:
                print(
                    f"    WARNING: label arm '{arm_name}' shortfall: "
                    f"requested={requested_rows:,}, available={available_rows:,}, selected={selected_rows:,}."
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
    for arm_name in label_arm_order:
        idxs = label_idx_by_arm[arm_name]
        arm_rows = [pool_rows[i] for i in idxs]
        arm_info = save_arm(arm_rows, arm_name)
        arm_info["selection"] = label_meta_by_arm[arm_name]
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
        "label_arms": [
            {
                "name": str(spec.get("name", "")),
                "qualities": sorted(str(q).strip().lower() for q in (spec.get("qualities") or []) if str(q).strip()),
                "size": (quality_size if spec.get("size") is None else int(spec["size"])),
                "token_match_probe": spec.get("token_match_probe"),
            }
            for spec in label_specs
        ],
        "embedding_source": embedding_source,
        "category_filter": sorted(category_filter) if category_filter else None,
        "arms": arms_info,
    }
    cont_path = out_dir / "continuation_manifest.json"
    cont_path.write_text(json.dumps(cont_manifest, indent=2))
    print(f"\nManifest → {cont_path}")

    run_dir = str(Path(cfg["manifest_path"]).parent)
    val_data = manifest["splits"]["val"]["path"]
    print("\nNext steps:")
    for arm_name in list(quality_arm_order) + list(random_arm_order) + list(label_arm_order):
        print(f"  uv run finetune.py --base-model {base_model} "
              f"--train-data {arms_info[arm_name]['path']} "
              f"--val-data {val_data} "
              f"--output-dir {run_dir}/adapter_{arm_name}")


if __name__ == "__main__":
    main()
