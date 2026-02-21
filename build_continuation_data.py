from __future__ import annotations

"""Build continuation pretraining datasets for the SAE fingerprint experiment.

Two arms saved to output_dir/:
  random/   — next n_target_chunks FineWeb chunks after Phase 1 (stream order)
  filtered/ — top n_target_chunks by SAE fingerprint score, drawn from
               pool_factor × n_target_chunks candidates

Both are HF datasets with a single 'input_ids' column, consumed by:
    uv run pretrain.py --base-model ... --dataset-path <path> --output-dir ...

Both arms skip the first skip_chunks chunks (= Phase 1 training data) and draw
from the subsequent stream, so the two arms never overlap with each other or
with the Phase 1 data.
"""

import heapq
import json
import shutil
from pathlib import Path
from typing import Iterator

import torch
from datasets import Dataset, load_dataset
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_bidir_classifier import SAEFingerprint

CONFIG = {
    # Models
    "base_model": "google/gemma-3-270m",
    "classifier_path": "runs/pretrain_attribution_v2/sae_classifier/K256/best_model.pt",
    "sae_release": "gemma-scope-2-270m-pt-res",
    "sae_id": "layer_12_width_16k_l0_small",
    "layer_idx": 12,
    "K": 256,
    # FineWeb stream
    "fineweb_dataset": "HuggingFaceFW/fineweb",
    "fineweb_config": "CC-MAIN-2024-10",
    "skip_chunks": 0,           # set dynamically from pretrain_metrics.json in main()
    "chunk_size": 1024,
    # Data budget
    "n_target_chunks": 0,       # set dynamically from pretrain_metrics.json in main()
    "pool_factor": 3,           # score 3× chunks before selecting top-N (filtered only)
    "score_batch_size": 8,
    # Output
    "output_dir": "runs/pretrain_continuation_v2",
    # Classifier architecture — must match trained model in classifier_path
    "d_sae": 16384,
    "d_embed": 64,
    "d_hidden": 128,
}


# ---------------------------------------------------------------------------
# Shared: FineWeb chunk stream with skip
# ---------------------------------------------------------------------------

def _fineweb_chunk_stream(
    cfg: dict,
    tokenizer,
    max_chunks: int,
) -> Iterator[list[int]]:
    """Yield packed chunks from FineWeb, skipping the first skip_chunks chunks.

    The skip re-tokenizes Phase 1 data (100M tokens ≈ ~2 min with fast tokenizer)
    to land at the exact same boundary as Phase 1 training.
    """
    raw = load_dataset(
        cfg["fineweb_dataset"],
        cfg["fineweb_config"],
        split="train",
        streaming=True,
    )
    eos_id = tokenizer.eos_token_id
    buffer: list[int] = []
    n_skipped = 0
    n_yielded = 0
    skip = cfg["skip_chunks"]
    chunk_size = cfg["chunk_size"]

    for row in raw:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            ids = ids + [eos_id]
        buffer.extend(ids)

        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]

            if n_skipped < skip:
                n_skipped += 1
                if n_skipped % 10_000 == 0:
                    print(f"\r  skipping Phase 1 data: {n_skipped:,}/{skip:,}", end="", flush=True)
                continue

            yield chunk
            n_yielded += 1
            if n_yielded >= max_chunks:
                return


# ---------------------------------------------------------------------------
# Arm 1: Random
# ---------------------------------------------------------------------------

def generate_random(cfg: dict, tokenizer) -> Dataset:
    """Take the next n_target_chunks chunks after Phase 1 in stream order."""
    n = cfg["n_target_chunks"]
    print(f"\nGenerating random arm: {n:,} chunks × {cfg['chunk_size']} tokens")

    chunks: list[list[int]] = []
    for chunk in _fineweb_chunk_stream(cfg, tokenizer, n):
        chunks.append(chunk)
        if len(chunks) % 10_000 == 0:
            print(f"\r  collected {len(chunks):,}/{n:,}", end="", flush=True)

    print(f"\n  Done. {len(chunks):,} chunks collected.")
    return Dataset.from_list([{"input_ids": c} for c in chunks])


# ---------------------------------------------------------------------------
# Arm 2: Filtered
# ---------------------------------------------------------------------------

def _resolve_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


def _score_batch(
    chunks: list[list[int]],
    model,
    sae_obj: SAE,
    classifier: SAEFingerprint,
    captured: dict,
    sae_device: str,
    classifier_device: str,
    k: int,
) -> list[float]:
    """Run model + SAE + classifier on a batch of chunks, return list of scores."""
    input_ids = torch.tensor(chunks, dtype=torch.long, device=next(model.parameters()).device)

    with torch.inference_mode():
        model(input_ids=input_ids)
        acts = captured["acts"].to(sae_device)      # (B, seq_len, d_model)
        sae_acts = sae_obj.encode(acts)             # (B, seq_len, d_sae)

        # Vectorized top-K extraction entirely on GPU — no per-sample Python loop
        mean_acts = sae_acts.clamp(min=0).mean(dim=1)           # (B, d_sae)
        topk_vals, topk_ids = torch.topk(mean_acts, k=k, dim=1) # (B, k)
        mask = topk_vals > 0                                     # (B, k)
        topk_vals = topk_vals * mask.float()                     # zero-out padding
        topk_ids = topk_ids * mask.long()

        logits = classifier(
            topk_ids.to(classifier_device),
            topk_vals.to(classifier_device),
            mask.to(classifier_device),
        )
        probs = torch.sigmoid(logits).cpu()

    return probs.tolist()


def generate_filtered(cfg: dict, tokenizer) -> tuple[Dataset, list[int]]:
    """Score pool_factor × n_target_chunks, select top n_target_chunks."""
    n_target = cfg["n_target_chunks"]
    n_pool = n_target * cfg["pool_factor"]
    k = cfg["K"]
    batch_size = cfg["score_batch_size"]
    device, dtype = _resolve_device()
    print(f"\nGenerating filtered arm: scoring {n_pool:,} chunks, selecting top {n_target:,}")
    print(f"  device={device}  dtype={dtype}  batch_size={batch_size}")

    # --- Load model ---
    print(f"Loading model from {cfg['base_model']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _input, output):
        captured["acts"] = output[0] if isinstance(output, tuple) else output

    hook = model.model.layers[cfg["layer_idx"]].register_forward_hook(hook_fn)  # type: ignore[union-attr,index]

    # --- Load SAE ---
    print(f"Loading SAE {cfg['sae_release']} / {cfg['sae_id']} ...")
    sae_obj: SAE = SAE.from_pretrained(  # type: ignore[assignment]
        release=cfg["sae_release"],
        sae_id=cfg["sae_id"],
        device=device,
    )
    sae_obj.eval()
    sae_device = str(next(sae_obj.parameters()).device)

    # --- Load classifier ---
    print(f"Loading classifier from {cfg['classifier_path']} ...")
    classifier = SAEFingerprint(
        d_sae=cfg["d_sae"],
        d_embed=cfg["d_embed"],
        d_hidden=cfg["d_hidden"],
        dropout=0.0,
    )
    classifier.load_state_dict(torch.load(cfg["classifier_path"], map_location=device))
    classifier = classifier.to(device)
    classifier.eval()

    # --- Streaming scoring with min-heap ---
    # heap entries: (score, counter, chunk_input_ids)
    # min-heap keeps the top-n_target highest-scoring chunks
    heap: list[tuple] = []
    counter = 0
    n_scored = 0
    batch_buf: list[list[int]] = []

    def _flush_batch(buf: list[list[int]]) -> None:
        nonlocal counter, n_scored
        scores = _score_batch(buf, model, sae_obj, classifier, captured, sae_device, device, k)
        for score, chunk in zip(scores, buf):
            if len(heap) < n_target:
                heapq.heappush(heap, (score, counter, chunk))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, chunk))
            counter += 1
        n_scored += len(buf)

    for chunk in _fineweb_chunk_stream(cfg, tokenizer, n_pool):
        batch_buf.append(chunk)
        if len(batch_buf) >= batch_size:
            _flush_batch(batch_buf)
            batch_buf = []
            if n_scored % (batch_size * 125) == 0:   # print every 1000 chunks
                heap_min = heap[0][0] if heap else 0.0
                print(
                    f"\r  scored {n_scored:,}/{n_pool:,}  heap_min={heap_min:.4f}",
                    end="", flush=True,
                )

    if batch_buf:
        _flush_batch(batch_buf)

    hook.remove()
    print(f"\n  Scored {n_scored:,} chunks, selected {len(heap):,}")
    if heap:
        all_scores = [e[0] for e in heap]
        print(f"  Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]  "
              f"mean={sum(all_scores)/len(all_scores):.4f}")

    # Sort by original stream position (counter) to preserve FineWeb ordering
    heap.sort(key=lambda entry: entry[1])
    positions = [entry[1] for entry in heap]
    rows = [{"input_ids": entry[2]} for entry in heap]
    return Dataset.from_list(rows), positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG.copy()
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve skip_chunks and n_target_chunks from Phase 1 pretrain metrics
    if cfg["skip_chunks"] == 0 or cfg["n_target_chunks"] == 0:
        metrics_path = Path(cfg.get("pretrain_metrics_path", "runs/pretrain_270m_v2/pretrain_metrics.json"))
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            n_chunks = metrics["num_chunks"]
            cfg["skip_chunks"] = n_chunks
            cfg["n_target_chunks"] = n_chunks
            print(f"Read num_chunks={n_chunks:,} from {metrics_path}")
        else:
            raise FileNotFoundError(
                f"Cannot resolve skip_chunks: {metrics_path} not found. "
                f"Run pretrain.py first, or set skip_chunks/n_target_chunks manually in CONFIG."
            )

    print(f"Loading tokenizer from {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Random arm ---
    random_path = out_root / "random"
    if random_path.exists():
        print(f"\nRandom dataset already exists at {random_path} — skipping.")
    else:
        random_ds = generate_random(cfg, tokenizer)
        tmp = Path(str(random_path) + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        random_ds.save_to_disk(str(tmp))
        tmp.rename(random_path)
        print(f"Saved random dataset → {random_path}  ({len(random_ds):,} chunks)")

    # --- Filtered arm ---
    filtered_path = out_root / "filtered"
    if filtered_path.exists():
        print(f"\nFiltered dataset already exists at {filtered_path} — skipping.")
    else:
        filtered_ds, filtered_pos_list = generate_filtered(cfg, tokenizer)
        tmp = Path(str(filtered_path) + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        filtered_ds.save_to_disk(str(tmp))
        tmp.rename(filtered_path)
        print(f"Saved filtered dataset → {filtered_path}  ({len(filtered_ds):,} chunks)")
        # Save positions for future overlap re-computation
        filtered_positions_path = out_root / "filtered_positions.json"
        filtered_positions_path.write_text(json.dumps(filtered_pos_list))
        filtered_pos_list = None  # allow GC

    # --- Overlap analysis ---
    n_target = cfg["n_target_chunks"]
    random_positions = set(range(n_target))  # random arm = stream positions [0, n_target)

    filtered_positions_path = out_root / "filtered_positions.json"
    if filtered_positions_path.exists():
        filtered_positions = set(json.loads(filtered_positions_path.read_text()))
    else:
        filtered_positions = set()
        print("\nWarning: cannot compute overlap — filtered positions not available.")

    if filtered_positions:
        overlap = random_positions & filtered_positions
        n_random = len(random_positions)
        n_filtered = len(filtered_positions)
        n_overlap = len(overlap)
        print(f"\n--- Overlap analysis ---")
        print(f"  Random arm:   {n_random:,} chunks (stream positions 0..{n_target - 1})")
        print(f"  Filtered arm: {n_filtered:,} chunks (selected from pool of {n_target * cfg['pool_factor']:,})")
        print(f"  Overlap:      {n_overlap:,} chunks ({n_overlap / n_random * 100:.1f}% of random arm)")
        print(f"  Filtered-only (from beyond random range): {n_filtered - n_overlap:,} chunks")

    # --- Manifest ---
    manifest: dict = {
        "config": cfg,
        "random_path": str(random_path),
        "filtered_path": str(filtered_path),
    }
    if filtered_positions:
        overlap = random_positions & filtered_positions
        manifest["overlap_count"] = len(overlap)
        manifest["overlap_fraction"] = len(overlap) / len(random_positions)
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest → {manifest_path}")

    print("\n--- Next: continued pretraining ---")
    base = cfg["base_model"]
    out = cfg["output_dir"]
    print(f"# Filtered arm:")
    print(f"uv run pretrain.py --base-model {base} --dataset-path {filtered_path} --output-dir {out}/filtered_model")
    print(f"# Random arm:")
    print(f"uv run pretrain.py --base-model {base} --dataset-path {random_path} --output-dir {out}/random_model")


if __name__ == "__main__":
    main()
