"""Profile peak VRAM for LoRA SFT across (model, seq_len, batch_size, attn_impl).

Runs each configuration in an isolated subprocess to avoid CUDA memory
fragmentation between trials. Reports a table of peak VRAM and OK/OOM status.

Usage:
    uv run vram_profile.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

# ---------------------------------------------------------------------------
# Configurations to sweep
# ---------------------------------------------------------------------------

MODELS = [
    "google/gemma-3-270m",
    "google/gemma-3-1b",
]
SEQ_LENS = [1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 2, 4, 8, 16]
ATTN_IMPLS = ["sdpa", "eager"]

LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Worker (runs inside subprocess)
# ---------------------------------------------------------------------------

WORKER_CODE = """
import json, sys, os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

def run(model_name, seq_len, batch_size, attn_impl):
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attn_impl,
    )
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r={lora_r},
        lora_alpha={lora_alpha},
        lora_dropout=0.05,
        target_modules={target_modules!r},
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={{"use_reentrant": False}})
    model.cuda()

    torch.cuda.reset_peak_memory_stats()

    vocab_size = base.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps({{"peak_gb": peak_gb}}))

cfg = json.loads(sys.argv[1])
run(cfg["model_name"], cfg["seq_len"], cfg["batch_size"], cfg["attn_impl"])
"""

_WORKER_TEMPLATE = WORKER_CODE.format(
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
)


def probe(model_name: str, seq_len: int, batch_size: int, attn_impl: str) -> float | None:
    """Return peak VRAM in GB, or None on OOM/error."""
    cfg = json.dumps(
        {"model_name": model_name, "seq_len": seq_len, "batch_size": batch_size, "attn_impl": attn_impl}
    )
    result = subprocess.run(
        [sys.executable, "-c", _WORKER_TEMPLATE, cfg],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    if result.returncode != 0:
        return None  # OOM or other error
    try:
        last_line = result.stdout.strip().split("\n")[-1]
        return json.loads(last_line)["peak_gb"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Total VRAM: {total_vram:.1f} GB\n")

    # Results keyed by (model_short, seq_len, batch_size, attn_impl)
    results: dict[tuple, float | None] = {}

    configs = [
        (m, s, b, a)
        for m in MODELS
        for s in SEQ_LENS
        for b in BATCH_SIZES
        for a in ATTN_IMPLS
    ]
    total = len(configs)

    for i, (model, seq_len, batch_size, attn_impl) in enumerate(configs, 1):
        short = model.split("/")[-1]
        label = f"{short} | seq={seq_len:4d} | bs={batch_size:2d} | {attn_impl}"
        print(f"[{i:3d}/{total}] {label} ...", end="", flush=True)
        peak = probe(model, seq_len, batch_size, attn_impl)
        results[(model, seq_len, batch_size, attn_impl)] = peak
        if peak is None:
            print("  OOM")
        else:
            flag = "OK" if peak < total_vram * 0.90 else "TIGHT"
            print(f"  {peak:.2f} GB  [{flag}]")

    # Summary table: best (highest throughput) OK config per model
    print("\n--- Summary: largest OK batch per (model, seq_len, attn_impl) ---")
    header = f"{'Model':<20} {'seq_len':>8} {'attn':>6}  {'max_bs':>6}  {'peak_gb':>8}"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        short = model.split("/")[-1]
        for seq_len in SEQ_LENS:
            for attn_impl in ATTN_IMPLS:
                best_bs = None
                best_peak = None
                for batch_size in BATCH_SIZES:
                    peak = results.get((model, seq_len, batch_size, attn_impl))
                    if peak is not None and peak < total_vram * 0.90:
                        best_bs = batch_size
                        best_peak = peak
                if best_bs is not None:
                    print(f"{short:<20} {seq_len:>8} {attn_impl:>6}  {best_bs:>6}  {best_peak:>8.2f}")
                else:
                    print(f"{short:<20} {seq_len:>8} {attn_impl:>6}  {'OOM':>6}")


if __name__ == "__main__":
    main()
