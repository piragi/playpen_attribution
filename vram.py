"""VRAM profiler for SmolLM2-1.7B under training and inference configs.

Measures peak VRAM for:
  1. Base model inference (seq_len=2048, sdpa, bfloat16)
  2. LoRA SFT training (seq_len=2048, bs=1/2/4/8, grad_ckpt on/off)
  3. Probe extraction forward pass (seq_len=2048, bs=16/32/64/128)

Usage:
    uv run vram.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")


def resolve_model_path(model_id: str) -> str:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = "models--" + model_id.replace("/", "--")
    snapshots_dir = hf_home / "hub" / slug / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[-1])
    return model_id


def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def reset() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak() -> int:
    return torch.cuda.max_memory_allocated()


def report(label: str, before: int) -> None:
    used = peak() - before
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"  {label:<45} {gb(used):>9}  (total avail: {gb(total)})")


BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B"
SEQ_LEN = 2048
DEVICE = "cuda"


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  |  Total VRAM: {gb(props.total_memory)}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_path = resolve_model_path(BASE_MODEL)
    print(f"Model path: {model_path}\n")

    # -----------------------------------------------------------------------
    # 1. Base model inference — seq_len=2048, sdpa, bfloat16
    # -----------------------------------------------------------------------
    print("=== 1. Base model inference (sdpa, bfloat16, seq_len=2048) ===")
    reset()
    before = peak()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    report("model load", before)

    dummy = torch.randint(0, 1000, (1, SEQ_LEN), device=DEVICE)
    before = peak()
    with torch.inference_mode():
        model(input_ids=dummy)
    report("forward pass (bs=1, seq=2048)", before)

    # -----------------------------------------------------------------------
    # 2. LoRA training — sweep batch sizes, with and without grad_ckpt
    # -----------------------------------------------------------------------
    torch.set_grad_enabled(True)
    model.train()
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, lora_cfg)
    lora_model.enable_input_require_grads()

    for use_ckpt in (False, True):
        tag = "grad_ckpt ON" if use_ckpt else "no grad_ckpt"
        print(f"\n=== 2. LoRA training ({tag}, seq=2048) ===")
        if use_ckpt:
            lora_model.gradient_checkpointing_enable()
        else:
            lora_model.gradient_checkpointing_disable()
        for bs in (1, 2, 4, 8):
            lora_model.train()
            inp = torch.randint(0, 1000, (bs, SEQ_LEN), device=DEVICE)
            lbl = inp.clone()
            lbl[:, :SEQ_LEN // 2] = -100
            reset()
            before = peak()
            try:
                out = lora_model(input_ids=inp, labels=lbl)
                out.loss.backward()
                report(f"forward+backward bs={bs}", before)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {'forward+backward bs='+str(bs):<45} OOM")
            del inp, lbl
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 3. Probe extraction — sweep batch sizes
    # -----------------------------------------------------------------------
    print("\n=== 3. Probe extraction (inference_mode, seq=2048) ===")
    lora_model.gradient_checkpointing_disable()
    model.eval()
    for bs in (16, 32, 64, 128):
        batch = torch.randint(0, 1000, (bs, SEQ_LEN), device=DEVICE)
        reset()
        before = peak()
        try:
            with torch.inference_mode():
                model(input_ids=batch)
            report(f"forward pass bs={bs}", before)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {'forward pass bs='+str(bs):<45} OOM")
        del batch
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
