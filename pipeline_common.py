"""Shared helpers for the attribution pipeline scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_HOME_DEFAULT = str(Path.home() / ".cache" / "huggingface")
DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B"
ATTN_IMPLEMENTATION = "sdpa"


def ensure_hf_home_env() -> str:
    """Set HF_HOME if missing and return it."""
    hf_home = os.environ.setdefault("HF_HOME", HF_HOME_DEFAULT)
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    return hf_home


def infer_instruct_tokenizer_model(base_model: str) -> str:
    """Map base model ID/path to its instruct tokenizer model ID."""
    if base_model.endswith("-Instruct"):
        return base_model
    if base_model.endswith("-pt"):
        return base_model[:-3] + "-it"
    return base_model + "-Instruct"


def resolve_device_dtype() -> tuple[str, torch.dtype]:
    """Return (device, dtype) based on what's available."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return "cpu", torch.float32


def pad_tokenized_batch(
    input_rows: Sequence[Sequence[int]],
    label_rows: Sequence[Sequence[int]],
    *,
    input_pad_token_id: int = 0,
    label_pad_token_id: int = -100,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad tokenized rows to batch max length and return (input_ids, labels)."""
    max_len = max(len(ids) for ids in input_rows)
    ids_padded = [list(ids) + [input_pad_token_id] * (max_len - len(ids)) for ids in input_rows]
    labels_padded = [list(lbls) + [label_pad_token_id] * (max_len - len(lbls)) for lbls in label_rows]
    ids_t = torch.tensor(ids_padded, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels_padded, dtype=torch.long)  # keep on CPU
    return ids_t, labels_t


def last_response_token_positions(labels_t: torch.Tensor, *, label_pad_token_id: int = -100) -> torch.Tensor:
    """Return last non-pad label position per row, or final token if none exist."""
    mask = labels_t.ne(label_pad_token_id)
    seq_len = labels_t.shape[1]
    # argmax on reversed mask gives distance from end to the last response token.
    last_from_end = mask.flip(dims=[1]).to(torch.int64).argmax(dim=1)
    last_idx = seq_len - 1 - last_from_end
    has_response = mask.any(dim=1)
    fallback = torch.full_like(last_idx, seq_len - 1)
    return torch.where(has_response, last_idx, fallback)


def pool_hidden_at_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather hidden states at one position per sequence in the batch."""
    idx = positions.to(hidden.device).unsqueeze(-1).unsqueeze(-1)
    idx = idx.expand(-1, 1, hidden.shape[-1])
    return hidden.gather(dim=1, index=idx).squeeze(1)


def load_tokenizer(base_model: str) -> AutoTokenizer:
    """Load the instruct tokenizer for a base model, setting pad_token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(infer_instruct_tokenizer_model(base_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_with_hook(
    base_model: str,
    adapter_path: str | None,
    extraction_layer: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[Any, dict]:
    """Load a CausalLM with optional PEFT adapter and register a residual stream hook.

    The hook captures the output of transformer layer `extraction_layer` into
    captured['acts'] on every forward pass. For decoder-only models, pooling
    the last response token from this hidden state is the standard reward-model
    pooling strategy (identical to InstructGPT-style scoring heads).

    Returns (model, captured).
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_m: Any, _i: Any, out: Any) -> None:
        captured["acts"] = out[0] if isinstance(out, tuple) else out

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, attn_implementation=ATTN_IMPLEMENTATION
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False, autocast_adapter_dtype=False)
    else:
        model = base
    model.eval()
    model.config.use_cache = False
    get_transformer_layers_for_hook(model)[extraction_layer].register_forward_hook(_hook)
    print(f"Loaded {'adapter' if adapter_path else 'base'} model, hook at layer {extraction_layer}")
    return model, captured


def mask_prompt(
    messages: list[dict],
    tokenizer: Any,
    max_length: int,
) -> dict | None:
    """Tokenize a conversation with prompt tokens masked to -100 in labels.

    Only assistant turn tokens are supervised; all other tokens are masked.
    Returns {"input_ids", "labels", "length"} or None if too long or empty.
    """
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(full_ids) > max_length:
        return None
    labels = [-100] * len(full_ids)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        start = len(tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True))
        end = min(
            len(tokenizer.apply_chat_template(messages[:i + 1], tokenize=True, add_generation_prompt=False)),
            len(full_ids),
        )
        labels[start:end] = full_ids[start:end]
    if all(lbl == -100 for lbl in labels):
        return None
    return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


def get_magpie_score(row: dict) -> float:
    """Extract Magpie quality score from a SmolTalk row."""
    for key in ("score", "quality_score", "magpie_score"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def get_transformer_layers_for_hook(model: Any) -> Any:
    """Return the transformer layers container for base or PEFT-wrapped CausalLMs."""
    for path in [
        ("model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
        ("base_model", "layers"),
    ]:
        try:
            cur = model
            for attr in path:
                cur = getattr(cur, attr)
            return cur
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate transformer layers on model. "
        "Expected one of: model.layers / base_model.model.model.layers / base_model.model.layers."
    )
