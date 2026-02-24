"""Shared helpers for the attribution pipeline scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch

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


def resolve_device_dtype(device_arg: str) -> tuple[str, torch.dtype]:
    """Resolve a device string and preferred dtype for model loading."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dtype
        return "cpu", torch.float32
    if device_arg == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    return device_arg, torch.float32


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
