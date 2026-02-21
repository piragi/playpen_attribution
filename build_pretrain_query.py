from __future__ import annotations

"""Build attribution datasets for continued pretraining experiment.

Creates two datasets (stored as pre-tokenized HuggingFace datasets):

  score_pool  — FineWeb packed chunks, same data used in continued pretraining
                (labels == input_ids, full causal supervision)

  attr_query  — WinoGrande + ARC-Challenge training examples
                (labels mask prompt with -100, only answer tokens supervised)

Both datasets carry input_ids / labels / length columns so Bergson skips its
internal tokenization step (it only tokenises when input_ids is absent).

The manifest written here is directly consumable by score.py:

    uv run score.py \\
        --manifest runs/pretrain_attribution_v1/manifest.json \\
        --adapter-path runs/pretrain_270m_v1 \\
        --base-model runs/pretrain_270m_v1 \\
        --pool-split score_pool \\
        --query-split attr_query \\
        --tokenization-mode bergson_chat \\
        --score-mode mean \\
        --preconditioning-mode query \\
        --projection-dim 32 \\
        --subset-k 5000 \\
        --output-dir runs/pretrain_attribution_v1/attribution_mean_k5000
"""

import json
import random
import shutil
from pathlib import Path
from typing import Iterator

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

CONFIG = {
    # Output
    "run_dir": "runs/pretrain_attribution_v2",
    # Pretrained model + tokenizer (full checkpoint, not PEFT)
    "base_model": "runs/pretrain_270m_v2",
    # FineWeb pool — identical parameters to the pretrain.py run
    "fineweb_dataset": "HuggingFaceFW/fineweb",
    "fineweb_config": "CC-MAIN-2024-10",
    "fineweb_max_tokens": 200_000_000,
    "chunk_size": 1024,           # must match pretrain.py --max-length
    "pool_progress_every": 5_000,
    # Benchmark query
    "winogrande_config": "winogrande_xl",
    "n_winogrande": 250,
    "n_arc_challenge": 250,
    "max_query_length": 256,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# FineWeb pool
# ---------------------------------------------------------------------------

def token_stream(dataset, tokenizer, max_tokens: int) -> Iterator[list[int]]:
    """Yield per-document token lists (EOS appended). Same logic as pretrain.py."""
    eos_id = tokenizer.eos_token_id
    total = 0
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            ids = ids + [eos_id]
        yield ids
        total += len(ids)
        if 0 < max_tokens <= total:
            break


def build_fineweb_pool(cfg: dict, tokenizer) -> Dataset:
    """Stream FineWeb, tokenize, pack into chunks via a generator.

    Uses Dataset.from_generator() to avoid holding ~195K Python dicts in memory
    (~11GB) — the Arrow writer processes rows one at a time and writes to disk.
    """
    chunk_size = cfg["chunk_size"]
    every = cfg["pool_progress_every"]

    def _gen():
        print(f"Streaming {cfg['fineweb_dataset']}/{cfg['fineweb_config']} ...")
        raw = load_dataset(
            cfg["fineweb_dataset"],
            cfg["fineweb_config"],
            split="train",
            streaming=True,
        )
        buffer: list[int] = []
        idx = 0
        for ids in token_stream(raw, tokenizer, cfg["fineweb_max_tokens"]):
            buffer.extend(ids)
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                if idx > 0 and idx % every == 0:
                    print(
                        f"\r  packed {idx:,} chunks "
                        f"({idx * chunk_size / 1e6:.1f}M tokens)",
                        end="", flush=True,
                    )
                yield {
                    "row_id": f"fineweb-{idx}",
                    "input_ids": chunk,
                    "labels": list(chunk),
                    "length": chunk_size,
                    "outcome": "",
                    "pair_key": "",
                }
                idx += 1
        print(f"\nBuilt {idx:,} FineWeb pool chunks × {chunk_size} tokens")

    ds: Dataset = Dataset.from_generator(_gen)  # type: ignore[assignment]
    print(f"  {len(ds):,} pool chunks materialized")
    return ds


# ---------------------------------------------------------------------------
# Benchmark query formatting
# ---------------------------------------------------------------------------

DIGIT_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}


def format_winogrande(row: dict) -> tuple[str, str] | None:
    """Return (prompt, completion) for a WinoGrande row, or None if malformed."""
    sentence = str(row.get("sentence", "")).strip()
    option1  = str(row.get("option1", "")).strip()
    option2  = str(row.get("option2", "")).strip()
    answer   = str(row.get("answer", "")).strip()  # "1" or "2"

    if not sentence or not option1 or not option2 or answer not in ("1", "2"):
        return None

    correct = option1 if answer == "1" else option2
    prompt = (
        f"Fill in the blank:\n{sentence}\n"
        f"Option 1: {option1}\nOption 2: {option2}\n"
        f"Answer:"
    )
    return prompt, f" {correct}"


def format_arc(row: dict) -> tuple[str, str] | None:
    """Return (prompt, completion) for an ARC row, or None if malformed."""
    question   = str(row.get("question", "")).strip()
    choices    = row.get("choices", {})
    answer_key = str(row.get("answerKey", "")).strip()

    if not question or not choices or not answer_key:
        return None

    texts  = choices.get("text",  [])
    labels = choices.get("label", [])
    if not texts or not labels or len(texts) != len(labels):
        return None

    # Normalise digit keys ("1","2",...) to letters ("A","B",...)
    if answer_key in DIGIT_TO_LETTER:
        answer_key = DIGIT_TO_LETTER[answer_key]

    choice_lines = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
    prompt = f"Question: {question}\n{choice_lines}\nAnswer:"
    return prompt, f" {answer_key}"


def tokenize_pair(
    prompt: str,
    completion: str,
    tokenizer,
    max_length: int,
) -> dict | None:
    """
    Tokenize prompt+completion without special tokens.
    Prompt tokens are masked with -100 in labels; only completion is supervised.
    Returns None if there are no supervised tokens.
    """
    eos = tokenizer.eos_token or ""
    if eos and not completion.endswith(eos):
        completion = completion + eos

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    full_ids   = full_ids[:max_length]

    supervised_start = min(len(prompt_ids), len(full_ids))
    labels = [-100] * supervised_start + full_ids[supervised_start:]

    if not any(l != -100 for l in labels):
        return None  # nothing to supervise

    return {
        "input_ids": full_ids,
        "labels":    labels,
        "length":    len(full_ids),
    }


def _sample_split(
    hf_dataset,
    formatter,
    prefix: str,
    n: int,
    tokenizer,
    max_length: int,
    rng: random.Random,
) -> list[dict]:
    indices = list(range(len(hf_dataset)))
    rng.shuffle(indices)
    rows = []
    for idx in indices:
        if len(rows) >= n:
            break
        pair = formatter(hf_dataset[idx])
        if pair is None:
            continue
        tok = tokenize_pair(pair[0], pair[1], tokenizer, max_length)
        if tok is None:
            continue
        rows.append({
            "row_id":   f"{prefix}-{idx}",
            "source":   prefix,
            "outcome":  "",
            "pair_key": "",
            **tok,
        })
    return rows


def build_benchmark_query(cfg: dict, tokenizer) -> Dataset:
    rng = random.Random(cfg["seed"])
    rows: list[dict] = []

    print(f"Loading WinoGrande ({cfg['winogrande_config']}) train split ...")
    wg = load_dataset("winogrande", cfg["winogrande_config"], split="train")
    wg_rows = _sample_split(
        wg, format_winogrande, "winogrande",
        cfg["n_winogrande"], tokenizer, cfg["max_query_length"], rng,
    )
    print(f"  {len(wg_rows)} WinoGrande examples kept")
    rows.extend(wg_rows)

    print("Loading ARC-Challenge train split ...")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_rows = _sample_split(
        arc, format_arc, "arc_challenge",
        cfg["n_arc_challenge"], tokenizer, cfg["max_query_length"], rng,
    )
    print(f"  {len(arc_rows)} ARC-Challenge examples kept")
    rows.extend(arc_rows)

    print(f"Total query examples: {len(rows)}")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG
    run_dir  = Path(cfg["run_dir"])
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- FineWeb pool (expensive; skip if already built) ---
    pool_path = data_dir / "score_pool"
    if pool_path.exists():
        print(f"Pool already exists at {pool_path} — loading ...")
        pool_ds = load_from_disk(str(pool_path))
        print(f"  {len(pool_ds):,} pool chunks loaded.")
    else:
        pool_ds = build_fineweb_pool(cfg, tokenizer)
        tmp = Path(str(pool_path) + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        pool_ds.save_to_disk(str(tmp))
        tmp.rename(pool_path)
        print(f"Saved pool → {pool_path}")

    # --- Benchmark query (fast) ---
    query_path = data_dir / "attr_query"
    query_ds = build_benchmark_query(cfg, tokenizer)
    if query_path.exists():
        shutil.rmtree(query_path)
    query_ds.save_to_disk(str(query_path))
    print(f"Saved query → {query_path}")

    # --- Manifest ---
    manifest = {
        "description": "FineWeb pretraining pool + WinoGrande/ARC-Challenge query",
        "base_model": cfg["base_model"],
        "splits": {
            "score_pool": {
                "path": str(pool_path.resolve()),
                "rows": len(pool_ds),
            },
            "attr_query": {
                "path": str(query_path.resolve()),
                "rows": len(query_ds),
            },
        },
        "config": cfg,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest → {manifest_path}")

    # --- Diagnostics ---
    sup_tokens = sum(
        sum(1 for l in row["labels"] if l != -100)
        for row in query_ds
    )
    print(f"\nQuery supervised tokens: {sup_tokens:,}")

    print("\n--- Next: run attribution ---")
    print(f"uv run score.py \\")
    print(f"    --manifest {manifest_path} \\")
    print(f"    --adapter-path {cfg['base_model']} \\")
    print(f"    --base-model {cfg['base_model']} \\")
    print(f"    --pool-split score_pool \\")
    print(f"    --query-split attr_query \\")
    print(f"    --tokenization-mode bergson_chat \\")
    print(f"    --score-mode mean \\")
    print(f"    --preconditioning-mode query \\")
    print(f"    --projection-dim 32 \\")
    print(f"    --subset-k 5000 \\")
    print(f"    --output-dir {run_dir}/attribution_mean_k5000")


if __name__ == "__main__":
    main()
