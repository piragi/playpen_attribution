from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sae_lens import HookedSAETransformer, SAE

SAMPLE_STAT_NAMES = [
    "seq_len",
    "active_entry_fraction",
    "active_feature_fraction",
    "mean_active_activation",
    "std_active_activation",
    "max_active_activation",
    "token_active_mean",
    "token_active_std",
    "token_active_cv",
    "feature_concentration_hhi",
]


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def build_prompt_completion(messages: list[dict]) -> tuple[str, str]:
    assistant_positions = [
        i for i, msg in enumerate(messages) if str(msg.get("role", "")).lower() == "assistant"
    ]
    if not assistant_positions:
        return "", ""

    last_assistant = assistant_positions[-1]
    prompt_lines = []
    for msg in messages[:last_assistant]:
        role = str(msg.get("role", "")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            prompt_lines.append(f"{role}: {content}")
    prompt = "\n\n".join(prompt_lines).strip()
    completion = str(messages[last_assistant].get("content", "")).strip()
    return prompt, completion


def resolve_dataset_path(manifest_path: str | None, split: str, dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    if not manifest_path:
        raise ValueError("Either --dataset-path or --manifest must be provided.")

    manifest = json.loads(Path(manifest_path).read_text())
    split_entry = manifest.get("splits", {}).get(split)
    if not split_entry or "path" not in split_entry:
        raise KeyError(f"Split '{split}' not found in manifest: {manifest_path}")
    return Path(split_entry["path"])


def load_examples(path: Path, max_examples: int | None) -> list[dict]:
    ds = load_from_disk(str(path))
    limit = len(ds) if max_examples is None else min(len(ds), int(max_examples))

    examples: list[dict] = []
    for i in range(limit):
        row = ds[int(i)]

        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", "")).strip()
        if not prompt and not completion and "messages" in row:
            prompt, completion = build_prompt_completion(list(row["messages"]))

        text = (prompt + ("\n\n" + completion if completion else "")).strip()
        if not text:
            raise ValueError(f"Empty text at row {i} in {path}")

        examples.append(
            {
                "dataset_index": int(i),
                "row_id": str(row.get("row_id", i)),
                "task_id": row.get("task_id"),
                "outcome": row.get("outcome"),
                "text": text,
            }
        )
    return examples


def get_device_and_dtypes(device_arg: str) -> tuple[str, str, torch.dtype]:
    if device_arg != "auto":
        if device_arg == "cuda":
            if torch.cuda.is_bf16_supported():
                return "cuda", "bfloat16", torch.bfloat16
            return "cuda", "float16", torch.float16
        return device_arg, "float32", torch.float32

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", "bfloat16", torch.bfloat16
        return "cuda", "float16", torch.float16
    return "cpu", "float32", torch.float32


def sample_stats_from_sae(sae_acts: torch.Tensor, active: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
    d_sae = sae_acts.shape[-1]
    flat_acts = sae_acts.reshape(-1, d_sae)
    flat_active = active.reshape(-1, d_sae)

    seq_len = int(sae_acts.shape[1])
    total_positions = float(flat_active.numel())
    total_active = float(flat_active.sum().item())

    feature_token_counts = flat_active.sum(dim=0).to(torch.float32)
    active_feature_fraction = float((feature_token_counts > 0).float().mean().item())
    active_entry_fraction = float(total_active / max(total_positions, 1.0))

    token_active_counts = flat_active.sum(dim=1).to(torch.float32)
    token_active_mean = float(token_active_counts.mean().item())
    token_active_std = float(token_active_counts.std(unbiased=False).item())
    token_active_cv = float(token_active_std / (token_active_mean + 1e-8))

    if total_active > 0:
        active_vals = flat_acts[flat_active]
        mean_active = float(active_vals.mean().item())
        std_active = float(active_vals.std(unbiased=False).item())
        max_active = float(active_vals.max().item())
    else:
        mean_active = 0.0
        std_active = 0.0
        max_active = 0.0

    feature_token_sum = float(feature_token_counts.sum().item())
    if feature_token_sum > 0:
        p = feature_token_counts / feature_token_sum
        hhi = float((p * p).sum().item())
    else:
        hhi = 0.0

    stats = np.array(
        [
            seq_len,
            active_entry_fraction,
            active_feature_fraction,
            mean_active,
            std_active,
            max_active,
            token_active_mean,
            token_active_std,
            token_active_cv,
            hhi,
        ],
        dtype=np.float32,
    )
    prompt_feature_active = (feature_token_counts > 0).to(torch.bool).cpu().numpy()
    return stats, prompt_feature_active, seq_len


def save_outputs(
    output_dir: Path,
    examples: list[dict],
    sae_id: str,
    feature_presence: np.ndarray,
    sample_stats: np.ndarray,
    seq_lens: np.ndarray,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "examples.jsonl"
    with metadata_path.open("w") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "dataset_index": ex["dataset_index"],
                        "row_id": ex["row_id"],
                        "task_id": ex.get("task_id"),
                        "outcome": ex.get("outcome"),
                    }
                )
                + "\n"
            )

    npz_path = output_dir / f"{safe_name(sae_id)}.npz"
    np.savez_compressed(
        npz_path,
        feature_presence=feature_presence,
        sample_stats=sample_stats,
        seq_lens=seq_lens,
        stat_feature_names=np.asarray(SAMPLE_STAT_NAMES, dtype="<U64"),
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved arrays:   {npz_path}")
    print(f"saved metadata: {metadata_path}")
    print(f"saved summary:  {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-sample SAE activations.")
    parser.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    parser.add_argument("--split", type=str, default="train_base")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/simple_wordguesser_v1/sae_samples_layer17_train_base",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2-1b-it-res")
    parser.add_argument("--sae-id", type=str, default="layer_17_width_16k_l0_small")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.manifest, args.split, args.dataset_path)
    examples = load_examples(dataset_path, args.max_examples)
    if not examples:
        raise RuntimeError(f"No examples found at {dataset_path}")

    device, sae_dtype, model_dtype = get_device_and_dtypes(args.device)
    print(f"dataset: {dataset_path}")
    print(f"examples: {len(examples)}")
    print(f"device={device} sae_dtype={sae_dtype}")

    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model_name,
        device=device,
        dtype=model_dtype,
    )
    model.eval()

    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
        dtype=sae_dtype,
    )
    sae.eval()

    metadata = getattr(sae.cfg, "metadata", None)
    hook_name = getattr(metadata, "hook_name", None) or getattr(sae.cfg, "hook_name", None)
    if hook_name is None:
        raise RuntimeError("Could not resolve SAE hook name.")

    d_sae = int(sae.cfg.d_sae)
    n = len(examples)
    feature_presence = np.zeros((n, d_sae), dtype=np.bool_)
    sample_stats = np.zeros((n, len(SAMPLE_STAT_NAMES)), dtype=np.float32)
    seq_lens = np.zeros(n, dtype=np.int32)

    total_tokens = 0
    for i, ex in enumerate(examples):
        tokens = model.to_tokens(ex["text"])
        if args.max_tokens and tokens.shape[1] > args.max_tokens:
            tokens = tokens[:, : args.max_tokens]
        total_tokens += int(tokens.shape[1])

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)

        if args.activation_threshold > 0.0:
            active = sae_acts > args.activation_threshold
        else:
            active = sae_acts > 0

        stats, row_presence, seq_len = sample_stats_from_sae(sae_acts, active)
        feature_presence[i] = row_presence
        sample_stats[i] = stats
        seq_lens[i] = int(seq_len)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"processed {i + 1}/{n}")

    summary = {
        "dataset_path": str(dataset_path),
        "split": args.split,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "hook_name": hook_name,
        "n_examples": int(n),
        "max_tokens": int(args.max_tokens),
        "total_tokens": int(total_tokens),
        "device": device,
    }

    save_outputs(
        output_dir=Path(args.output_dir),
        examples=examples,
        sae_id=args.sae_id,
        feature_presence=feature_presence,
        sample_stats=sample_stats,
        seq_lens=seq_lens,
        summary=summary,
    )


if __name__ == "__main__":
    main()
