import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sae_lens import HookedSAETransformer, SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformers import AutoModelForCausalLM

from prompts import build_prompt_completion, safe_name

CONFIG = {
    "train_dataset_path": "./runs/taboo_attr/data/train",
    "output_dir": "./runs/taboo_attr/sae_samples_layer17_all_train",
    "max_examples": None,  # None = all rows
    # Base model name (used for HookedSAETransformer config lookup)
    "model_name": "google/gemma-3-1b-it",
    # Path to merged model (LoRA merged into base). Set to None to use base model.
    "merged_model_path": None,
    "sae_release": "gemma-scope-2-1b-it-res",
    "sae_width": "16k",
    "sae_l0": "small",
    "sae_ids": ["layer_17_width_16k_l0_small"],  # overrides width/l0 discovery if non-empty
    "max_tokens": 384,
    "activation_threshold": 0.0,
}

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


def get_device_and_dtypes():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", "bfloat16", torch.bfloat16
        return "cuda", "float16", torch.float16
    return "cpu", "float32", torch.float32


def load_examples(path: Path, max_examples):
    if not path.exists():
        raise FileNotFoundError(f"Train dataset path not found: {path}")

    dataset = load_from_disk(str(path))
    examples = []
    for idx in range(len(dataset)):
        row = dataset[int(idx)]
        # Use prompt+completion concatenation to match training/scoring format.
        # The saved dataset has both messages and prompt/completion columns;
        # prefer prompt/completion since that's what SFTTrainer and Bergson see.
        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", "")).strip()
        if prompt and completion:
            text = prompt + completion
        elif not prompt and not completion:
            # Fallback: reconstruct from messages if prompt/completion missing
            messages = row.get("messages", [])
            if messages:
                pc = build_prompt_completion(messages)
                text = pc["prompt"] + pc["completion"]
            else:
                text = ""
        else:
            text = prompt or completion

        if not text:
            raise ValueError(
                f"Empty text at dataset index {idx} (row_id={row.get('row_id', idx)}). "
                "This would cause index misalignment with attribution scores."
            )

        meta = row.get("meta") or {}
        examples.append({
            "dataset_index": idx,
            "row_id": row.get("row_id", idx),
            "task_id": meta.get("task_id", row.get("task_id")),
            "outcome": meta.get("outcome", row.get("outcome")),
            "text": text,
        })

        if max_examples is not None and len(examples) >= int(max_examples):
            break

    if not examples:
        raise ValueError("No valid prompts found in train dataset.")

    return examples


def discover_sae_ids():
    if CONFIG["sae_ids"]:
        return list(CONFIG["sae_ids"])

    release = CONFIG["sae_release"]
    pretrained = get_pretrained_saes_directory()
    if release not in pretrained:
        raise ValueError(
            f"Unknown SAE release: {release}. "
            "Check available release names in sae_lens pretrained SAEs directory."
        )

    saes_map = pretrained[release].saes_map
    pattern = re.compile(
        rf"^layer_(\d+)_width_{re.escape(CONFIG['sae_width'])}_l0_{re.escape(CONFIG['sae_l0'])}$"
    )

    matches = []
    for sae_id in saes_map:
        match = pattern.match(sae_id)
        if match:
            matches.append((int(match.group(1)), sae_id))

    if not matches:
        sample = list(saes_map.keys())[:10]
        raise ValueError(
            "No SAE IDs matched width/l0 selection. "
            f"release={release} width={CONFIG['sae_width']} l0={CONFIG['sae_l0']} "
            f"sample_ids={sample}"
        )

    matches.sort(key=lambda x: x[0])
    return [sae_id for _, sae_id in matches]


def get_hook_name(sae):
    metadata = getattr(sae.cfg, "metadata", None)
    hook_name = getattr(metadata, "hook_name", None)
    if hook_name is None:
        hook_name = getattr(sae.cfg, "hook_name", None)
    if hook_name is None:
        raise ValueError("Could not resolve SAE hook name from config metadata.")
    return hook_name


def get_layer_from_sae_id(sae_id: str):
    match = re.match(r"^layer_(\d+)_", sae_id)
    if not match:
        return None
    return int(match.group(1))


def load_saes(sae_ids, device: str, sae_dtype: str):
    sae_infos = []
    for i, sae_id in enumerate(sae_ids, start=1):
        print(f"loading SAE {i}/{len(sae_ids)}: {sae_id}")
        sae = SAE.from_pretrained(
            release=CONFIG["sae_release"],
            sae_id=sae_id,
            device=device,
            dtype=sae_dtype,
        )
        sae.eval()
        sae_infos.append({
            "sae_id": sae_id,
            "layer": get_layer_from_sae_id(sae_id),
            "hook_name": get_hook_name(sae),
            "d_sae": int(sae.cfg.d_sae),
            "sae": sae,
        })
    return sae_infos


def init_per_sample_arrays(sae_infos, n_examples: int):
    outputs = {}
    for info in sae_infos:
        d_sae = info["d_sae"]
        outputs[info["sae_id"]] = {
            "feature_presence": np.zeros((n_examples, d_sae), dtype=np.bool_),
            "sample_stats": np.zeros((n_examples, len(SAMPLE_STAT_NAMES)), dtype=np.float32),
            "seq_lens": np.zeros(n_examples, dtype=np.int32),
        }
    return outputs


def sample_stats_from_sae(sae_acts, active):
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

    stats = np.array([
        seq_len, active_entry_fraction, active_feature_fraction,
        mean_active, std_active, max_active,
        token_active_mean, token_active_std, token_active_cv, hhi,
    ], dtype=np.float32)

    prompt_feature_active = (feature_token_counts > 0).to(torch.bool).cpu().numpy()
    return stats, prompt_feature_active, seq_len


def encode_examples(model, sae_infos, per_sample_arrays, examples):
    hook_names = sorted({info["hook_name"] for info in sae_infos})
    total_tokens = 0

    for i, example in enumerate(examples):
        tokens = model.to_tokens(example["text"])
        if CONFIG["max_tokens"] and tokens.shape[1] > CONFIG["max_tokens"]:
            tokens = tokens[:, : CONFIG["max_tokens"]]
        total_tokens += int(tokens.shape[1])

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for info in sae_infos:
            sae_id = info["sae_id"]
            sae = info["sae"]
            hook_name = info["hook_name"]

            acts = cache[hook_name]
            sae_acts = sae.encode(acts)
            if CONFIG["activation_threshold"] > 0.0:
                active = sae_acts > CONFIG["activation_threshold"]
            else:
                active = sae_acts > 0

            stats, prompt_feature_active, seq_len = sample_stats_from_sae(sae_acts, active)
            per_sample_arrays[sae_id]["feature_presence"][i] = prompt_feature_active
            per_sample_arrays[sae_id]["sample_stats"][i] = stats
            per_sample_arrays[sae_id]["seq_lens"][i] = int(seq_len)

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"processed {i + 1}/{len(examples)} prompts")

    return total_tokens


def save_outputs(per_sample_arrays, examples, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "examples.jsonl"
    with metadata_path.open("w") as f:
        for example in examples:
            row = {
                "dataset_index": example.get("dataset_index"),
                "row_id": example.get("row_id"),
                "task_id": example.get("task_id"),
                "outcome": example.get("outcome"),
            }
            f.write(json.dumps(row) + "\n")

    for sae_id, arrs in per_sample_arrays.items():
        out_file = output_dir / f"{safe_name(sae_id)}.npz"
        np.savez_compressed(
            out_file,
            feature_presence=arrs["feature_presence"],
            sample_stats=arrs["sample_stats"],
            seq_lens=arrs["seq_lens"],
            stat_feature_names=np.asarray(SAMPLE_STAT_NAMES, dtype="<U64"),
        )
        print(f"saved {out_file}")

    print(f"saved metadata to {metadata_path}")


def main():
    device, sae_dtype, model_dtype = get_device_and_dtypes()
    print(f"device={device} sae_dtype={sae_dtype}")

    examples = load_examples(
        Path(CONFIG["train_dataset_path"]),
        max_examples=CONFIG["max_examples"],
    )
    print(f"loaded {len(examples)} prompts")

    sae_ids = discover_sae_ids()
    print(f"selected {len(sae_ids)} SAE IDs from release={CONFIG['sae_release']}")

    hf_model = None
    if CONFIG["merged_model_path"]:
        hf_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["merged_model_path"], torch_dtype=model_dtype,
        )
        print(f"loaded merged model from {CONFIG['merged_model_path']}")

    model = HookedSAETransformer.from_pretrained_no_processing(
        CONFIG["model_name"],
        hf_model=hf_model,
        device=device,
        dtype=model_dtype,
    )
    model.eval()
    print(f"loaded HookedSAETransformer (base={CONFIG['model_name']})")

    sae_infos = load_saes(sae_ids, device=device, sae_dtype=sae_dtype)
    per_sample_arrays = init_per_sample_arrays(sae_infos, len(examples))

    total_tokens = encode_examples(model, sae_infos, per_sample_arrays, examples)
    print(f"encoded {total_tokens} tokens total")

    save_outputs(per_sample_arrays, examples, Path(CONFIG["output_dir"]))


if __name__ == "__main__":
    main()
