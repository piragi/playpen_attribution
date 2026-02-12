import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sae_lens import HookedSAETransformer, SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory


CONFIG = {
    # Input mode: "train_dataset" (all filtered train rows) or "ranked_examples"
    "input_mode": "train_dataset",
    # Source attribution output from score_samples.py (used when input_mode=ranked_examples)
    "ranked_examples_path": "./runs/taboo_attr/ranked_examples.json",
    # Filtered train split from score_samples.py
    "train_dataset_path": "./runs/taboo_attr/data/train",
    # Where to write SAE feature activity summary
    "output_path": "./runs/taboo_attr/sae_feature_activity_layer17_all_train.json",
    # Number of highest-attribution examples to analyze (ranked_examples mode only)
    "top_n": 100,
    # Optional cap for train_dataset mode. None = all rows.
    "max_examples": None,
    # Base model used for attribution / SFT
    "model_name": "google/gemma-3-1b-it",
    # SAE release (from google/gemma-scope-2-1b-it)
    # Common choices: gemma-scope-2-1b-it-res-all, -att-all, -mlp-all
    "sae_release": "gemma-scope-2-1b-it-res",
    # Select SAEs by width / l0 within the release
    "sae_width": "16k",
    "sae_l0": "small",
    # Optional explicit SAE IDs. If non-empty, this overrides width/l0 discovery.
    "sae_ids": ["layer_17_width_16k_l0_small"],
    # Inference controls
    "max_tokens": 384,
    "activation_threshold": 0.0,
    # How to rank top features in output:
    # token_count | prompt_count | total_activation | mean_activation | max_activation
    "ranking_metric": "mean_activation",
    # Ignore features with very few active tokens when ranking by activation strength.
    "min_token_count_for_ranking": 10,
    "top_features_per_layer": 50,
    # Optional dense outputs for downstream scripts.
    "save_per_sample_arrays": True,
    "per_sample_output_dir": "./runs/taboo_attr/sae_samples_layer17_all_train",
    "save_feature_token_counts": False,
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


def build_history(messages):
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)


def get_device_and_dtypes():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", "bfloat16", torch.bfloat16
        return "cuda", "float16", torch.float16
    return "cpu", "float32", torch.float32


def load_top_examples(path: Path, top_n: int):
    if not path.exists():
        raise FileNotFoundError(f"Ranked examples file not found: {path}")

    with path.open("r") as f:
        rows = json.load(f)

    top_rows = [row for row in rows if row.get("bucket") == "top"]
    if not top_rows:
        top_rows = rows
    top_rows.sort(key=lambda row: row.get("rank", 10**9))
    top_rows = top_rows[:top_n]

    examples = []
    for row in top_rows:
        prompt = ""
        messages = row.get("messages", [])
        if messages:
            prompt = build_history(messages).strip()
        elif row.get("prompt"):
            prompt = str(row["prompt"]).strip()

        if not prompt:
            continue

        examples.append(
            {
                "row_id": row.get("row_id"),
                "task_id": row.get("task_id"),
                "outcome": row.get("outcome"),
                "score": row.get("score"),
                "prompt": prompt,
            }
        )

    if not examples:
        raise ValueError("No valid prompts found in ranked examples.")

    return examples


def load_train_examples(path: Path, max_examples):
    if not path.exists():
        raise FileNotFoundError(f"Train dataset path not found: {path}")

    dataset = load_from_disk(str(path))
    examples = []
    for idx in range(len(dataset)):
        row = dataset[int(idx)]
        prompt = ""
        messages = row.get("messages", [])
        if messages:
            prompt = build_history(messages).strip()
        else:
            prompt_part = str(row.get("prompt", "")).strip()
            completion_part = str(row.get("completion", "")).strip()
            if prompt_part and completion_part:
                prompt = f"{prompt_part}\n\nASSISTANT: {completion_part}"
            else:
                prompt = prompt_part or completion_part

        if not prompt:
            continue

        meta = row.get("meta") or {}
        examples.append(
            {
                "dataset_index": idx,
                "row_id": row.get("row_id", idx),
                "task_id": meta.get("task_id", row.get("task_id")),
                "outcome": meta.get("outcome", row.get("outcome")),
                "score": None,
                "prompt": prompt,
            }
        )

        if max_examples is not None and len(examples) >= int(max_examples):
            break

    if not examples:
        raise ValueError("No valid prompts found in train dataset.")

    return examples


def load_examples():
    mode = CONFIG["input_mode"]
    if mode == "ranked_examples":
        return load_top_examples(
            path=Path(CONFIG["ranked_examples_path"]),
            top_n=int(CONFIG["top_n"]),
        )
    if mode == "train_dataset":
        return load_train_examples(
            path=Path(CONFIG["train_dataset_path"]),
            max_examples=CONFIG["max_examples"],
        )
    raise ValueError("Invalid input_mode. Use 'train_dataset' or 'ranked_examples'.")


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
        hook_name = get_hook_name(sae)
        layer = get_layer_from_sae_id(sae_id)
        sae_infos.append(
            {
                "sae_id": sae_id,
                "layer": layer,
                "hook_name": hook_name,
                "d_sae": int(sae.cfg.d_sae),
                "sae": sae,
            }
        )

    return sae_infos


def init_layer_stats(sae_infos):
    stats = {}
    for info in sae_infos:
        d_sae = info["d_sae"]
        stats[info["sae_id"]] = {
            "token_counts": torch.zeros(d_sae, dtype=torch.int64),
            "prompt_counts": torch.zeros(d_sae, dtype=torch.int64),
            "activation_sums": torch.zeros(d_sae, dtype=torch.float64),
            "activation_max": torch.full(
                (d_sae,), float("-inf"), dtype=torch.float64
            ),
        }
    return stats


def init_per_sample_arrays(sae_infos, n_examples: int):
    outputs = {}
    for info in sae_infos:
        d_sae = info["d_sae"]
        block = {
            "feature_presence": np.zeros((n_examples, d_sae), dtype=np.bool_),
            "sample_stats": np.zeros(
                (n_examples, len(SAMPLE_STAT_NAMES)),
                dtype=np.float32,
            ),
            "seq_lens": np.zeros(n_examples, dtype=np.int32),
        }
        if CONFIG["save_feature_token_counts"]:
            block["feature_token_counts"] = np.zeros((n_examples, d_sae), dtype=np.uint16)
        outputs[info["sae_id"]] = block
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
    prompt_feature_counts = feature_token_counts.to(torch.int64).cpu().numpy()
    return stats, prompt_feature_active, prompt_feature_counts, seq_len


def encode_examples(model, sae_infos, layer_stats, per_sample_arrays, examples):
    hook_names = sorted({info["hook_name"] for info in sae_infos})
    total_tokens = 0

    for i, example in enumerate(examples):
        tokens = model.to_tokens(example["prompt"])
        if CONFIG["max_tokens"] and tokens.shape[1] > CONFIG["max_tokens"]:
            tokens = tokens[:, : CONFIG["max_tokens"]]
        total_tokens += int(tokens.shape[1])

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for info in sae_infos:
            sae_id = info["sae_id"]
            sae = info["sae"]
            hook_name = info["hook_name"]
            d_sae = info["d_sae"]

            acts = cache[hook_name]
            sae_acts = sae.encode(acts)
            if CONFIG["activation_threshold"] > 0.0:
                active = sae_acts > CONFIG["activation_threshold"]
            else:
                active = sae_acts > 0

            flat_active = active.reshape(-1, d_sae)
            token_counts = flat_active.sum(dim=0).to(torch.int64).cpu()
            prompt_counts = flat_active.any(dim=0).to(torch.int64).cpu()

            active_values = (
                sae_acts.masked_fill(~active, 0.0)
                .reshape(-1, d_sae)
                .sum(dim=0)
                .to(torch.float64)
                .detach()
                .cpu()
            )
            feature_max = (
                sae_acts.masked_fill(~active, float("-inf"))
                .reshape(-1, d_sae)
                .max(dim=0)
                .values.to(torch.float64)
                .detach()
                .cpu()
            )

            layer_stats[sae_id]["token_counts"] += token_counts
            layer_stats[sae_id]["prompt_counts"] += prompt_counts
            layer_stats[sae_id]["activation_sums"] += active_values
            layer_stats[sae_id]["activation_max"] = torch.maximum(
                layer_stats[sae_id]["activation_max"], feature_max
            )

            if per_sample_arrays is not None:
                sample_stats, prompt_feature_active, prompt_feature_counts, seq_len = (
                    sample_stats_from_sae(sae_acts=sae_acts, active=active)
                )
                per_sample_arrays[sae_id]["feature_presence"][i] = prompt_feature_active
                per_sample_arrays[sae_id]["sample_stats"][i] = sample_stats
                per_sample_arrays[sae_id]["seq_lens"][i] = int(seq_len)
                if CONFIG["save_feature_token_counts"]:
                    clipped_counts = np.minimum(prompt_feature_counts, 65535).astype(np.uint16)
                    per_sample_arrays[sae_id]["feature_token_counts"][i] = clipped_counts

        row_num = i + 1
        if row_num % 10 == 0 or row_num == len(examples):
            print(f"processed {row_num}/{len(examples)} prompts")

    return total_tokens


def select_feature_order(token_counts, prompt_counts, activation_sums, activation_max):
    token_counts_safe = np.maximum(token_counts, 1)
    mean_activation = activation_sums / token_counts_safe

    min_tokens = int(CONFIG["min_token_count_for_ranking"])
    candidate_ids = np.where(token_counts >= min_tokens)[0]
    if candidate_ids.size == 0:
        candidate_ids = np.where(token_counts > 0)[0]

    metric = CONFIG["ranking_metric"]
    if metric == "token_count":
        values = token_counts
    elif metric == "prompt_count":
        values = prompt_counts
    elif metric == "total_activation":
        values = activation_sums
    elif metric == "max_activation":
        values = activation_max
    elif metric == "mean_activation":
        values = mean_activation
    else:
        raise ValueError(
            "Unknown ranking_metric. Use one of: "
            "token_count, prompt_count, total_activation, mean_activation, max_activation"
        )

    sorted_ids = candidate_ids[np.argsort(-values[candidate_ids])]
    return sorted_ids, mean_activation


def summarize_layers(sae_infos, layer_stats, num_prompts: int):
    summaries = []
    for info in sorted(
        sae_infos,
        key=lambda x: (x["layer"] is None, x["layer"], x["sae_id"]),
    ):
        sae_id = info["sae_id"]
        token_counts = layer_stats[sae_id]["token_counts"].detach().numpy()
        prompt_counts = layer_stats[sae_id]["prompt_counts"].detach().numpy()
        activation_sums = layer_stats[sae_id]["activation_sums"].detach().numpy()
        activation_max = layer_stats[sae_id]["activation_max"].detach().numpy()

        sorted_ids, mean_activation_arr = select_feature_order(
            token_counts=token_counts,
            prompt_counts=prompt_counts,
            activation_sums=activation_sums,
            activation_max=activation_max,
        )
        top_ids = sorted_ids[: CONFIG["top_features_per_layer"]]

        top_features = []
        for feat_id in top_ids:
            feat_token_count = int(token_counts[feat_id])
            feat_prompt_count = int(prompt_counts[feat_id])
            mean_activation = float(mean_activation_arr[feat_id]) if feat_token_count > 0 else 0.0
            max_activation = float(activation_max[feat_id])
            if not np.isfinite(max_activation):
                max_activation = 0.0
            top_features.append(
                {
                    "feature_id": int(feat_id),
                    "token_count": feat_token_count,
                    "prompt_count": feat_prompt_count,
                    "prompt_fraction": float(feat_prompt_count / num_prompts),
                    "total_activation": float(activation_sums[feat_id]),
                    "mean_activation_when_active": mean_activation,
                    "max_activation": max_activation,
                }
            )

        summaries.append(
            {
                "sae_id": sae_id,
                "layer": info["layer"],
                "hook_name": info["hook_name"],
                "d_sae": info["d_sae"],
                "num_active_features": int((token_counts > 0).sum()),
                "ranking_metric": CONFIG["ranking_metric"],
                "min_token_count_for_ranking": int(CONFIG["min_token_count_for_ranking"]),
                "top_features": top_features,
            }
        )

    return summaries


def safe_name(name: str):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def save_per_sample_outputs(per_sample_arrays, examples):
    if not CONFIG["save_per_sample_arrays"] or per_sample_arrays is None:
        return []

    output_dir = Path(CONFIG["per_sample_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "examples.jsonl"
    with metadata_path.open("w") as f:
        for example in examples:
            row = {
                "dataset_index": example.get("dataset_index"),
                "row_id": example.get("row_id"),
                "task_id": example.get("task_id"),
                "outcome": example.get("outcome"),
                "score": example.get("score"),
            }
            f.write(json.dumps(row) + "\n")

    written = [str(metadata_path)]
    for sae_id, arrs in per_sample_arrays.items():
        out_file = output_dir / f"{safe_name(sae_id)}.npz"
        payload = {
            "feature_presence": arrs["feature_presence"],
            "sample_stats": arrs["sample_stats"],
            "seq_lens": arrs["seq_lens"],
            "stat_feature_names": np.asarray(SAMPLE_STAT_NAMES, dtype="<U64"),
        }
        if CONFIG["save_feature_token_counts"]:
            payload["feature_token_counts"] = arrs["feature_token_counts"]

        np.savez_compressed(out_file, **payload)
        written.append(str(out_file))

    return written


def main():
    output_path = Path(CONFIG["output_path"])

    device, sae_dtype, model_dtype = get_device_and_dtypes()
    print(f"device={device} sae_dtype={sae_dtype}")

    examples = load_examples()
    print(f"loaded {len(examples)} prompts from input_mode={CONFIG['input_mode']}")

    sae_ids = discover_sae_ids()
    print(f"selected {len(sae_ids)} SAE IDs from release={CONFIG['sae_release']}")

    model = HookedSAETransformer.from_pretrained_no_processing(
        CONFIG["model_name"],
        device=device,
        dtype=model_dtype,
    )
    model.eval()
    print(f"loaded model={CONFIG['model_name']}")

    sae_infos = load_saes(sae_ids, device=device, sae_dtype=sae_dtype)
    layer_stats = init_layer_stats(sae_infos)
    per_sample_arrays = (
        init_per_sample_arrays(sae_infos, len(examples))
        if CONFIG["save_per_sample_arrays"]
        else None
    )
    total_tokens = encode_examples(
        model=model,
        sae_infos=sae_infos,
        layer_stats=layer_stats,
        per_sample_arrays=per_sample_arrays,
        examples=examples,
    )

    layer_summaries = summarize_layers(
        sae_infos=sae_infos,
        layer_stats=layer_stats,
        num_prompts=len(examples),
    )

    output = {
        "config": CONFIG,
        "model_name": CONFIG["model_name"],
        "sae_release": CONFIG["sae_release"],
        "num_prompts": len(examples),
        "total_tokens": total_tokens,
        "examples": [
            {
                "row_id": e["row_id"],
                "task_id": e["task_id"],
                "outcome": e["outcome"],
                "score": e["score"],
            }
            for e in examples
        ],
        "layers": layer_summaries,
    }
    written_files = save_per_sample_outputs(per_sample_arrays, examples)
    if written_files:
        output["per_sample_files"] = written_files

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"saved SAE activity summary to: {output_path}")
    if written_files:
        print(f"saved per-sample outputs to: {CONFIG['per_sample_output_dir']}")


if __name__ == "__main__":
    main()
