import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from bergson.config import DataConfig, IndexConfig, ReduceConfig, ScoreConfig
from bergson.data import load_scores
from bergson.reduce import reduce
from bergson.score.score import score_dataset
from transformers import AutoConfig


CONFIG = {
    "dataset_name": "colab-potsdam/playpen-data",
    "dataset_config": "interactions",
    "game": "taboo",
    "role": "WordGuesser",  # WordGuesser or WordDescriber
    "base_model": "google/gemma-3-1b-it",
    "adapter_path": "./taboo_sft_lora",
    "token_batch_size": 384,
    "projection_dim": 0,  # used only when preconditioning.mode == "none"
    "unit_normalize": True,
    "score_mode": "mean",  # mean, nearest, individual
    "top_k": 100,
    "output_root": "./runs/taboo_attr",
    "preconditioning": {
        # "none"  -> cosine-like / inner-product scoring with no H^-1
        # "query" -> use query preconditioner only (H from val/query side)
        # "mixed" -> blend query + train preconditioners with mixing_coefficient
        "mode": "mixed",
        # Used only when mode != "none". Must be > 0; this keeps H manageable.
        "projection_dim": 32,
        "mixing_coefficient": 0.99,
    },
}


def keep_example(example):
    meta = example["meta"]
    return (
        meta["game"] == CONFIG["game"]
        and meta["game_role"] == CONFIG["role"]
        and meta["outcome"] != "aborted"
    )


def build_history(messages):
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)


def remove_path(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def part_path(path: Path) -> Path:
    return Path(str(path) + ".part")


def cleanup_paths(paths):
    for path in paths:
        remove_path(path)


def get_preconditioner_mode() -> str:
    mode = CONFIG["preconditioning"]["mode"]
    valid = {"none", "query", "mixed"}
    if mode not in valid:
        raise ValueError(f"Invalid preconditioning mode: {mode}. Valid: {sorted(valid)}")
    return mode


def uses_query_preconditioner(mode: str) -> bool:
    return mode in {"query", "mixed"}


def uses_index_preconditioner(mode: str) -> bool:
    return mode == "mixed"


def get_effective_projection_dim(mode: str) -> int:
    if mode == "none":
        return int(CONFIG["projection_dim"])
    return int(CONFIG["preconditioning"]["projection_dim"])


def validate_preconditioner_config(mode: str, projection_dim: int):
    if mode != "none" and projection_dim <= 0:
        raise ValueError(
            "Preconditioning needs projection_dim > 0 in this setup. "
            "Set CONFIG['preconditioning']['projection_dim'] to a small value "
            "(e.g. 8 or 12)."
        )


def print_scoring_plan(mode: str, projection_dim: int):
    print(
        f"scoring mode={CONFIG['score_mode']} preconditioning={mode} "
        f"projection_dim={projection_dim}"
    )


def add_row_id(dataset):
    return dataset.map(lambda _, idx: {"row_id": idx}, with_indices=True)


def add_prompt_completion(example):
    messages = example["messages"]
    assistant_positions = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    if not assistant_positions:
        return {"prompt": "", "completion": ""}

    last_assistant_idx = assistant_positions[-1]
    prompt = build_history(messages[:last_assistant_idx]).strip()
    completion = messages[last_assistant_idx]["content"].strip()
    return {"prompt": prompt, "completion": completion}


def ensure_adapter_has_base_config():
    adapter_path = Path(CONFIG["adapter_path"])
    config_path = adapter_path / "config.json"
    if config_path.exists():
        return

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    AutoConfig.from_pretrained(CONFIG["base_model"]).save_pretrained(adapter_path)
    print(f"wrote base config.json to: {config_path}")


def save_filtered_splits(train_path: Path, val_path: Path):
    dataset = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_config"])
    train_dataset = add_row_id(dataset["train"].filter(keep_example)).map(add_prompt_completion)
    val_dataset = add_row_id(dataset["validation"].filter(keep_example)).map(add_prompt_completion)
    train_dataset = train_dataset.filter(lambda x: x["completion"] != "")
    val_dataset = val_dataset.filter(lambda x: x["completion"] != "")

    print(f"train rows: {len(train_dataset)}")
    print(f"validation rows: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Filtered train/val dataset is empty. Check CONFIG values.")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_path))
    val_dataset.save_to_disk(str(val_path))
    return train_dataset, train_path, val_path


def make_index_config(run_path: Path, data_path: Path, projection_dim: int, skip_preconditioners: bool):
    return IndexConfig(
        run_path=str(run_path),
        model=CONFIG["adapter_path"],
        tokenizer=CONFIG["adapter_path"],
        projection_dim=projection_dim,
        token_batch_size=CONFIG["token_batch_size"],
        skip_preconditioners=skip_preconditioners,
        data=DataConfig(
            dataset=str(data_path),
            split="train",
            prompt_column="prompt",
            completion_column="completion",
            truncation=True,
        ),
        distributed={"nproc_per_node": 1},
    )


def run_reduce(data_path: Path, run_path: Path, projection_dim: int, compute_preconditioners: bool):
    reduce_cfg = ReduceConfig(method="mean", unit_normalize=CONFIG["unit_normalize"])
    reduce(
        make_index_config(
            run_path,
            data_path,
            projection_dim=projection_dim,
            skip_preconditioners=not compute_preconditioners,
        ),
        reduce_cfg,
    )


def run_score(
    train_path: Path,
    query_run: Path,
    score_run: Path,
    projection_dim: int,
    mode: str,
    index_preconditioner_run: Path | None,
):
    score_cfg = ScoreConfig(
        query_path=str(query_run),
        score=CONFIG["score_mode"],
        unit_normalize=CONFIG["unit_normalize"],
        query_preconditioner_path=str(query_run) if uses_query_preconditioner(mode) else None,
        index_preconditioner_path=(
            str(index_preconditioner_run)
            if uses_index_preconditioner(mode) and index_preconditioner_run is not None
            else None
        ),
        mixing_coefficient=float(CONFIG["preconditioning"]["mixing_coefficient"]),
    )
    score_dataset(
        make_index_config(
            score_run,
            train_path,
            projection_dim=projection_dim,
            skip_preconditioners=True,
        ),
        score_cfg,
    )


def get_score_vector(score_run: Path) -> np.ndarray:
    scores = load_scores(score_run)
    names = list(scores.dtype.names or [])
    score_cols = [name for name in names if name.startswith("score_")]
    if not score_cols:
        raise RuntimeError(f"No score columns found in {score_run / 'scores.bin'}")
    if CONFIG["score_mode"] == "mean" and len(score_cols) != 1:
        raise RuntimeError(
            f"Expected exactly one score column for mean mode, got {len(score_cols)}"
        )

    score_col = score_cols[0]
    written_col = score_col.replace("score_", "written_")
    if written_col in names and not np.asarray(scores[written_col]).all():
        raise RuntimeError("Some rows were not written in score output.")

    return np.asarray(scores[score_col], dtype=np.float32)


def dump_top_bottom(train_dataset, values: np.ndarray, out_path: Path):
    k = min(CONFIG["top_k"], len(train_dataset))
    top_ids = np.argsort(-values)[:k]
    bottom_ids = np.argsort(values)[:k]

    def build_row(rank: int, idx: int, bucket: str):
        row = train_dataset[int(idx)]
        meta = row["meta"]
        return {
            "bucket": bucket,
            "rank": rank,
            "row_id": row["row_id"],
            "score": float(values[idx]),
            "task_id": meta.get("task_id"),
            "outcome": meta.get("outcome"),
            "messages": row["messages"],
        }

    rows = []
    for i, idx in enumerate(top_ids, 1):
        rows.append(build_row(i, int(idx), "top"))
    for i, idx in enumerate(bottom_ids, 1):
        rows.append(build_row(i, int(idx), "bottom"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"saved ranked examples to: {out_path}")
    print("\nTop samples:")
    for row in rows[:k]:
        print(
            f"#{row['rank']:02d} score={row['score']:.6f} "
            f"task_id={row['task_id']} outcome={row['outcome']}"
        )
    print("\nBottom samples:")
    for row in rows[k : 2 * k]:
        print(
            f"#{row['rank']:02d} score={row['score']:.6f} "
            f"task_id={row['task_id']} outcome={row['outcome']}"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Bergson reduce/score currently requires CUDA in this setup.")

    ensure_adapter_has_base_config()
    mode = get_preconditioner_mode()
    projection_dim = get_effective_projection_dim(mode)
    validate_preconditioner_config(mode, projection_dim)
    print_scoring_plan(mode, projection_dim)

    root = Path(CONFIG["output_root"])
    data_root = root / "data"
    train_path = data_root / "train"
    val_path = data_root / "val"
    query_run = root / "query_mean"
    index_preconditioner_run = root / "train_preconditioner"
    score_run = root / "train_scores"
    ranked_path = root / "ranked_examples.json"
    paths_to_cleanup = [
        data_root,
        query_run,
        part_path(query_run),
        score_run,
        part_path(score_run),
        ranked_path,
    ]
    if uses_index_preconditioner(mode):
        paths_to_cleanup.extend(
            [index_preconditioner_run, part_path(index_preconditioner_run)]
        )
    cleanup_paths(paths_to_cleanup)

    train_dataset, train_path, val_path = save_filtered_splits(train_path, val_path)
    run_reduce(
        val_path,
        query_run,
        projection_dim=projection_dim,
        compute_preconditioners=uses_query_preconditioner(mode),
    )
    if uses_index_preconditioner(mode):
        run_reduce(
            train_path,
            index_preconditioner_run,
            projection_dim=projection_dim,
            compute_preconditioners=True,
        )
    run_score(
        train_path,
        query_run,
        score_run,
        projection_dim=projection_dim,
        mode=mode,
        index_preconditioner_run=index_preconditioner_run if uses_index_preconditioner(mode) else None,
    )
    values = get_score_vector(score_run)
    dump_top_bottom(train_dataset, values, ranked_path)


if __name__ == "__main__":
    main()
