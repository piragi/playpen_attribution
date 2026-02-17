from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def keep_row(example: dict, game: str, role: str) -> bool:
    meta = example.get("meta") or {}
    return (
        meta.get("game") == game
        and meta.get("game_role") == role
        and meta.get("outcome") != "aborted"
    )


def to_prompt_completion(messages: list[dict]) -> tuple[str | None, str | None]:
    last_assistant = None
    for i, msg in enumerate(messages):
        if str(msg.get("role")) == "assistant":
            last_assistant = i

    if last_assistant is None:
        return None, None

    completion = str(messages[last_assistant].get("content", "")).strip()
    if not completion:
        return None, None

    prompt = "\n\n".join(
        f"{str(msg.get('role', 'user')).upper()}: {str(msg.get('content', ''))}"
        for msg in messages[:last_assistant]
    ).strip()
    return prompt, completion


def flatten_split(ds: Dataset, source_split: str) -> Dataset:
    def row_fn(example: dict, idx: int) -> dict:
        messages = list(example.get("messages") or [])
        prompt, completion = to_prompt_completion(messages)
        if completion is None:
            return {"keep": False}

        meta = example.get("meta") or {}
        experiment = str(meta.get("experiment", ""))
        task_id = str(meta.get("task_id", ""))
        game = str(meta.get("game", ""))
        role = str(meta.get("game_role", ""))

        return {
            "keep": True,
            "row_id": f"{source_split}-{idx}",
            "prompt": prompt,
            "completion": completion,
            "source_split": source_split,
            "game": game,
            "game_role": role,
            "pair_key": f"{game}::{role}",
            "outcome": str(meta.get("outcome", "")),
            "experiment": experiment,
            "task_id": task_id,
            "group_id": f"{experiment}::{task_id}",
        }

    out = ds.map(row_fn, with_indices=True, remove_columns=ds.column_names)
    out = out.filter(lambda row: bool(row["keep"]))
    return out.remove_columns("keep")


def split_query_eval(ds: Dataset, query_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, gid in enumerate(ds["group_id"]):
        groups[str(gid)].append(i)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    if len(group_ids) >= 2:
        n_query = int(round(len(group_ids) * query_fraction))
        n_query = max(1, min(n_query, len(group_ids) - 1))
        query_groups = set(group_ids[:n_query])
        query_idx = [i for gid in query_groups for i in groups[gid]]
        eval_idx = [i for gid in group_ids[n_query:] for i in groups[gid]]
        return ds.select(sorted(query_idx)), ds.select(sorted(eval_idx))

    idx = list(range(len(ds)))
    rng.shuffle(idx)
    n_query = int(round(len(ds) * query_fraction))
    n_query = max(1, min(n_query, max(1, len(ds) - 1)))
    query_idx = sorted(idx[:n_query])
    eval_idx = sorted(idx[n_query:])
    if not eval_idx and idx:
        eval_idx = [idx[-1]]
    return ds.select(query_idx), ds.select(eval_idx)


def build(args: argparse.Namespace) -> None:
    raw = load_dataset(args.dataset_name, args.dataset_config)

    train_raw = raw["train"]
    validation_raw = raw["validation"]

    train_filtered = train_raw.filter(lambda ex: keep_row(ex, args.game, args.role))
    validation_filtered = validation_raw.filter(
        lambda ex: keep_row(ex, args.game, args.role)
    )

    train_base = flatten_split(train_filtered, "train")
    validation_flat = flatten_split(validation_filtered, "validation")
    attr_query, eval_ds = split_query_eval(
        validation_flat, query_fraction=args.query_fraction, seed=args.seed
    )

    run_dir = Path(args.run_dir)
    data_dir = run_dir / "data"
    train_path = data_dir / "train_base"
    query_path = data_dir / "attr_query"
    eval_path = data_dir / "eval"
    manifest_path = run_dir / "manifest.json"

    for path in (train_path, query_path, eval_path, manifest_path):
        if path.exists():
            remove_path(path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_base.save_to_disk(str(train_path))
    attr_query.save_to_disk(str(query_path))
    eval_ds.save_to_disk(str(eval_path))

    manifest = {
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "game": args.game,
            "role": args.role,
        },
        "split_policy": {
            "train_base": "filtered train split",
            "attr_query": f"{args.query_fraction:.3f} of validation groups",
            "eval": "remaining validation groups",
            "group_key": "experiment::task_id",
            "seed": args.seed,
        },
        "splits": {
            "train_base": {"path": str(train_path), "rows": len(train_base)},
            "score_pool": {"path": str(train_path), "rows": len(train_base)},
            "attr_query": {"path": str(query_path), "rows": len(attr_query)},
            "eval": {"path": str(eval_path), "rows": len(eval_ds)},
        },
        "filter_stats": {
            "train_raw_rows": len(train_raw),
            "validation_raw_rows": len(validation_raw),
            "train_pair_rows": len(train_base),
            "validation_pair_rows": len(validation_flat),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {manifest_path}")
    print(f"train_base rows: {len(train_base)}")
    print(f"attr_query rows: {len(attr_query)}")
    print(f"eval rows:      {len(eval_ds)}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build minimal Playpen manifest/datasets.")
    parser.add_argument("--run-dir", type=str, default="runs/simple_wordguesser_v1")
    parser.add_argument("--dataset-name", type=str, default="colab-potsdam/playpen-data")
    parser.add_argument("--dataset-config", type=str, default="interactions")
    parser.add_argument("--game", type=str, default="taboo")
    parser.add_argument("--role", type=str, default="WordGuesser")
    parser.add_argument("--query-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    build(args)


if __name__ == "__main__":
    main()
