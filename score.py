from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from bergson.build import build
from bergson.config import DataConfig, IndexConfig, ReduceConfig, ScoreConfig
from bergson.data import load_scores
from bergson.reduce import reduce
from bergson.score.score import score_dataset
from datasets import Dataset, load_from_disk
from transformers import AutoConfig

from pipeline_common import (
    DEFAULT_BASE_MODEL,
    ensure_hf_home_env,
    infer_instruct_tokenizer_model,
)

ensure_hf_home_env()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def part_path(path: Path) -> Path:
    return Path(str(path) + ".part")


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def split_path(manifest: dict, split_name: str) -> Path:
    return Path(manifest["splits"][split_name]["path"])


def ensure_adapter_config(adapter_path: Path, base_model: str) -> None:
    config_path = adapter_path / "config.json"
    if not config_path.exists():
        AutoConfig.from_pretrained(base_model).save_pretrained(adapter_path)


def make_index_config(
    run_path: Path,
    data_path: Path,
    args: argparse.Namespace,
    skip_preconditioners: bool,
) -> IndexConfig:
    model_path = args.adapter_path or args.base_model
    # Use IT model for tokenizer (same vocab, has chat template); fall back to base.
    it_model = infer_instruct_tokenizer_model(args.base_model)
    tokenizer_path = args.adapter_path or it_model
    return IndexConfig(
        run_path=str(run_path),
        model=model_path,
        tokenizer=tokenizer_path,
        projection_dim=args.projection_dim,
        token_batch_size=args.token_batch_size,
        skip_preconditioners=skip_preconditioners,
        loss_reduction=args.loss_reduction,
        label_smoothing=args.label_smoothing,
        data=DataConfig(
            dataset=str(data_path),
            split="train",
            prompt_column="prompt",
            completion_column="completion",
            truncation=True,
        ),
        distributed={"nproc_per_node": 1},
    )


def run_reduce(
    data_path: Path,
    run_path: Path,
    args: argparse.Namespace,
    compute_preconditioners: bool,
) -> None:
    reduce(
        make_index_config(
            run_path=run_path,
            data_path=data_path,
            args=args,
            skip_preconditioners=not compute_preconditioners,
        ),
        ReduceConfig(method="mean", unit_normalize=args.unit_normalize),
    )


def run_build(
    data_path: Path,
    run_path: Path,
    args: argparse.Namespace,
    compute_preconditioners: bool,
) -> None:
    build(
        make_index_config(
            run_path=run_path,
            data_path=data_path,
            args=args,
            skip_preconditioners=not compute_preconditioners,
        )
    )


def run_score(
    pool_path: Path,
    query_run: Path,
    pool_preconditioner_run: Path | None,
    score_run: Path,
    args: argparse.Namespace,
) -> None:
    bergson_mode = "individual" if args.score_mode == "nearest" else args.score_mode

    if args.preconditioning_mode == "none":
        query_preconditioner_path = None
        index_preconditioner_path = None
    elif args.preconditioning_mode == "query":
        query_preconditioner_path = str(query_run)
        index_preconditioner_path = None
    else:
        query_preconditioner_path = str(query_run)
        index_preconditioner_path = (
            str(pool_preconditioner_run) if pool_preconditioner_run else None
        )

    score_dataset(
        make_index_config(
            run_path=score_run,
            data_path=pool_path,
            args=args,
            skip_preconditioners=True,
        ),
        ScoreConfig(
            query_path=str(query_run),
            score=bergson_mode,
            unit_normalize=args.unit_normalize,
            query_preconditioner_path=query_preconditioner_path,
            index_preconditioner_path=index_preconditioner_path,
            mixing_coefficient=args.mixing_coefficient,
        ),
    )


def get_score_vector(score_run: Path, score_mode: str) -> np.ndarray:
    scores = load_scores(score_run)
    raw = scores[:]
    # Bergson stores a structured array with fields like score_0 (bfloat16), written_0 (bool).
    # Extract the first score field as float32.
    if raw.dtype.names is not None:
        score_fields = [n for n in raw.dtype.names if n.startswith("score_")]
        matrix = raw[score_fields[0]].astype(np.float32)
    else:
        matrix = np.asarray(raw, dtype=np.float32)

    if score_mode == "nearest":
        values = matrix if matrix.ndim == 1 else matrix.max(axis=1)
    else:
        values = matrix if matrix.ndim == 1 else matrix[:, 0]

    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_row_diagnostics(
    pool_ds: Dataset,
    score_data_ds: Dataset,
    scores: np.ndarray,
    out_path: Path,
) -> list[dict]:
    lengths = (
        np.asarray(score_data_ds["length"], dtype=np.int32)
        if "length" in score_data_ds.column_names
        else np.zeros(len(pool_ds), dtype=np.int32)
    )
    losses = (
        np.asarray(score_data_ds["loss"], dtype=np.float32)
        if "loss" in score_data_ds.column_names
        else np.zeros(len(pool_ds), dtype=np.float32)
    )
    magpie_scores = (
        np.asarray(pool_ds["magpie_score"], dtype=np.float32)
        if "magpie_score" in pool_ds.column_names
        else np.zeros(len(pool_ds), dtype=np.float32)
    )

    rows = [
        {
            "index": int(i),
            "score": float(scores[i]),
            "magpie_score": float(magpie_scores[i]),
            "length_tokens": int(lengths[i]),
            "loss": float(losses[i]),
        }
        for i in range(len(pool_ds))
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    scores = np.asarray([r["score"] for r in rows], dtype=np.float32)
    lengths = np.asarray([r["length_tokens"] for r in rows], dtype=np.float32)
    losses = np.asarray([r["loss"] for r in rows], dtype=np.float32)
    magpie = np.asarray([r["magpie_score"] for r in rows], dtype=np.float32)

    summary = {
        "rows": len(rows),
        "finite_score_coverage": {
            "finite": int(np.isfinite(scores).sum()),
            "total": int(len(scores)),
        },
        "score_stats": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
        },
        "correlations": {
            "score_vs_length": safe_corr(scores, lengths),
            "score_vs_loss": safe_corr(scores, losses),
            "score_vs_magpie": safe_corr(scores, magpie),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Bergson scoring pipeline.")
    parser.add_argument("--manifest", type=str, default="runs/smoltalk_v4/manifest.json")
    parser.add_argument("--pool-split", type=str, default="score_pool")
    parser.add_argument("--query-split", type=str, default="attr_query")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="PEFT adapter path. If omitted, --base-model is used directly.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default="runs/smoltalk_v4/scores_math_da")
    parser.add_argument("--token-batch-size", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--preconditioning-mode", choices=["none", "query", "mixed"], default="query")
    parser.add_argument("--mixing-coefficient", type=float, default=0.99)
    parser.add_argument("--score-mode", choices=["mean", "nearest"], default="mean")
    parser.add_argument("--unit-normalize", dest="unit_normalize", action="store_true")
    parser.add_argument("--no-unit-normalize", dest="unit_normalize", action="store_false")
    parser.add_argument("--loss-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.set_defaults(unit_normalize=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    pool_source = split_path(manifest, args.pool_split)
    query_source = split_path(manifest, args.query_split)

    out = Path(args.output_dir)
    query_run = out / "query_index"
    pool_pre = out / "pool_preconditioner"
    score_run = out / "train_scores"
    diagnostics_path = out / "row_diagnostics.jsonl"
    summary_path = out / "summary.json"
    to_clean = [
        query_run,
        part_path(query_run),
        score_run,
        part_path(score_run),
        diagnostics_path,
        summary_path,
    ]
    if args.preconditioning_mode == "mixed":
        to_clean.extend([pool_pre, part_path(pool_pre)])
    for p in to_clean:
        if p.exists():
            remove_path(p)

    # Resolve the effective model handle: adapter if given, else the base model ID.
    # Overwrite args.adapter_path so downstream config uses one value consistently.
    if args.adapter_path:
        ensure_adapter_config(Path(args.adapter_path), args.base_model)
    else:
        # No adapter: use the base model ID directly.
        # Do NOT call ensure_adapter_config — it would create a local stub directory
        # with only config.json and no tokenizer files, causing tokenizer load failures.
        args.adapter_path = args.base_model

    use_query_pre = args.preconditioning_mode in {"query", "mixed"}
    if args.score_mode == "nearest":
        run_build(query_source, query_run, args, compute_preconditioners=use_query_pre)
    else:
        run_reduce(query_source, query_run, args, compute_preconditioners=use_query_pre)

    if args.preconditioning_mode == "mixed":
        run_reduce(pool_source, pool_pre, args, compute_preconditioners=True)

    run_score(
        pool_path=pool_source,
        query_run=query_run,
        pool_preconditioner_run=pool_pre if args.preconditioning_mode == "mixed" else None,
        score_run=score_run,
        args=args,
    )

    scores = get_score_vector(score_run, args.score_mode)
    pool_ds = load_from_disk(str(pool_source))
    score_data_ds = load_from_disk(str(score_run / "data.hf"))

    rows = write_row_diagnostics(pool_ds, score_data_ds, scores, diagnostics_path)
    write_summary(rows, summary_path)

    print(f"row diagnostics: {diagnostics_path}")
    print(f"summary:         {summary_path}")


if __name__ == "__main__":
    main()
