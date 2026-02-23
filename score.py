from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

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

import numpy as np
from bergson.build import build
from bergson.config import DataConfig, IndexConfig, ReduceConfig, ScoreConfig
from bergson.data import load_scores
from bergson.reduce import reduce
from bergson.score.score import score_dataset
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer


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


def build_finetune_aligned_dataset(
    src_path: Path,
    dst_path: Path,
    tokenizer_name: str,
    max_length: int,
) -> Path:
    ds = load_from_disk(str(src_path))
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eos = tok.eos_token or ""

    def to_features(row: dict) -> dict:
        prompt = str(row["prompt"])
        completion = str(row["completion"])
        if eos and not completion.endswith(eos):
            completion = completion + eos

        prompt_ids = tok(prompt)["input_ids"]
        full_ids = tok(prompt + completion)["input_ids"][:max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        return {
            "input_ids": full_ids,
            "labels": labels,
            "length": len(full_ids),
        }

    out = ds.map(to_features, remove_columns=ds.column_names)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        remove_path(dst_path)
    out.save_to_disk(str(dst_path))
    return dst_path


def make_index_config(
    run_path: Path,
    data_path: Path,
    args: argparse.Namespace,
    skip_preconditioners: bool,
) -> IndexConfig:
    model_path = args.adapter_path or resolve_model_path(args.base_model)
    # Use IT model for tokenizer (same vocab, has chat template); fall back to base.
    it_model = args.base_model if args.base_model.endswith("-Instruct") \
        else args.base_model + "-Instruct"
    tokenizer_path = args.adapter_path or resolve_model_path(it_model)
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

    rows = []
    for i in range(len(pool_ds)):
        rows.append(
            {
                "index": int(i),
                "score": float(scores[i]),
                "magpie_score": float(magpie_scores[i]),
                "length_tokens": int(lengths[i]),
                "loss": float(losses[i]),
            }
        )

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
            "min": float(np.min(scores)) if len(scores) else 0.0,
            "max": float(np.max(scores)) if len(scores) else 0.0,
            "mean": float(np.mean(scores)) if len(scores) else 0.0,
        },
        "correlations": {
            "score_vs_length": safe_corr(scores, lengths),
            "score_vs_loss": safe_corr(scores, losses),
            "score_vs_magpie": safe_corr(scores, magpie),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))


def compute_length_bins(lengths: np.ndarray, bins: int = 5) -> np.ndarray:
    edges = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.zeros_like(lengths, dtype=np.int32)
    return np.searchsorted(edges[1:-1], lengths, side="right").astype(np.int32)


def build_subsets(
    pool_ds: Dataset,
    score_data_ds: Dataset,
    scores: np.ndarray,
    out_dir: Path,
    k: int,
    seed: int,
    allow_random_overlap: bool,
) -> dict:
    n = len(pool_ds)
    k = max(1, min(k, n))

    ranked = np.argsort(-scores)
    top_idx = ranked[:k].astype(np.int32).tolist()
    top_set = set(top_idx)
    available = list(range(n)) if allow_random_overlap else [i for i in range(n) if i not in top_set]

    rng = np.random.default_rng(seed)
    lengths = (
        np.asarray(score_data_ds["length"], dtype=np.float32)
        if "length" in score_data_ds.column_names
        else np.zeros(n, dtype=np.float32)
    )
    length_bins = compute_length_bins(lengths)

    # Stratify random baseline by length bin only (seq_len is the dominant
    # confound in attribution scores, so we control for it explicitly).
    from collections import Counter
    top_strata = Counter(int(length_bins[i]) for i in top_idx)
    remaining = set(available)
    random_idx: list[int] = []

    for length_bin, target_count in top_strata.items():
        bin_pool = [i for i in remaining if int(length_bins[i]) == length_bin]
        take = min(target_count, len(bin_pool))
        if take > 0:
            chosen = rng.choice(bin_pool, size=take, replace=False).tolist()
            random_idx.extend(int(i) for i in chosen)
            for i in chosen:
                remaining.remove(int(i))

    # Fill any remaining slots uniformly at random
    if len(random_idx) < k:
        pool = list(remaining)
        need = min(k - len(random_idx), len(pool))
        if need > 0:
            chosen = rng.choice(pool, size=need, replace=False).tolist()
            random_idx.extend(int(i) for i in chosen)

    if len(random_idx) > k:
        random_idx = random_idx[:k]

    top_dir = out_dir / "top_k"
    rand_dir = out_dir / "matched_random_k"
    for p in (top_dir, rand_dir):
        if p.exists():
            remove_path(p)

    pool_ds.select(sorted(top_idx)).save_to_disk(str(top_dir))
    pool_ds.select(sorted(random_idx)).save_to_disk(str(rand_dir))

    top_tokens = int(lengths[top_idx].sum()) if len(lengths) else 0
    rand_tokens = int(lengths[random_idx].sum()) if len(lengths) else 0
    overlap_rows = int(len(set(top_idx).intersection(set(random_idx))))

    payload = {
        "k": int(k),
        "seed": int(seed),
        "allow_random_overlap": bool(allow_random_overlap),
        "arms": {
            "top_k": {
                "path": str(top_dir),
                "rows": int(len(top_idx)),
                "token_count": top_tokens,
            },
            "matched_random_k": {
                "path": str(rand_dir),
                "rows": int(len(random_idx)),
                "token_count": rand_tokens,
            },
        },
        "token_budget_delta": abs(top_tokens - rand_tokens),
        "overlap_rows": overlap_rows,
    }

    (out_dir / "subset_manifest.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Bergson scoring pipeline.")
    parser.add_argument("--manifest", type=str, default="runs/smoltalk_v4/manifest.json")
    parser.add_argument("--pool-split", type=str, default="score_pool")
    parser.add_argument("--query-split", type=str, default="attr_query")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="PEFT adapter path. If omitted, --base-model is used directly.")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--output-dir", type=str, default="runs/smoltalk_v4/scores_math_da")
    parser.add_argument("--token-batch-size", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--preconditioning-mode", choices=["none", "query", "mixed"], default="query")
    parser.add_argument(
        "--tokenization-mode",
        choices=["bergson_chat", "finetune_raw"],
        default="bergson_chat",
    )
    parser.add_argument("--mixing-coefficient", type=float, default=0.99)
    parser.add_argument("--score-mode", choices=["mean", "nearest"], default="mean")
    parser.add_argument("--unit-normalize", dest="unit_normalize", action="store_true")
    parser.add_argument("--no-unit-normalize", dest="unit_normalize", action="store_false")
    parser.add_argument("--loss-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--subset-k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-random-overlap", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        default=False,
        help="Skip cleanup and gradient/score computation; re-run diagnostics/subsets on existing scores.",
    )
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
    subsets_dir = out / "subsets"
    pretokenized_root = out / "_pretokenized_finetune"

    if not args.postprocess_only:
        to_clean = [
            query_run,
            part_path(query_run),
            score_run,
            part_path(score_run),
            diagnostics_path,
            summary_path,
            subsets_dir,
        ]
        if args.preconditioning_mode == "mixed":
            to_clean.extend([pool_pre, part_path(pool_pre)])
        if args.tokenization_mode == "finetune_raw":
            to_clean.append(pretokenized_root)
        for p in to_clean:
            if p.exists():
                remove_path(p)

        # Resolve the effective model path: adapter if given, else the base model itself.
        # Overwrite args.adapter_path so all downstream calls (make_index_config etc.) see it.
        if args.adapter_path:
            ensure_adapter_config(Path(args.adapter_path), args.base_model)
        else:
            # No adapter: point directly at the cached base model snapshot.
            # Do NOT call ensure_adapter_config — it would create a local stub directory
            # with only config.json and no tokenizer files, causing tokenizer load failures.
            args.adapter_path = resolve_model_path(args.base_model)

        pool_path = pool_source
        query_path = query_source
        if args.tokenization_mode == "finetune_raw":
            tokenizer_name = args.adapter_path if Path(args.adapter_path).exists() else args.base_model
            pool_path = build_finetune_aligned_dataset(
                src_path=pool_source,
                dst_path=pretokenized_root / args.pool_split,
                tokenizer_name=tokenizer_name,
                max_length=args.token_batch_size,
            )
            query_path = build_finetune_aligned_dataset(
                src_path=query_source,
                dst_path=pretokenized_root / args.query_split,
                tokenizer_name=tokenizer_name,
                max_length=args.token_batch_size,
            )

        use_query_pre = args.preconditioning_mode in {"query", "mixed"}
        if args.score_mode == "nearest":
            run_build(query_path, query_run, args, compute_preconditioners=use_query_pre)
        else:
            run_reduce(query_path, query_run, args, compute_preconditioners=use_query_pre)

        if args.preconditioning_mode == "mixed":
            run_reduce(pool_path, pool_pre, args, compute_preconditioners=True)

        run_score(
            pool_path=pool_path,
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
    subsets_dir.mkdir(parents=True, exist_ok=True)
    subset_payload = build_subsets(
        pool_ds=pool_ds,
        score_data_ds=score_data_ds,
        scores=scores,
        out_dir=subsets_dir,
        k=args.subset_k,
        seed=args.seed,
        allow_random_overlap=args.allow_random_overlap,
    )

    continuation_manifest = dict(manifest)
    continuation_splits = dict(continuation_manifest.get("splits", {}))
    continuation_splits["top_k"] = subset_payload["arms"]["top_k"]
    continuation_splits["matched_random_k"] = subset_payload["arms"]["matched_random_k"]
    continuation_manifest["splits"] = continuation_splits

    continuation_path = out / "continuation_manifest.json"
    continuation_path.write_text(json.dumps(continuation_manifest, indent=2))

    print(f"row diagnostics: {diagnostics_path}")
    print(f"summary:         {summary_path}")
    print(f"subset manifest: {subsets_dir / 'subset_manifest.json'}")
    print(f"continue manifest: {continuation_path}")


if __name__ == "__main__":
    main()
