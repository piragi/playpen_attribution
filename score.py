from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from bergson.build import build
from bergson.config import DataConfig, IndexConfig, ReduceConfig, ScoreConfig
from bergson.data import load_scores
from bergson.reduce import reduce
from bergson.score.score import score_dataset
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoConfig


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
    return IndexConfig(
        run_path=str(run_path),
        model=args.adapter_path,
        tokenizer=args.adapter_path,
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
    if score_mode == "nearest":
        matrix = np.asarray(scores[:], dtype=np.float32)
        values = matrix if matrix.ndim == 1 else matrix.max(axis=1)
    else:
        values = np.asarray(scores.get(slice(None), score_idx=0), dtype=np.float32)

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

    rows = []
    for i in range(len(pool_ds)):
        row = pool_ds[int(i)]
        rows.append(
            {
                "index": int(i),
                "row_id": str(row.get("row_id", i)),
                "score": float(scores[i]),
                "pair_key": str(row.get("pair_key", "")),
                "outcome": str(row.get("outcome", "")),
                "length_tokens": int(lengths[i]),
                "loss": float(losses[i]),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def topk_counts(rows: list[dict], key: str, k: int) -> dict[str, int]:
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)[: min(k, len(rows))]
    counts = Counter(str(row.get(key, "")) for row in ranked)
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def write_summary(rows: list[dict], out_path: Path) -> None:
    scores = np.asarray([r["score"] for r in rows], dtype=np.float32)
    lengths = np.asarray([r["length_tokens"] for r in rows], dtype=np.float32)
    losses = np.asarray([r["loss"] for r in rows], dtype=np.float32)

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
        },
        "top_k_composition": {
            "k50_pair_key": topk_counts(rows, "pair_key", 50),
            "k100_pair_key": topk_counts(rows, "pair_key", 100),
            "k200_pair_key": topk_counts(rows, "pair_key", 200),
            "k50_outcome": topk_counts(rows, "outcome", 50),
            "k100_outcome": topk_counts(rows, "outcome", 100),
            "k200_outcome": topk_counts(rows, "outcome", 200),
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
    outcomes = [str(x) for x in pool_ds["outcome"]]

    top_strata = Counter((outcomes[i], int(length_bins[i])) for i in top_idx)
    remaining = set(available)
    random_idx: list[int] = []

    for (outcome, length_bin), target_count in top_strata.items():
        exact_pool = [
            i
            for i in remaining
            if outcomes[i] == outcome and int(length_bins[i]) == int(length_bin)
        ]
        exact_take = min(target_count, len(exact_pool))
        if exact_take > 0:
            chosen = rng.choice(exact_pool, size=exact_take, replace=False).tolist()
            random_idx.extend(int(i) for i in chosen)
            for i in chosen:
                remaining.remove(int(i))

        if exact_take < target_count:
            outcome_pool = [i for i in remaining if outcomes[i] == outcome]
            need = target_count - exact_take
            take = min(need, len(outcome_pool))
            if take > 0:
                chosen = rng.choice(outcome_pool, size=take, replace=False).tolist()
                random_idx.extend(int(i) for i in chosen)
                for i in chosen:
                    remaining.remove(int(i))

    if len(random_idx) < k:
        need = k - len(random_idx)
        pool = list(remaining)
        if need > 0 and pool:
            chosen = rng.choice(pool, size=min(need, len(pool)), replace=False).tolist()
            random_idx.extend(int(i) for i in chosen)
            for i in chosen:
                remaining.remove(int(i))

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
    parser.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    parser.add_argument("--pool-split", type=str, default="score_pool")
    parser.add_argument("--query-split", type=str, default="attr_query")
    parser.add_argument("--adapter-path", type=str, default="runs/simple_wordguesser_v1/base_adapter")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--output-dir", type=str, default="runs/simple_wordguesser_v1/attribution")
    parser.add_argument("--token-batch-size", type=int, default=1024)
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

    ensure_adapter_config(Path(args.adapter_path), args.base_model)

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
