from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bergson.data import load_scores
from datasets import load_from_disk


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def split_path(manifest: dict, split_name: str) -> Path:
    split = manifest.get("splits", {}).get(split_name)
    if not split or "path" not in split:
        raise KeyError(f"Split '{split_name}' not found in manifest.")
    return Path(split["path"])


def row_field(row: dict, name: str):
    if name in row and row[name] is not None:
        return row[name]
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta.get(name)
    return None


def make_group_key(row: dict, fields: list[str]) -> str:
    return "||".join(str(row_field(row, f) or "<missing>") for f in fields)


def load_rows(path: Path, max_rows: int | None, group_fields: list[str]) -> list[dict]:
    ds = load_from_disk(str(path))
    limit = len(ds) if max_rows is None else min(len(ds), int(max_rows))
    rows: list[dict] = []
    for i in range(limit):
        row = ds[int(i)]
        rows.append(
            {
                "index": int(i),
                "row_id": str(row.get("row_id", i)),
                "group_key": make_group_key(row, group_fields),
                "game": row_field(row, "game"),
                "game_role": row_field(row, "game_role"),
                "experiment": row_field(row, "experiment"),
                "task_id": row_field(row, "task_id"),
                "outcome": row_field(row, "outcome"),
                "pair_key": row_field(row, "pair_key"),
                "source_split": row_field(row, "source_split"),
            }
        )
    return rows


def load_score_vector(score_run_path: Path) -> np.ndarray:
    scores = load_scores(score_run_path)
    try:
        values = np.asarray(scores.get(slice(None), score_idx=0), dtype=np.float32)
        if values.ndim == 2:
            values = values[:, 0]
    except Exception:
        values = np.asarray(scores[:], dtype=np.float32)
        if values.ndim == 2:
            values = values[:, 0]
    finite = np.isfinite(values)
    if finite.all():
        return values.astype(np.float32)
    fill = float(np.nanmin(values[finite]) - 1.0) if finite.any() else -1.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def align_scores(rows: list[dict], scores: np.ndarray, score_run_path: Path) -> np.ndarray:
    diag_path = score_run_path.parent / "row_diagnostics.jsonl"
    if diag_path.exists():
        score_by_row_id: dict[str, float] = {}
        with diag_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                score_by_row_id[str(rec["row_id"])] = float(rec["score"])
        missing = [row["row_id"] for row in rows if row["row_id"] not in score_by_row_id]
        if missing:
            raise KeyError(f"Missing scores for {len(missing)} row_ids in {diag_path}")
        return np.asarray([score_by_row_id[row["row_id"]] for row in rows], dtype=np.float32)

    max_idx = max(row["index"] for row in rows)
    if max_idx >= len(scores):
        raise ValueError(
            f"Score vector too short for split indices: max_index={max_idx} scores={len(scores)}"
        )
    return np.asarray([scores[row["index"]] for row in rows], dtype=np.float32)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    n = int(values.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks / float(n - 1)


def grouped_split(group_keys: list[str], test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(group_keys):
        groups[str(key)].append(i)
    if len(groups) < 2:
        return random_split(len(group_keys), test_fraction, seed)

    rng = np.random.default_rng(seed)
    ids = list(groups.keys())
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_fraction))
    n_test = max(1, min(n_test, len(ids) - 1))
    test_groups = set(ids[:n_test])

    train_idx = np.asarray([i for i, k in enumerate(group_keys) if k not in test_groups], dtype=np.int64)
    test_idx = np.asarray([i for i, k in enumerate(group_keys) if k in test_groups], dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        return random_split(len(group_keys), test_fraction, seed)
    return train_idx, test_idx


def random_split(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 rows to split.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_test = int(round(n * test_fraction))
    n_test = max(1, min(n_test, n - 1))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return safe_corr(rank_percentile(x), rank_percentile(y))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err * err))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "pearson": safe_corr(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
    }


def topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    n = int(y_true.shape[0])
    k = max(1, min(int(k), n))
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    overlap = len(true_top & pred_top)
    return {"k": int(k), "overlap_count": int(overlap), "overlap_fraction": float(overlap / k)}


def project_features(x: np.ndarray, projection_dim: int, seed: int) -> np.ndarray:
    if projection_dim <= 0 or x.shape[1] <= projection_dim:
        return x.astype(np.float32)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((x.shape[1], projection_dim), dtype=np.float32) / np.sqrt(
        float(projection_dim)
    )
    return (x @ proj).astype(np.float32)


def build_sae_matrix(
    sae_dir: Path,
    rows: list[dict],
    projection_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    examples_path = sae_dir / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing SAE metadata file: {examples_path}")
    npz_files = sorted(sae_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz SAE file found in {sae_dir}")
    npz_path = npz_files[0]
    data = np.load(npz_path)

    row_ids: list[str] = []
    with examples_path.open() as f:
        for line in f:
            if line.strip():
                row_ids.append(str(json.loads(line)["row_id"]))
    row_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    if "sample_stats" not in data:
        raise KeyError("SAE npz missing 'sample_stats'")
    if "feature_activation_mean" not in data:
        raise KeyError(
            "SAE npz missing 'feature_activation_mean'. Re-run sae_analysis.py with activation export."
        )

    stats = np.asarray(data["sample_stats"], dtype=np.float32)
    activation = np.asarray(data["feature_activation_mean"], dtype=np.float32)
    activation = project_features(activation, projection_dim, seed + 17)
    full = np.concatenate([stats, activation], axis=1)
    dims = {
        "sample_stats": int(stats.shape[1]),
        "feature_activation_mean_projected": int(activation.shape[1]),
    }
    missing = [row["row_id"] for row in rows if row["row_id"] not in row_to_idx]
    if missing:
        raise KeyError(f"Missing {len(missing)} row_ids in SAE artifacts.")
    aligned = np.asarray([full[row_to_idx[row["row_id"]]] for row in rows], dtype=np.float32)

    meta = {
        "sae_dir": str(sae_dir),
        "sae_npz": str(npz_path),
        "sae_feature_mode": "stats+activation (locked)",
        "sae_projection_dim": int(projection_dim),
        "sae_feature_dims": dims,
        "sae_total_dim": int(aligned.shape[1]),
    }
    return aligned, meta


@dataclass
class RidgeModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.mean) / self.std
        xb = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float64)], axis=1)
        return np.asarray(xb @ self.coef, dtype=np.float32)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, l2: float) -> RidgeModel:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("Ridge regression received an empty training set.")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xn = (x - mean) / std
    xb = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float64)], axis=1)

    reg = np.eye(xb.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    lhs = xb.T @ xb + reg
    rhs = xb.T @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return RidgeModel(mean=mean, std=std, coef=coef)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact SAE-only attribution score predictor.")
    parser.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    parser.add_argument("--split", type=str, default="score_pool")
    parser.add_argument(
        "--score-run-path",
        type=str,
        default="runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/train_scores",
    )
    parser.add_argument("--output-root", type=str, default="runs/simple_wordguesser_v1/sae_rank_predictor")

    parser.add_argument("--target-type", choices=["score", "rank"], default="rank")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--split-mode", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--group-key-fields", type=str, default="game,game_role,experiment,task_id")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sae-dir", type=str, required=True)
    parser.add_argument("--sae-projection-dim", type=int, default=512)
    parser.add_argument("--sae-ridge-l2", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    ds_path = split_path(manifest, args.split)
    group_fields = [x.strip() for x in args.group_key_fields.split(",") if x.strip()]
    rows = load_rows(ds_path, args.max_rows, group_fields)
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows for train/test split.")

    scores = load_score_vector(Path(args.score_run_path))
    y_raw = align_scores(rows, scores, Path(args.score_run_path))
    y = y_raw if args.target_type == "score" else rank_percentile(y_raw)

    group_keys = [row["group_key"] for row in rows]
    if args.split_mode == "grouped":
        train_idx, test_idx = grouped_split(group_keys, args.test_fraction, args.seed)
    else:
        train_idx, test_idx = random_split(len(rows), args.test_fraction, args.seed)

    x, sae_meta = build_sae_matrix(
        sae_dir=Path(args.sae_dir),
        rows=rows,
        projection_dim=args.sae_projection_dim,
        seed=args.seed,
    )
    model = fit_ridge(x[train_idx], y[train_idx], l2=args.sae_ridge_l2)
    y_pred = model.predict(x)

    train_metrics = regression_metrics(y[train_idx], y_pred[train_idx])
    test_metrics = regression_metrics(y[test_idx], y_pred[test_idx])
    full_metrics = regression_metrics(y, y_pred)
    overlap = topk_overlap(y, y_pred, args.top_k)

    row_scores_path = out / "row_scores.jsonl"
    rank_order = np.argsort(-y_pred)
    with row_scores_path.open("w") as f:
        for rank, i in enumerate(rank_order, start=1):
            row = rows[int(i)]
            rec = {
                "rank": int(rank),
                "index": int(row["index"]),
                "row_id": row["row_id"],
                "target_score_raw": float(y_raw[int(i)]),
                "target_value": float(y[int(i)]),
                "pred_value": float(y_pred[int(i)]),
                "abs_error": float(abs(y_pred[int(i)] - y[int(i)])),
                "game": row.get("game"),
                "game_role": row.get("game_role"),
                "experiment": row.get("experiment"),
                "task_id": row.get("task_id"),
                "outcome": row.get("outcome"),
                "pair_key": row.get("pair_key"),
                "source_split": row.get("source_split"),
            }
            f.write(json.dumps(rec) + "\n")

    summary = {
        "manifest": args.manifest,
        "split": args.split,
        "dataset_path": str(ds_path),
        "score_run_path": args.score_run_path,
        "mode": "sae",
        "target_type": args.target_type,
        "n_rows": int(len(rows)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
        "topk_overlap_with_target": overlap,
        "row_scores_path": str(row_scores_path),
        "sae_meta": sae_meta,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved outputs to: {out}")
    print(
        f"test pearson={test_metrics['pearson']} "
        f"spearman={test_metrics['spearman']} "
        f"rmse={test_metrics['rmse']:.6f}"
    )
    print(
        f"top-k overlap with target ranking: {overlap['overlap_count']}/{overlap['k']} "
        f"({overlap['overlap_fraction']:.4f})"
    )
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
