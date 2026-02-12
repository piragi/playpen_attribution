import json
import re
from pathlib import Path

import numpy as np
import torch
from bergson.data import load_scores


CONFIG = {
    "score_run_path": "./runs/taboo_attr/train_scores",
    "sae_samples_dir": "./runs/taboo_attr/sae_samples_layer17_all_train",
    "sae_id": "layer_17_width_16k_l0_small",
    "output_path": "./runs/taboo_attr/sae_contrast_layer17.json",
    "top_k": 100,
    "seed": 42,
    "test_fraction": 0.2,
    "lr": 0.05,
    "weight_decay": 0.001,
    "steps": 1500,
    "top_feature_count": 50,
}


def safe_name(name: str):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def get_score_vector(score_run_path: Path):
    scores = load_scores(score_run_path)
    names = list(scores.dtype.names or [])
    score_cols = sorted(name for name in names if name.startswith("score_"))
    if not score_cols:
        raise RuntimeError(f"No score columns found in {score_run_path / 'scores.bin'}")

    score_col = score_cols[0]
    values = np.asarray(scores[score_col], dtype=np.float32)
    written_col = score_col.replace("score_", "written_")
    if written_col in names and not np.asarray(scores[written_col]).all():
        raise RuntimeError("Some rows were not written in score output.")

    return values, score_col


def load_precomputed_samples(samples_dir: Path, sae_id: str):
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory: {samples_dir}")

    npz_path = samples_dir / f"{safe_name(sae_id)}.npz"
    metadata_path = samples_dir / "examples.jsonl"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing precomputed npz: {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    npz = np.load(npz_path)
    required = {"feature_presence", "sample_stats", "stat_feature_names"}
    missing = required - set(npz.files)
    if missing:
        raise RuntimeError(f"Missing keys in {npz_path}: {sorted(missing)}")

    feature_presence = np.asarray(npz["feature_presence"], dtype=np.bool_)
    sample_stats = np.asarray(npz["sample_stats"], dtype=np.float32)
    stat_names = [str(x) for x in npz["stat_feature_names"].tolist()]

    metadata = []
    with metadata_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    if len(metadata) != sample_stats.shape[0]:
        raise RuntimeError(
            "Metadata row count does not match sample array rows: "
            f"metadata={len(metadata)} sample_stats={sample_stats.shape[0]}"
        )

    if feature_presence.shape[0] != sample_stats.shape[0]:
        raise RuntimeError(
            "feature_presence rows do not match sample_stats rows: "
            f"{feature_presence.shape[0]} vs {sample_stats.shape[0]}"
        )

    return {
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "feature_presence": feature_presence,
        "sample_stats": sample_stats,
        "stat_names": stat_names,
        "metadata": metadata,
    }


def top_k_labels(scores: np.ndarray, top_k: int):
    n = len(scores)
    k = min(max(1, int(top_k)), n)
    top_ids = np.argsort(-scores)[:k]
    y = np.zeros(n, dtype=np.int64)
    y[top_ids] = 1
    return y, top_ids


def stratified_split(y: np.ndarray, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both positive and negative labels for regression.")

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_test = max(1, int(round(len(pos) * test_fraction)))
    n_neg_test = max(1, int(round(len(neg) * test_fraction)))
    n_pos_test = min(n_pos_test, len(pos) - 1) if len(pos) > 1 else 0
    n_neg_test = min(n_neg_test, len(neg) - 1) if len(neg) > 1 else 0

    test_idx = np.concatenate([pos[:n_pos_test], neg[:n_neg_test]])
    train_idx = np.concatenate([pos[n_pos_test:], neg[n_neg_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def auc_score(y_true: np.ndarray, y_score: np.ndarray):
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def standardize(X: np.ndarray, train_idx: np.ndarray):
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xz = (X - mean) / std
    return Xz, mean, std


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    torch.manual_seed(int(CONFIG["seed"]))
    x_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)

    w = torch.zeros((X.shape[1], 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)

    n_pos = float((y[train_idx] == 1).sum())
    n_neg = float((y[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    optimizer = torch.optim.Adam(
        [w, b],
        lr=float(CONFIG["lr"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )

    for step in range(1, int(CONFIG["steps"]) + 1):
        logits = x_train @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 300 == 0 or step == int(CONFIG["steps"]):
            print(f"logreg step {step}/{CONFIG['steps']} loss={loss.item():.4f}")

    return w.detach().cpu().numpy().reshape(-1), float(b.detach().cpu().item())


def predict_logits(X: np.ndarray, w: np.ndarray, b: float):
    return X @ w + b


def summarize_stats(X: np.ndarray, y: np.ndarray, stat_names):
    top = y == 1
    rest = y == 0
    out = []
    for i, name in enumerate(stat_names):
        top_vals = X[top, i]
        rest_vals = X[rest, i]
        top_mean = float(np.mean(top_vals))
        rest_mean = float(np.mean(rest_vals))
        top_std = float(np.std(top_vals))
        rest_std = float(np.std(rest_vals))
        pooled = np.sqrt(((top_std**2) + (rest_std**2)) / 2.0) + 1e-8
        effect_size = (top_mean - rest_mean) / pooled
        out.append(
            {
                "feature": name,
                "top_mean": top_mean,
                "rest_mean": rest_mean,
                "top_minus_rest": float(top_mean - rest_mean),
                "cohen_d": float(effect_size),
            }
        )
    return out


def get_enriched_features(top_prompt_counts, rest_prompt_counts, n_top, n_rest):
    eps = 1e-8
    top_rate = top_prompt_counts / max(n_top, 1)
    rest_rate = rest_prompt_counts / max(n_rest, 1)
    delta = top_rate - rest_rate
    lift = (top_rate + eps) / (rest_rate + eps)

    feature_ids = np.arange(len(top_prompt_counts), dtype=np.int64)
    rows = []
    for feat_id in feature_ids:
        rows.append(
            {
                "feature_id": int(feat_id),
                "top_rate": float(top_rate[feat_id]),
                "rest_rate": float(rest_rate[feat_id]),
                "delta_rate": float(delta[feat_id]),
                "lift": float(lift[feat_id]),
                "top_prompt_count": int(top_prompt_counts[feat_id]),
                "rest_prompt_count": int(rest_prompt_counts[feat_id]),
            }
        )

    enriched = sorted(rows, key=lambda r: r["delta_rate"], reverse=True)[
        : CONFIG["top_feature_count"]
    ]
    depleted = sorted(rows, key=lambda r: r["delta_rate"])[
        : CONFIG["top_feature_count"]
    ]
    return enriched, depleted


def main():
    score_run_path = Path(CONFIG["score_run_path"])
    samples_dir = Path(CONFIG["sae_samples_dir"])
    output_path = Path(CONFIG["output_path"])

    if not score_run_path.exists():
        raise FileNotFoundError(f"Missing score run path: {score_run_path}")

    scores, score_col = get_score_vector(score_run_path)
    sample_data = load_precomputed_samples(samples_dir, CONFIG["sae_id"])
    X = sample_data["sample_stats"]
    feature_presence = sample_data["feature_presence"]
    stat_names = sample_data["stat_names"]
    metadata = sample_data["metadata"]

    n = len(scores)
    if X.shape[0] != n:
        raise ValueError(
            f"Length mismatch: scores={n} precomputed_rows={X.shape[0]}. "
            "Make sure score run and SAE sample files belong to the same train split."
        )

    y, top_ids = top_k_labels(scores, int(CONFIG["top_k"]))
    print(f"loaded {n} rows from precomputed SAE arrays, positives(top_k)={int(y.sum())}")

    train_idx, test_idx = stratified_split(
        y,
        test_fraction=float(CONFIG["test_fraction"]),
        seed=int(CONFIG["seed"]),
    )
    Xz, mean, std = standardize(X, train_idx)
    w, b = fit_logistic_regression(Xz, y, train_idx)

    logits = predict_logits(Xz, w, b)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    train_auc = auc_score(y[train_idx], probs[train_idx])
    test_auc = auc_score(y[test_idx], probs[test_idx])
    full_auc = auc_score(y, probs)

    coef_rows = []
    for i, name in enumerate(stat_names):
        coef_rows.append(
            {
                "feature": name,
                "coef": float(w[i]),
                "abs_coef": float(abs(w[i])),
                "train_mean": float(mean[i]),
                "train_std": float(std[i]),
            }
        )
    coef_rows.sort(key=lambda r: r["abs_coef"], reverse=True)

    stat_diffs = summarize_stats(X, y, stat_names=stat_names)
    stat_diffs.sort(key=lambda r: abs(r["cohen_d"]), reverse=True)

    n_top = int((y == 1).sum())
    n_rest = int((y == 0).sum())
    top_prompt_counts = feature_presence[y == 1].sum(axis=0).astype(np.int64)
    rest_prompt_counts = feature_presence[y == 0].sum(axis=0).astype(np.int64)
    enriched, depleted = get_enriched_features(
        top_prompt_counts, rest_prompt_counts, n_top=n_top, n_rest=n_rest
    )

    top_rows = []
    for rank, idx in enumerate(top_ids.tolist(), start=1):
        meta = metadata[int(idx)]
        top_rows.append(
            {
                "rank": rank,
                "index": int(idx),
                "row_id": meta.get("row_id"),
                "task_id": meta.get("task_id"),
                "outcome": meta.get("outcome"),
                "score": float(scores[int(idx)]),
            }
        )

    out = {
        "config": CONFIG,
        "score_column": score_col,
        "precomputed_npz": sample_data["npz_path"],
        "precomputed_metadata": sample_data["metadata_path"],
        "n_samples": n,
        "n_top": n_top,
        "n_rest": n_rest,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "full_auc": full_auc,
        "regression_coefficients": coef_rows,
        "stat_feature_differences": stat_diffs,
        "top_enriched_features": enriched,
        "top_depleted_features": depleted,
        "top_rows": top_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"saved contrast analysis to: {output_path}")
    print(f"AUC train={train_auc:.4f} test={test_auc:.4f} full={full_auc:.4f}")
    print("top regression signals:")
    for row in coef_rows[:8]:
        print(f"  {row['feature']}: coef={row['coef']:.4f}")


if __name__ == "__main__":
    main()
