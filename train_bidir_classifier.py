from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from bergson.data import load_scores
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def split_path(manifest: dict, split_name: str) -> Path:
    split = manifest.get("splits", {}).get(split_name)
    if not split or "path" not in split:
        raise KeyError(f"Split '{split_name}' not found in manifest.")
    return Path(split["path"])


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


def row_text(row: dict) -> str:
    prompt = str(row.get("prompt", "")).strip()
    completion = str(row.get("completion", "")).strip()
    if not prompt and not completion and "messages" in row:
        prompt, completion = build_prompt_completion(list(row["messages"]))

    text = (prompt + ("\n\n" + completion if completion else "")).strip()
    if not text:
        raise ValueError("Found empty row text while building classifier dataset.")
    return text


def row_field(row: dict, name: str):
    if name in row and row[name] is not None:
        return row[name]
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta.get(name)
    return None


def make_group_key(row: dict, group_fields: list[str]) -> str:
    return "||".join(str(row_field(row, f) or "<missing>") for f in group_fields)


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
                "text": row_text(row),
                "group_key": make_group_key(row, group_fields),
                "game": row_field(row, "game"),
                "game_role": row_field(row, "game_role"),
                "experiment": row_field(row, "experiment"),
                "task_id": row_field(row, "task_id"),
                "outcome": row_field(row, "outcome"),
                "pair_key": row_field(row, "pair_key"),
            }
        )
    return rows


def load_score_vector(score_run_path: Path) -> tuple[np.ndarray, str]:
    scores = load_scores(score_run_path)
    try:
        values = np.asarray(scores.get(slice(None), score_idx=0), dtype=np.float32)
        if values.ndim == 2:
            values = values[:, 0]
        return values.astype(np.float32), "score_0"
    except Exception:
        arr = np.asarray(scores[:], dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return arr.astype(np.float32), "score_0"


def sanitize_scores(scores: np.ndarray, drop_non_finite: bool) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(scores)
    if finite.all():
        keep = np.arange(len(scores), dtype=np.int64)
        return scores.astype(np.float32), keep

    if drop_non_finite:
        keep = np.where(finite)[0].astype(np.int64)
        if keep.size <= 1:
            raise RuntimeError("Too few finite attribution scores after filtering.")
        clean = scores[keep].astype(np.float32)
        return clean, keep

    fill = float(np.nanmin(scores[finite]) - 1.0) if finite.any() else -1.0
    clean = np.nan_to_num(scores, nan=fill, posinf=fill, neginf=fill).astype(np.float32)
    keep = np.arange(len(scores), dtype=np.int64)
    return clean, keep


def label_top(scores: np.ndarray, top_k: int, top_fraction: float | None) -> tuple[np.ndarray, np.ndarray]:
    n = int(scores.shape[0])
    if n <= 1:
        raise ValueError("Need at least 2 rows for top-vs-rest labels.")
    k = int(round(n * float(top_fraction))) if top_fraction is not None else int(top_k)
    k = max(1, min(n - 1, k))
    top_ids = np.argsort(-scores)[:k]
    labels = np.zeros(n, dtype=np.int64)
    labels[top_ids] = 1
    return labels, top_ids


def stratified_split(y: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both positive and negative labels.")

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
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def grouped_split(
    y: np.ndarray, group_keys: list[str], test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(group_keys):
        groups[str(key)].append(i)

    if len(groups) < 2:
        return stratified_split(y, test_fraction, seed)

    pos_groups = []
    neg_groups = []
    for key, idxs in groups.items():
        has_pos = bool(np.sum(y[np.asarray(idxs, dtype=np.int64)]) > 0)
        if has_pos:
            pos_groups.append(key)
        else:
            neg_groups.append(key)

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    n_pos_test = max(1, int(round(len(pos_groups) * test_fraction)))
    n_neg_test = max(1, int(round(len(neg_groups) * test_fraction)))
    n_pos_test = min(n_pos_test, len(pos_groups) - 1) if len(pos_groups) > 1 else 0
    n_neg_test = min(n_neg_test, len(neg_groups) - 1) if len(neg_groups) > 1 else 0

    test_group_set = set(pos_groups[:n_pos_test] + neg_groups[:n_neg_test])
    train_idx = np.asarray(
        [i for i, key in enumerate(group_keys) if key not in test_group_set], dtype=np.int64
    )
    test_idx = np.asarray(
        [i for i, key in enumerate(group_keys) if key in test_group_set], dtype=np.int64
    )

    if train_idx.size == 0 or test_idx.size == 0:
        return stratified_split(y, test_fraction, seed)
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return stratified_split(y, test_fraction, seed)
    return train_idx, test_idx


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    return float((pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = y_true.astype(np.int64)
    y_pred = (probs >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_score(y_true, probs)),
        "n": int(y_true.size),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def make_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(args.num_train_epochs),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=int(args.logging_steps),
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        seed=int(args.seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bidirectional BERT detector on attribution top-k labels.")
    parser.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    parser.add_argument("--label-split", type=str, default="score_pool")
    parser.add_argument("--score-split", type=str, default="score_pool")
    parser.add_argument(
        "--score-run-path",
        type=str,
        default="runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/train_scores",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs/simple_wordguesser_v1/bidir_classifier",
    )
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--score-max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--drop-non-finite-scores", action="store_true")
    parser.add_argument("--no-drop-non-finite-scores", dest="drop_non_finite_scores", action="store_false")
    parser.set_defaults(drop_non_finite_scores=True)
    parser.add_argument("--split-mode", choices=["grouped", "stratified"], default="grouped")
    parser.add_argument("--group-key-fields", type=str, default="game,game_role,experiment,task_id")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=20)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    label_path = split_path(manifest, args.label_split)
    score_path = split_path(manifest, args.score_split)
    group_fields = [x.strip() for x in args.group_key_fields.split(",") if x.strip()]

    label_rows_raw = load_rows(label_path, args.max_rows, group_fields)
    score_rows = load_rows(score_path, args.score_max_rows, group_fields)
    scores_raw, score_col = load_score_vector(Path(args.score_run_path))

    if len(scores_raw) < len(label_rows_raw):
        raise ValueError(
            "Score vector shorter than labeled rows: "
            f"scores={len(scores_raw)} rows={len(label_rows_raw)}"
        )
    if len(scores_raw) != len(label_rows_raw):
        print(
            "warning: score length and label rows differ; using prefix alignment "
            f"(scores={len(scores_raw)} rows={len(label_rows_raw)})"
        )
        scores_raw = scores_raw[: len(label_rows_raw)]

    scores_clean, keep_idx = sanitize_scores(scores_raw, args.drop_non_finite_scores)
    label_rows = [label_rows_raw[int(i)] for i in keep_idx.tolist()]
    labels, top_ids = label_top(scores_clean, args.top_k, args.top_fraction)
    group_keys = [row["group_key"] for row in label_rows]

    if args.split_mode == "grouped":
        train_idx, test_idx = grouped_split(labels, group_keys, args.test_fraction, args.seed)
    else:
        train_idx, test_idx = stratified_split(labels, args.test_fraction, args.seed)

    raw_ds = Dataset.from_dict(
        {
            "text": [row["text"] for row in label_rows],
            "label": labels.tolist(),
            "row_id": [row["row_id"] for row in label_rows],
        }
    )
    score_ds = Dataset.from_dict(
        {
            "text": [row["text"] for row in score_rows],
            "row_id": [row["row_id"] for row in score_rows],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    def tokenize_fn(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = raw_ds.map(tokenize_fn, batched=True)
    tokenized_score = score_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    train_args = make_training_args(args, output_root / "model")

    def compute_metrics(eval_pred) -> dict:
        logits, y = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = softmax_2d(np.asarray(logits))[:, 1]
        metrics = binary_metrics(np.asarray(y), probs)
        return {f"eval_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized.select(train_idx.tolist()),
        eval_dataset=tokenized.select(test_idx.tolist()),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    pred_all = trainer.predict(tokenized)
    logits_all = np.asarray(pred_all.predictions)
    probs_all = softmax_2d(logits_all)[:, 1]
    full_metrics = binary_metrics(labels, probs_all)
    train_metrics = binary_metrics(labels[train_idx], probs_all[train_idx])
    test_metrics = binary_metrics(labels[test_idx], probs_all[test_idx])

    pred_score = trainer.predict(tokenized_score)
    logits_score = np.asarray(pred_score.predictions)
    probs_score = softmax_2d(logits_score)[:, 1]

    k = int(labels.sum())
    top_pred = np.argsort(-probs_all)[:k]
    top_overlap = int(len(set(top_ids.tolist()) & set(top_pred.tolist())))

    label_by_row_id = {row["row_id"]: int(labels[i]) for i, row in enumerate(label_rows)}
    rank_order = np.argsort(-probs_score)
    row_scores_path = output_root / "row_scores.jsonl"
    with row_scores_path.open("w") as f:
        for rank, idx in enumerate(rank_order, start=1):
            row = score_rows[int(idx)]
            rec = {
                "rank": int(rank),
                "index": int(idx),
                "row_id": row["row_id"],
                "attribution_label_top": label_by_row_id.get(row["row_id"]),
                "classifier_prob_top": float(probs_score[int(idx)]),
                "classifier_logit_top": float(logits_score[int(idx), 1]),
                "game": row.get("game"),
                "game_role": row.get("game_role"),
                "experiment": row.get("experiment"),
                "task_id": row.get("task_id"),
                "outcome": row.get("outcome"),
                "pair_key": row.get("pair_key"),
            }
            f.write(json.dumps(rec) + "\n")

    trainer.save_model(str(output_root / "model"))
    tokenizer.save_pretrained(str(output_root / "model"))

    summary = {
        "manifest": args.manifest,
        "label_split": args.label_split,
        "score_split": args.score_split,
        "label_dataset_path": str(label_path),
        "score_dataset_path": str(score_path),
        "score_run_path": args.score_run_path,
        "score_column": score_col,
        "n_rows_raw": int(len(label_rows_raw)),
        "n_rows_labeled": int(len(label_rows)),
        "n_score_rows": int(len(score_rows)),
        "n_positive_labels": int(labels.sum()),
        "split_mode": args.split_mode,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
        "topk_overlap_with_attribution": {
            "k": int(k),
            "overlap_count": int(top_overlap),
            "overlap_fraction": float(top_overlap / max(k, 1)),
        },
        "trainer_train_metrics": {
            k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))
        },
        "trainer_eval_metrics": {
            k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))
        },
        "row_scores_path": str(row_scores_path),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved model + outputs to: {output_root}")
    print(
        f"test auc={test_metrics['auc']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )
    print(
        f"top-k overlap with attribution labels: "
        f"{top_overlap}/{k} ({top_overlap / max(k, 1):.4f})"
    )
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
