from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Iterator
import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
def make_cosine_with_min_lr(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float
) -> LambdaLR:
    """Cosine decay from max_lr to min_lr_ratio * max_lr, with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)
class PretrainTrainer(Trainer):
    """Trainer subclass that installs a cosine schedule with a non-zero floor LR."""
    def __init__(self, *args, min_lr_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_lr_ratio = min_lr_ratio
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = make_cosine_with_min_lr(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=self.min_lr_ratio,
        )
        return self.lr_scheduler
def token_stream(dataset, tokenizer, text_field: str, max_tokens: int) -> Iterator[list[int]]:
    """Yield token lists (with EOS appended) from a streaming dataset.
    Stops after approximately max_tokens total tokens have been yielded.
    """
    eos_id = tokenizer.eos_token_id
    total = 0
    for row in dataset:
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            ids = ids + [eos_id]
        yield ids
        total += len(ids)
        if 0 < max_tokens <= total:
            break
def build_packed_dataset(args: argparse.Namespace, tokenizer) -> Dataset:
    """Stream the source dataset, tokenize, and pack into fixed-length chunks."""
    print(f"Streaming {args.dataset_name}/{args.dataset_config} split={args.dataset_split}")
    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    )
    buffer: list[int] = []
    chunks: list[dict] = []
    chunk_size = args.max_length
    for ids in token_stream(raw, tokenizer, args.text_field, args.max_tokens):
        buffer.extend(ids)
        while len(buffer) >= chunk_size:
            chunks.append({"input_ids": buffer[:chunk_size]})
            buffer = buffer[chunk_size:]
        if chunks and len(chunks) % 1000 == 0:
            print(
                f"\r  packed {len(chunks):,} chunks "
                f"({len(chunks) * chunk_size / 1e6:.1f}M tokens)",
                end="",
                flush=True,
            )
    print(
        f"\nPacked {len(chunks):,} chunks × {chunk_size} tokens"
        f" = {len(chunks) * chunk_size / 1e6:.1f}M tokens"
    )
    return Dataset.from_list(chunks)
def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.dataset_path:
        print(f"Loading dataset from disk: {args.dataset_path}")
        train_ds = load_from_disk(args.dataset_path)
        # Keep only input_ids — collator creates labels by shifting
        extra = [c for c in train_ds.column_names if c != "input_ids"]
        if extra:
            train_ds = train_ds.remove_columns(extra)
        print(f"  Loaded {len(train_ds):,} chunks")
    else:
        train_ds = build_packed_dataset(args, tokenizer)
    if len(train_ds) == 0:
        raise RuntimeError("No packed chunks — check --max-tokens and dataset availability.")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    print(f"Loading {args.base_model} ({dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    num_steps = len(train_ds) // effective_batch
    min_lr_ratio = args.min_learning_rate / args.learning_rate
    print(
        f"steps={num_steps}  effective_batch={effective_batch}"
        f"  warmup={args.warmup_steps}"
    )
    print(
        f"lr: {args.learning_rate} → {args.min_learning_rate}"
        f"  (ratio={min_lr_ratio:.2f}, cosine)"
    )
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        max_steps=num_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        # "constant" is a valid type; PretrainTrainer.create_scheduler overrides it.
        lr_scheduler_type="constant",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = PretrainTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
        min_lr_ratio=min_lr_ratio,
    )
    result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    payload = {
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "max_tokens_target": args.max_tokens,
        "max_length": args.max_length,
        "num_chunks": len(train_ds),
        "total_tokens": len(train_ds) * args.max_length,
        "num_steps": num_steps,
        "effective_batch_size": effective_batch,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "train_metrics": dict(result.metrics),
    }
    (out_dir / "pretrain_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved model + tokenizer to: {out_dir}")
    print(f"Saved metrics to: {out_dir / 'pretrain_metrics.json'}")
def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continued pretraining on FineWeb (general web data)."
    )
    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m")
    parser.add_argument("--output-dir", type=str, default="runs/pretrain_270m_v1")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a saved HF dataset (load_from_disk). "
             "If set, skips FineWeb streaming and trains on this dataset directly.",
    )
    # Dataset — using raw FineWeb (not FineWeb-Edu) to preserve quality variance.
    # CC-MAIN-2024-10 is a recent, high-quality dump recommended by the FineWeb team.
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", type=str, default="CC-MAIN-2024-10")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100_000_000,
        help="Stop after approximately this many tokens have been streamed (100M default).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Packed sequence length for each training chunk.",
    )
    # Optimiser / schedule
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-6,
        help="Cosine schedule decays to this floor LR.",
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # Batch / hardware
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Attention implementation. Gemma-3 officially recommends 'eager'.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade ~30%% compute for lower activation memory (on by default).",
    )
    # Checkpointing / logging
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser
def main() -> None:
    args = make_parser().parse_args()
    run(args)
if __name__ == "__main__":
    main()