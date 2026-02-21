"""LoRA SFT on pre-tokenized SmolTalk data.

Loads output of build_sft_data.py (input_ids + labels columns), fine-tunes
a Gemma base model with LoRA adapters using HF Trainer.

Data must already be tokenized with prompt tokens masked (labels = -100).
This keeps training and attribution in sync — Bergson sees the same masking.

Usage:
    uv run finetune.py            # train with CONFIG defaults
    uv run finetune.py --help

Subcommands:
    train   (default) Fine-tune and save adapter
    merge   Merge adapter weights into base model and save full model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

CONFIG = {
    "base_model": "google/gemma-3-270m",  # base model weights: larger gradient variance, matches GemmaScope SAE distribution
    "train_data": "runs/smoltalk_v1/data/train",
    "val_data": "runs/smoltalk_v1/data/val",
    "output_dir": "runs/smoltalk_v1/adapter",
    # Training
    "num_train_epochs": 2,
    "learning_rate": 3e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "logging_steps": 20,
    "save_steps": 500,
    "save_total_limit": 2,
    "seed": 42,
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
}


def parse_modules(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    train_ds = load_from_disk(args.train_data)
    val_ds = load_from_disk(args.val_data) if args.val_data else None

    # Keep only the columns Trainer needs; drop magpie_score, length, etc.
    keep = [c for c in ("input_ids", "labels") if c in train_ds.column_names]
    train_ds = train_ds.select_columns(keep)

    print(f"train rows: {len(train_ds):,}")
    # Val data is not used for in-training eval: Gemma 3's 262k vocab causes OOM
    # when casting logits to float32 during eval. Use eval_harness.py post-training.
    if val_ds is not None:
        print(f"val rows:   {len(val_ds):,} (not used for in-training eval)")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    base.config.use_cache = False

    if args.resume_adapter:
        model = PeftModel.from_pretrained(
            base, args.resume_adapter, is_trainable=True, autocast_adapter_dtype=False
        )
        tokenizer_source = args.resume_adapter
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_modules(args.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        tokenizer_source = args.base_model

    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Always use the IT tokenizer for its chat template.
    # If resuming from an adapter that already has the IT tokenizer saved, load from there.
    # Otherwise fall back to google/gemma-3-270m-it (same vocab as base, adds chat_template).
    it_tokenizer_source = tokenizer_source if args.resume_adapter else "google/gemma-3-270m-it"
    tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # DataCollatorForSeq2Seq pads input_ids with pad_token_id and labels with -100.
    # This is exactly what we want for pre-masked decoder-only SFT.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,  # overrides num_train_epochs when > 0
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    out_dir = Path(args.output_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    payload = {
        "base_model": args.base_model,
        "train_data": args.train_data,
        "train_rows": len(train_ds),
        "train_metrics": dict(result.metrics),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"Adapter saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def run_merge(args: argparse.Namespace) -> None:
    base_model = args.base_model or str(
        PeftConfig.from_pretrained(args.adapter_path).base_model_name_or_path
    )
    out_dir = args.output_dir or f"{args.adapter_path.rstrip('/')}_merged"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, attn_implementation="eager"
    )
    merged = PeftModel.from_pretrained(base, args.adapter_path).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(args.adapter_path).save_pretrained(out_dir)
    print(f"Merged model saved to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT on pre-tokenized data.")
    sub = parser.add_subparsers(dest="command")

    # Default: train (no subcommand needed)
    train = sub.add_parser("train")
    _add_train_args(train)

    merge = sub.add_parser("merge")
    merge.add_argument("--adapter-path", required=True)
    merge.add_argument("--base-model", default=None)
    merge.add_argument("--output-dir", default=None)

    # Allow running without a subcommand (defaults to train with CONFIG)
    parser.set_defaults(command="train")
    _add_train_args(parser)

    return parser


def _add_train_args(p: argparse.ArgumentParser) -> None:
    c = CONFIG
    p.add_argument("--base-model", default=c["base_model"])
    p.add_argument("--train-data", default=c["train_data"])
    p.add_argument("--val-data", default=c["val_data"])
    p.add_argument("--output-dir", default=c["output_dir"])
    p.add_argument("--resume-adapter", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--num-train-epochs", type=float, default=c["num_train_epochs"])
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Hard step limit overriding num_train_epochs. -1 = use epochs.")
    p.add_argument("--learning-rate", type=float, default=c["learning_rate"])
    p.add_argument("--warmup-ratio", type=float, default=c["warmup_ratio"])
    p.add_argument("--weight-decay", type=float, default=c["weight_decay"])
    p.add_argument("--lr-scheduler-type", default=c["lr_scheduler_type"])
    p.add_argument("--per-device-train-batch-size", type=int, default=c["per_device_train_batch_size"])
    p.add_argument("--gradient-accumulation-steps", type=int, default=c["gradient_accumulation_steps"])
    p.add_argument("--logging-steps", type=int, default=c["logging_steps"])
    p.add_argument("--save-steps", type=int, default=c["save_steps"])
    p.add_argument("--save-total-limit", type=int, default=c["save_total_limit"])
    p.add_argument("--seed", type=int, default=c["seed"])
    p.add_argument("--lora-r", type=int, default=c["lora_r"])
    p.add_argument("--lora-alpha", type=int, default=c["lora_alpha"])
    p.add_argument("--lora-dropout", type=float, default=c["lora_dropout"])
    p.add_argument("--lora-target-modules", default=c["lora_target_modules"])


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "merge":
        run_merge(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
