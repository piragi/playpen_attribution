from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_split(manifest: dict, split_name: str):
    split_path = manifest["splits"][split_name]["path"]
    ds = load_from_disk(split_path)
    ds = ds.filter(lambda row: str(row.get("completion", "")).strip() != "")
    cols = [c for c in ("prompt", "completion") if c in ds.column_names]
    return ds.select_columns(cols)


def parse_modules(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def run_train(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    train_ds = load_split(manifest, args.train_split)
    eval_ds = load_split(manifest, args.eval_split) if args.eval_split else None

    print(f"train rows: {len(train_ds)}")
    if eval_ds is not None:
        print(f"eval rows:  {len(eval_ds)}")

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
            base,
            args.resume_adapter,
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
        peft_config = None
        tokenizer_source = args.resume_adapter
    else:
        model = base
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_modules(args.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        tokenizer_source = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate() if eval_ds is not None else {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    payload = {
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "base_model": args.base_model,
        "resume_adapter": args.resume_adapter,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "train_metrics": dict(result.metrics),
        "eval_metrics": eval_metrics,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(payload, indent=2))

    print(f"saved adapter + tokenizer to: {args.output_dir}")
    print(f"saved metrics to: {out_dir / 'train_metrics.json'}")


def infer_base_model(adapter_path: str) -> str:
    return str(PeftConfig.from_pretrained(adapter_path).base_model_name_or_path)


def run_merge(args: argparse.Namespace) -> None:
    base_model = args.base_model or infer_base_model(args.adapter_path)
    output_dir = args.output_dir or f"{args.adapter_path.rstrip('/')}_merged"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)

    tok = AutoTokenizer.from_pretrained(args.adapter_path)
    tok.save_pretrained(output_dir)
    print(f"saved merged model to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal finetuning entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    train.add_argument("--train-split", type=str, default="train_base")
    train.add_argument("--eval-split", type=str, default="eval")
    train.add_argument("--base-model", type=str, default="google/gemma-3-1b-it")
    train.add_argument("--resume-adapter", type=str, default=None)
    train.add_argument("--output-dir", type=str, default="runs/simple_wordguesser_v1/base_adapter")
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--seed", type=int, default=42)

    train.add_argument("--num-train-epochs", type=float, default=3.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--per-device-train-batch-size", type=int, default=1)
    train.add_argument("--per-device-eval-batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train.add_argument("--warmup-ratio", type=float, default=0.03)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--lr-scheduler-type", type=str, default="cosine")
    train.add_argument("--logging-steps", type=int, default=10)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--resume-from-checkpoint", type=str, default=None)

    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--lora-target-modules", type=str, default=DEFAULT_TARGET_MODULES)

    merge = sub.add_parser("merge")
    merge.add_argument("--adapter-path", type=str, required=True)
    merge.add_argument("--base-model", type=str, default=None)
    merge.add_argument("--output-dir", type=str, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
