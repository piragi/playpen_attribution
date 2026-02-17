from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


_GUESS_RE = re.compile(r"GUESS\s*:\s*([^\n\r]+)", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_eval_rows(manifest: dict, split_name: str) -> list[dict]:
    split_path = manifest["splits"][split_name]["path"]
    ds = load_from_disk(split_path)
    rows = []
    for row in ds:
        rows.append(
            {
                "row_id": str(row.get("row_id", "")),
                "prompt": str(row["prompt"]),
                "gold": str(row["completion"]),
            }
        )
    return rows


def extract_guess(text: str) -> str:
    m = _GUESS_RE.search(text)
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = text.strip().splitlines()[0] if text.strip() else ""

    w = _WORD_RE.search(candidate.lower())
    return w.group(0) if w else ""


def load_model_and_tokenizer(base_model: str, adapter_path: str | None):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer_source = adapter_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    prompts: list[str],
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    out_texts: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_width = enc["input_ids"].shape[1]
        for i in range(gen.shape[0]):
            # With left padding, generated continuation starts after the full
            # padded input width (not after per-row non-pad token count).
            new_tokens = gen[i, input_width:]
            out_texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return out_texts


def evaluate_model(
    model_name: str,
    base_model: str,
    adapter_path: str | None,
    rows: list[dict],
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    model, tok = load_model_and_tokenizer(base_model=base_model, adapter_path=adapter_path)

    prompts = [r["prompt"] for r in rows]
    generations = generate_predictions(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    preds = []
    correct = 0
    for row, gen in zip(rows, generations):
        gold_word = extract_guess(row["gold"])
        pred_word = extract_guess(gen)
        is_correct = pred_word != "" and pred_word == gold_word
        if is_correct:
            correct += 1

        preds.append(
            {
                "row_id": row["row_id"],
                "gold_completion": row["gold"],
                "gold_word": gold_word,
                "pred_text": gen,
                "pred_word": pred_word,
                "correct": is_correct,
            }
        )

    total = len(rows)
    summary = {
        "model_name": model_name,
        "adapter_path": adapter_path,
        "correct": int(correct),
        "total": int(total),
        "word_accuracy": float(correct / total if total else 0.0),
    }
    return summary, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare word-guess accuracy across models.")
    parser.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    parser.add_argument("--eval-split", type=str, default="eval")
    parser.add_argument("--base-model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument(
        "--top-adapter",
        type=str,
        default="runs/simple_wordguesser_v1/scratch_top_k_mean_strict_smoke",
    )
    parser.add_argument(
        "--random-adapter",
        type=str,
        default="runs/simple_wordguesser_v1/scratch_random_k_mean_strict_smoke",
    )
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="runs/simple_wordguesser_v1/word_eval_compare.json",
    )
    parser.add_argument(
        "--output-preds-jsonl",
        type=str,
        default="runs/simple_wordguesser_v1/word_eval_compare_predictions.jsonl",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    manifest = load_manifest(args.manifest)
    rows = load_eval_rows(manifest, args.eval_split)

    model_defs = [
        ("base", None),
        ("finetuned_top", args.top_adapter),
        ("finetuned_random", args.random_adapter),
    ]

    summaries = []
    all_preds = []
    for model_name, adapter in model_defs:
        print(f"evaluating: {model_name}")
        summary, preds = evaluate_model(
            model_name=model_name,
            base_model=args.base_model,
            adapter_path=adapter,
            rows=rows,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        summaries.append(summary)
        for p in preds:
            p["model_name"] = model_name
            all_preds.append(p)
        print(
            f"{model_name}: {summary['correct']}/{summary['total']} "
            f"({summary['word_accuracy']:.4f})"
        )

    summaries = sorted(summaries, key=lambda x: x["word_accuracy"], reverse=True)
    output = {
        "manifest": args.manifest,
        "eval_split": args.eval_split,
        "base_model": args.base_model,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "results": summaries,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2))

    out_preds = Path(args.output_preds_jsonl)
    with out_preds.open("w") as f:
        for row in all_preds:
            f.write(json.dumps(row) + "\n")

    print(f"saved: {out_json}")
    print(f"saved: {out_preds}")


if __name__ == "__main__":
    main()
