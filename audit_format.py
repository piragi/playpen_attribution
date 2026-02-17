from __future__ import annotations

import argparse
import json
from pathlib import Path

from bergson.config import DataConfig
from bergson.data import tokenize
from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_indices(raw: str, n_rows: int) -> list[int]:
    if raw.strip().lower() == "head":
        return list(range(min(3, n_rows)))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_rows:
            raise ValueError(f"Index {idx} out of range [0, {n_rows - 1}]")
        out.append(idx)
    if not out:
        raise ValueError("No valid row indices parsed.")
    return out


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def finetune_token_view(tokenizer, prompt: str, completion: str) -> dict:
    eos = tokenizer.eos_token or ""
    completion_for_train = completion
    if eos and not completion_for_train.endswith(eos):
        completion_for_train = completion_for_train + eos

    prompt_ids = tokenizer(prompt)["input_ids"]
    full_ids = tokenizer(prompt + completion_for_train)["input_ids"]
    completion_ids = full_ids[len(prompt_ids) :]

    return {
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "completion_ids": completion_ids,
        "prompt_len": len(prompt_ids),
        "full_len": len(full_ids),
        "completion_len": len(completion_ids),
        "prefix_ok": full_ids[: len(prompt_ids)] == prompt_ids,
        "completion_for_train": completion_for_train,
    }


def bergson_token_view(tokenizer, prompt: str, completion: str, max_length: int) -> dict:
    cfg = DataConfig(prompt_column="prompt", completion_column="completion", truncation=True)
    batch = {"prompt": [prompt], "completion": [completion]}
    tokd = tokenize(batch, args=cfg, tokenizer=tokenizer, max_length=max_length)
    ids = tokd["input_ids"][0]
    labels = tokd["labels"][0]
    supervised_ids = [tid for tid, lab in zip(ids, labels) if lab != -100]

    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=False,
    )

    return {
        "full_ids": ids,
        "labels": labels,
        "supervised_ids": supervised_ids,
        "full_len": len(ids),
        "supervised_len": len(supervised_ids),
        "chat_text": chat_text,
    }


def eval_token_view(tokenizer, prompt: str, max_length: int) -> dict:
    ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return {
        "input_ids": ids,
        "input_len": len(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit token-format consistency across stages.")
    parser.add_argument("--manifest", type=str, default="runs/simple_wordguesser_v1/manifest.json")
    parser.add_argument("--split", type=str, default="train_base")
    parser.add_argument("--tokenizer", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--indices", type=str, default="head")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--score-run", type=str, default="")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    split_path = manifest["splits"][args.split]["path"]
    ds = load_from_disk(split_path)
    idxs = parse_indices(args.indices, len(ds))

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    score_ds = None
    if args.score_run:
        score_path = Path(args.score_run) / "data.hf"
        if score_path.exists():
            score_ds = load_from_disk(str(score_path))

    print(f"split: {args.split} rows={len(ds)} tokenizer={args.tokenizer}")
    print(f"indices: {idxs}")

    for idx in idxs:
        row = ds[idx]
        prompt = str(row["prompt"])
        completion = str(row["completion"])

        ft = finetune_token_view(tok, prompt, completion)
        bg = bergson_token_view(tok, prompt, completion, args.max_length)
        ev = eval_token_view(tok, prompt, args.max_length)

        print("\n" + "=" * 88)
        print(f"row_idx={idx} row_id={row.get('row_id')}")
        print(f"completion_raw={repr(completion[:200])}")
        print(f"prompt_tail={repr(prompt[-220:])}")
        print(f"concat_boundary={repr(prompt[-50:] + completion[:50])}")

        print("\n[finetune]")
        print(f"prompt_len={ft['prompt_len']} full_len={ft['full_len']} completion_len={ft['completion_len']}")
        print(f"prefix_ok={ft['prefix_ok']}")
        print(f"completion_for_train={repr(ft['completion_for_train'][:200])}")
        print(
            "completion_decoded="
            + repr(tok.decode(ft["completion_ids"], skip_special_tokens=False)[:260])
        )
        print("full_tail_decoded=" + repr(tok.decode(ft["full_ids"], skip_special_tokens=False)[-260:]))

        print("\n[bergson]")
        print(f"full_len={bg['full_len']} supervised_len={bg['supervised_len']}")
        print(
            "supervised_decoded="
            + repr(tok.decode(bg["supervised_ids"], skip_special_tokens=False)[:260])
        )
        print("full_tail_decoded=" + repr(tok.decode(bg["full_ids"], skip_special_tokens=False)[-260:]))
        print("chat_text_tail=" + repr(bg["chat_text"][-260:]))

        print("\n[eval]")
        print(f"input_len={ev['input_len']}")
        print("input_tail_decoded=" + repr(tok.decode(ev["input_ids"], skip_special_tokens=False)[-260:]))

        print("\n[comparisons]")
        print(f"finetune_vs_bergson_full_ids_equal={ft['full_ids'] == bg['full_ids']}")
        print(f"finetune_vs_bergson_supervised_equal={ft['completion_ids'] == bg['supervised_ids']}")

        if score_ds is not None and idx < len(score_ds):
            score_row = score_ds[idx]
            labels = score_row.get("labels")
            if labels is not None:
                supervised = sum(1 for t in labels if t != -100)
                print("\n[bergson_score_artifact]")
                print(f"score_data.length={score_row.get('length')} score_data.loss={score_row.get('loss')}")
                print(f"score_data.supervised_tokens={supervised}")


if __name__ == "__main__":
    main()
