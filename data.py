import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

CONFIG = {
    "model_name": "google/gemma-3-1b-it",
    "dataset_name": "colab-potsdam/playpen-data",
    "dataset_config": "interactions",
    "game": "taboo",
    "role": "WordGuesser",  # WordGuesser or WordDescriber
    "output_dir": "./taboo_sft_lora",
    "max_length": 384,
    "seed": 42,
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "train": {
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "logging_steps": 10,
    },
}


def keep_example(example):
    meta = example["meta"]
    return (
        meta["game"] == CONFIG["game"]
        and meta["game_role"] == CONFIG["role"]
        and meta["outcome"] != "aborted"
    )


def build_history(messages):
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)


def to_prompt_completion(example):
    messages = example["messages"]
    assistant_positions = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    if not assistant_positions:
        return {"prompt": "", "completion": ""}

    last_assistant_idx = assistant_positions[-1]
    prompt_history = build_history(messages[:last_assistant_idx]).strip()
    completion = messages[last_assistant_idx]["content"].strip()

    return {
        "prompt": prompt_history,
        "completion": completion,
    }


def main():
    dataset = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_config"])
    train_dataset = dataset["train"].filter(keep_example).map(to_prompt_completion)
    eval_dataset = dataset["validation"].filter(keep_example).map(to_prompt_completion)
    train_dataset = train_dataset.filter(lambda x: x["completion"] != "")
    eval_dataset = eval_dataset.filter(lambda x: x["completion"] != "")
    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("messages")
    if "messages" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns("messages")

    print(f"train rows: {len(train_dataset)}")
    print(f"validation rows: {len(eval_dataset)}")

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Filtered train/eval dataset is empty. Check CONFIG values.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["alpha"],
        lora_dropout=CONFIG["lora"]["dropout"],
        target_modules=CONFIG["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["train"]["num_train_epochs"],
        learning_rate=CONFIG["train"]["learning_rate"],
        per_device_train_batch_size=CONFIG["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["train"]["gradient_accumulation_steps"],
        warmup_ratio=CONFIG["train"]["warmup_ratio"],
        lr_scheduler_type="cosine",
        weight_decay=CONFIG["train"]["weight_decay"],
        max_length=CONFIG["max_length"],
        packing=False,
        assistant_only_loss=False,
        completion_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=CONFIG["train"]["logging_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none",
        seed=CONFIG["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"saved adapter and tokenizer to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
