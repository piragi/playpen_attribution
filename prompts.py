import re


def keep_example(example, game: str, role: str):
    meta = example["meta"]
    return (
        meta["game"] == game
        and meta["game_role"] == role
        and meta["outcome"] != "aborted"
    )


def build_history(messages) -> str:
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)


def build_prompt_completion(messages) -> dict[str, str]:
    assistant_positions = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    if not assistant_positions:
        return {"prompt": "", "completion": ""}

    last_assistant_idx = assistant_positions[-1]
    prompt = build_history(messages[:last_assistant_idx]).strip()
    completion = messages[last_assistant_idx]["content"].strip()
    return {"prompt": prompt, "completion": completion}


def build_full_text(messages) -> str:
    return build_history(messages).strip()


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
