"""Synthetic Taboo data generation via two-player LLM simulation."""

import asyncio
import json
import random
import re
from pathlib import Path

from datasets import Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "output_dir": "./runs/taboo_attr/data/synthetic",
    "raw_output": "./runs/taboo_attr/synthetic_raw.jsonl",
    "episodes_per_word": 50,
    "models": {
        "haiku": "claude-haiku-4-5-20251001",
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-4o-mini",
    },
    "max_rounds": 3,
    "max_retries": 3,
    "concurrency_per_provider": 10,
    "seed": 42,
}

# Original 20 words (extracted from existing task_ids via successful guesses)
ORIGINAL_WORDS = [
    "winter", "chicken", "fly", "fort", "hatch",
    "energy", "translate", "bradley", "gibson", "fault",
    "song", "ferry", "sure", "pool", "council",
    "institutional", "colored", "landmark", "market", "realistic",
]

# New words — mix of polysemous, abstract, part-of-speech-ambiguous, and concrete
# Categories:
#   polysemous (multiple meanings): bank, cabinet, draft, spring, pitch, etc.
#   abstract / hard to describe: subtle, eventual, contingent, etc.
#   verb/noun ambiguous: plant, bolt, jam, seal, etc.
#   proper-noun-adjacent: sterling, darwin, morse, etc.
#   concrete but tricky: compass, anchor, prism, etc.
NEW_WORDS = [
    # --- polysemous / multiple meanings ---
    "bank", "cabinet", "draft", "spring", "pitch",
    "bark", "crane", "current", "match", "plot",
    "stake", "novel", "cell", "compound", "harbor",
    "scale", "board", "check", "file", "wave",
    "charge", "palm", "bow", "organ", "trip",
    "light", "date", "train", "lead", "bass",
    "seal", "bolt", "jam", "plant", "rock",
    "tire", "ring", "tap", "cast", "blow",
    # --- abstract / hard to clue ---
    "subtle", "eventual", "contingent", "inherent", "peripheral",
    "arbitrary", "provisional", "nominal", "implicit", "aggregate",
    "threshold", "paradox", "nuance", "leverage", "catalyst",
    "paradigm", "entropy", "rhetoric", "consensus", "momentum",
    # --- part-of-speech ambiguous ---
    "desert", "address", "project", "record", "permit",
    "conflict", "suspect", "produce", "rebel", "refuse",
    "contract", "digest", "escort", "extract", "impact",
    "insult", "object", "perfect", "present", "progress",
    # --- proper-noun-adjacent / cultural ---
    "sterling", "darwin", "morse", "pascal", "summit",
    "marathon", "spartan", "mercury", "atlas", "phoenix",
    "titan", "olympia", "alpine", "aurora", "delta",
    "savannah", "sierra", "amazon", "cannon", "jordan",
    # --- concrete but with twist ---
    "compass", "anchor", "prism", "valve", "lever",
    "furnace", "scaffold", "turbine", "circuit", "satellite",
    "fossil", "glacier", "canyon", "reef", "eclipse",
    "mosaic", "labyrinth", "pendulum", "siren", "beacon",
    # --- adjectives that resist easy cluing ---
    "hollow", "blunt", "crude", "steep", "volatile",
    "dormant", "opaque", "rigid", "stale", "acute",
    "chronic", "obscure", "pristine", "synthetic", "terminal",
    "benign", "candid", "elusive", "mundane", "orthodox",
]

ALL_WORDS = ORIGINAL_WORDS + NEW_WORDS

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GUESSER_SYSTEM_PROMPT = (
    "You are playing a collaborative word guessing game in which you have to "
    "guess a target word that another player describes to you.\n\n"
    "You can make one guess at each trial. You win when you guess the target "
    "word. You lose when you cannot guess it in 3 tries.\n\n"
    "After each trial you will get a new hint from the other player which "
    "starts with CLUE.\n\n"
    "Make your guesses by just saying the word using the following form: "
    "GUESS: <a word>\n\n"
    "Let us start."
)


def describer_system_prompt(word: str, prev_guesses: list[str]) -> str:
    lines = [
        f'You are playing Taboo as the clue-giver. The target word is "{word}".',
        "Give a single clue to help the guesser find the target word.",
        "- Start your clue with \"CLUE:\"",
        "- Do not use the target word or parts of it in your clue",
        "- Be creative and descriptive",
    ]
    if prev_guesses:
        wrong = ", ".join(f'"{g}"' for g in prev_guesses)
        lines.append(f"The guesser previously guessed {wrong} incorrectly. Adapt your clue.")
    lines.append("Respond with only the clue, nothing else.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM client abstraction (async)
# ---------------------------------------------------------------------------

# Per-provider semaphores (created lazily)
_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(provider: str) -> asyncio.Semaphore:
    if provider not in _semaphores:
        _semaphores[provider] = asyncio.Semaphore(CONFIG["concurrency_per_provider"])
    return _semaphores[provider]


async def call_llm(model_key: str, system_prompt: str, messages: list[dict]) -> str:
    """Dispatch an LLM call to the appropriate provider. Returns raw text."""
    model_name = CONFIG["models"][model_key]

    if model_key == "haiku":
        return await _call_anthropic(model_name, system_prompt, messages)
    elif model_key == "gemini":
        return await _call_gemini(model_name, system_prompt, messages)
    elif model_key == "openai":
        return await _call_openai(model_name, system_prompt, messages)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


async def _call_anthropic(model: str, system: str, messages: list[dict]) -> str:
    import anthropic

    sem = _get_semaphore("anthropic")
    async with sem:
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model=model,
            max_tokens=128,
            system=system,
            messages=messages,
        )
        return resp.content[0].text


async def _call_gemini(model: str, system: str, messages: list[dict]) -> str:
    from google import genai
    from google.genai import types

    sem = _get_semaphore("gemini")
    async with sem:
        client = genai.Client()
        # Convert messages to Gemini Content format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        resp = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=128,
            ),
        )
        return resp.text


async def _call_openai(model: str, system: str, messages: list[dict]) -> str:
    import openai

    sem = _get_semaphore("openai")
    async with sem:
        client = openai.AsyncOpenAI()
        oai_messages = [{"role": "system", "content": system}]
        oai_messages.extend(messages)
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=128,
            messages=oai_messages,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------


def parse_guess(text: str) -> str:
    """Extract the guessed word from a GUESS: response."""
    text = text.strip()
    m = re.search(r"GUESS:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(".").strip()
    # Fallback: return the whole text stripped
    return text.strip().strip(".").strip()


def parse_clue(text: str) -> str:
    """Ensure clue text starts with CLUE:."""
    text = text.strip()
    if not text.upper().startswith("CLUE:"):
        text = "CLUE: " + text
    return text


async def generate_episode(
    word: str, task_id: int, model_key: str
) -> dict | None:
    """Run a two-player Taboo game and return the episode dict."""
    max_retries = CONFIG["max_retries"]

    for attempt in range(max_retries):
        try:
            return await _generate_episode_inner(word, task_id, model_key)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                print(f"  FAILED after {max_retries} retries: word={word}, model={model_key}: {e}")
                return None


async def _generate_episode_inner(
    word: str, task_id: int, model_key: str
) -> dict:
    prev_guesses: list[str] = []
    # Final messages list in the output format
    final_messages: list[dict] = []
    outcome = "failure"

    for round_idx in range(CONFIG["max_rounds"]):
        # --- Describer turn ---
        desc_system = describer_system_prompt(word, prev_guesses)
        # Describer doesn't need multi-turn context, just system prompt
        desc_messages: list[dict] = []
        if prev_guesses:
            desc_messages.append({
                "role": "user",
                "content": f"The guesser said: GUESS: {prev_guesses[-1]}. That was wrong. Give another clue.",
            })
        else:
            desc_messages.append({"role": "user", "content": "Give your first clue."})

        raw_clue = await call_llm(model_key, desc_system, desc_messages)
        clue = parse_clue(raw_clue)

        # --- Build guesser's user message ---
        if round_idx == 0:
            # First round: system prompt + first clue (matches existing format)
            guesser_user_msg = f"{GUESSER_SYSTEM_PROMPT}\n\n\n{clue}"
        else:
            guesser_user_msg = clue

        final_messages.append({"role": "user", "content": guesser_user_msg})

        # --- Guesser turn ---
        # Build guesser context: all messages so far
        guesser_system = GUESSER_SYSTEM_PROMPT
        # For guesser API call, use only user/assistant message pairs (not system prompt in content)
        guesser_api_messages = []
        for msg in final_messages:
            guesser_api_messages.append({"role": msg["role"], "content": msg["content"]})

        raw_guess = await call_llm(model_key, guesser_system, guesser_api_messages)
        guess_word = parse_guess(raw_guess)

        # Format the assistant response
        guess_text = raw_guess.strip()
        if not guess_text.upper().startswith("GUESS:"):
            guess_text = f"GUESS: {guess_word}"
        final_messages.append({"role": "assistant", "content": guess_text})

        # --- Check outcome ---
        if guess_word.lower() == word.lower():
            outcome = "success"
            break

        prev_guesses.append(guess_word)

    model_name = CONFIG["models"][model_key]
    return {
        "messages": final_messages,
        "meta": {
            "game": "taboo",
            "game_role": "WordGuesser",
            "outcome": outcome,
            "task_id": task_id,
            "model": model_name,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main():
    rng = random.Random(CONFIG["seed"])

    # Build generation plan: (word, task_id, model_key)
    model_keys = list(CONFIG["models"].keys())
    episodes_per_word = CONFIG["episodes_per_word"]
    eps_per_model = episodes_per_word // len(model_keys)  # 16 each
    remainder = episodes_per_word % len(model_keys)       # 2 extra

    plan: list[tuple[str, int, str]] = []
    for task_id, word in enumerate(ALL_WORDS):
        for i, mk in enumerate(model_keys):
            count = eps_per_model + (1 if i < remainder else 0)
            for _ in range(count):
                plan.append((word, task_id, mk))

    rng.shuffle(plan)
    print(f"Generation plan: {len(plan)} episodes across {len(ALL_WORDS)} words, {len(model_keys)} models")

    # Run generation
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(CONFIG["raw_output"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(plan)

    # Process in batches
    batch_size = CONFIG["concurrency_per_provider"] * len(model_keys)
    for start in range(0, total, batch_size):
        batch = plan[start : start + batch_size]
        tasks = [generate_episode(w, tid, mk) for w, tid, mk in batch]
        batch_results = await asyncio.gather(*tasks)

        for ep in batch_results:
            if ep is not None:
                results.append(ep)

        done = min(start + batch_size, total)
        successes = sum(1 for r in results if r["meta"]["outcome"] == "success")
        print(f"  Progress: {done}/{total} planned, {len(results)} succeeded, "
              f"success rate so far: {successes}/{len(results)}")

    # Save raw JSONL
    with open(raw_path, "w") as f:
        for ep in results:
            f.write(json.dumps(ep) + "\n")
    print(f"\nSaved raw JSONL: {raw_path} ({len(results)} episodes)")

    # Build HuggingFace Dataset
    ds = Dataset.from_list(results)
    ds.save_to_disk(str(output_dir))
    print(f"Saved HF dataset: {output_dir} ({len(ds)} rows)")

    # Summary stats
    print("\n--- Summary ---")
    model_counts: dict[str, dict[str, int]] = {}
    total_turns = 0
    for ep in results:
        model = ep["meta"]["model"]
        outcome = ep["meta"]["outcome"]
        model_counts.setdefault(model, {"success": 0, "failure": 0})
        model_counts[model][outcome] += 1
        total_turns += len(ep["messages"]) // 2

    for model, counts in sorted(model_counts.items()):
        total_m = counts["success"] + counts["failure"]
        rate = counts["success"] / total_m if total_m > 0 else 0
        print(f"  {model}: {total_m} episodes, "
              f"{counts['success']} success / {counts['failure']} failure "
              f"({rate:.1%})")
    print(f"  Total episodes: {len(results)}")
    print(f"  Avg turns per episode: {total_turns / len(results):.1f}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
