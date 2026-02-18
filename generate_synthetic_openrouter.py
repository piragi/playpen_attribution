from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI

_GUESS_RE = re.compile(r"GUESS\s*:\s*([^\n\r]+)", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
DEFAULT_TABOO_INSTANCES_URL = "https://raw.githubusercontent.com/clp-research/clembench/main/taboo/in/instances.json"
DEFAULT_PLACEHOLDER_GUESS_WORDS = ("guess",)
OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/piragi/playpen_attribution", "X-Title": "playpen_attribution_synth"}
BASE_GAME_INSTRUCTION = "You are playing a collaborative word guessing game in which you have to guess a target word that another player describes to you.\n\nYou can make one guess at each trial. You will have up to {max_turns} tries.\n\nAfter each trial you will get a new hint from the other player which starts with CLUE.\n\nMake your guesses by just saying the word using the following form: GUESS: <a word>\n\nLet us start."
DESCRIBER_SYSTEM = "You are the clue giver in a taboo-style word guessing game.\nReturn exactly one line using this format: CLUE: <short clue>\nRules:\n- Do not use the target word.\n- Do not use obvious inflections/variants of the target word.\n- Do not reveal spelling, initials, letter counts, or rhymes.\n- Keep the clue concise and concrete."
GUESSER_SYSTEM = "You are the guesser in a collaborative word game.\nReturn exactly one line in this format: GUESS: <a word>\nDo not add any explanation."


def resolve_api_key(env_file: Path) -> str:
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if not k:
                continue
            if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
                v = v[1:-1]
            os.environ.setdefault(k, v)
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API")
    if key:
        return key
    raise RuntimeError("Missing OPENROUTER_API_KEY/OPENROUTER_API in env or .env")


def extract_guess_word(text: str) -> str:
    m = _GUESS_RE.search(text)
    if m:
        candidate = m.group(1).strip()
    else:
        stripped = text.strip()
        if not stripped:
            return ""
        candidate = stripped.splitlines()[0].strip()
        if candidate.upper().startswith("GUESS") and ":" in candidate:
            candidate = candidate.split(":", 1)[1].strip()
    if not candidate:
        return ""
    wm = _WORD_RE.search(candidate.lower())
    return wm.group(0) if wm else ""


def normalize_prefixed_line(text: str, prefix: str) -> str:
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first:
        return prefix
    if first.upper().startswith(prefix.upper()):
        body = first.split(":", 1)[1].strip() if ":" in first else ""
        return f"{prefix} {body}".strip()
    body = first
    if ":" in body and body.split(":", 1)[0].strip().isalpha():
        body = body.split(":", 1)[1].strip()
    return f"{prefix} {body}".strip()


def load_instances(source: str, timeout_seconds: float) -> dict[str, list[dict[str, Any]]]:
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, method="GET", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch JSON from URL: {source}: {exc}") from exc
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"JSON path not found: {path}")
        obj = json.loads(path.read_text())

    exps = obj.get("experiments")
    if not isinstance(exps, list):
        raise RuntimeError(f"Unexpected instances JSON shape at {source}")

    out: dict[str, list[dict[str, Any]]] = {}
    for exp in exps:
        if not isinstance(exp, dict):
            continue
        name = str(exp.get("name", "")).strip()
        rows: list[dict[str, Any]] = []
        for inst in exp.get("game_instances") or []:
            if not isinstance(inst, dict):
                continue
            target_word = str(inst.get("target_word", "")).strip().lower()
            if not target_word:
                continue
            try:
                source_task_id = int(inst.get("game_id"))
            except Exception:
                continue
            rows.append(
                {
                    "source_task_id": source_task_id,
                    "target_word": target_word,
                    "related_words": [str(x).strip().lower() for x in (inst.get("related_word") or []) if str(x).strip()],
                    "target_stem": str(inst.get("target_word_stem", "")).strip().lower(),
                    "related_stems": [str(x).strip().lower() for x in (inst.get("related_word_stem") or []) if str(x).strip()],
                }
            )
        if name and rows:
            out[name] = rows
    if not out:
        raise RuntimeError(f"No valid instances parsed from {source}")
    return out


def call_chat(client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float, max_output_tokens: int, retries: int, retry_sleep_seconds: float) -> tuple[str, int, int]:
    attempt = 0
    while True:
        attempt += 1
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                extra_headers=OPENROUTER_HEADERS,
            )
            content = rsp.choices[0].message.content
            if isinstance(content, list):
                bits: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        piece = part.get("text") or part.get("content")
                    else:
                        piece = getattr(part, "text", None) or str(part)
                    if piece:
                        bits.append(str(piece))
                text = "".join(bits).strip()
            else:
                text = str(content or "").strip()
            usage = rsp.usage
            return text, int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(retry_sleep_seconds * attempt)


def play_game(args: argparse.Namespace, client: OpenAI, placeholder_words: set[str], game_id_num: int, task_id: int, guesser_model: str, experiment: str, target_item: dict[str, Any], quiet: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    game_id = f"synthetic-{game_id_num:05d}"
    target_word = str(target_item["target_word"]).lower()
    related_words = [str(x).lower() for x in target_item.get("related_words") or []]
    target_stem = str(target_item.get("target_stem") or "").lower()
    related_stems = [str(x).lower() for x in target_item.get("related_stems") or []]
    source_task_id = target_item.get("source_task_id")

    turns: list[dict[str, str]] = []
    final_prompt = ""
    final_completion = ""
    final_turn = 0
    outcome = "failure"
    error: str | None = None
    ptok_total = 0
    ctok_total = 0

    def game_prompt(clue: str) -> str:
        parts = [f"USER: {BASE_GAME_INSTRUCTION.format(max_turns=args.max_turns)}"]
        for t in turns:
            parts += [f"USER: CLUE: {t['clue']}", f"ASSISTANT: {t['guess']}"]
        parts.append(f"USER: CLUE: {clue}")
        return "\n\n".join(parts).strip()

    def describer_messages(turn: int) -> list[dict[str, str]]:
        hist = "\n".join([f"- Turn {t['turn']} clue: {t['clue']}\n- Turn {t['turn']} guess: {t['guess']}" for t in turns]) or "- (none)"
        rel = ", ".join(related_words) if related_words else "(none)"
        user = (
            f"Target word: {target_word}\n"
            f"Related forbidden words: {rel}\n"
            f"Turn: {turn}/{args.max_turns}\n"
            f"Previous clues/guesses:\n{hist}\n\n"
            "Generate the next clue now."
        )
        return [{"role": "system", "content": DESCRIBER_SYSTEM}, {"role": "user", "content": user}]

    def violations(clue_text: str) -> list[str]:
        toks = [tok.lower() for tok in _WORD_RE.findall(clue_text.lower())]
        if not toks:
            return []
        tset = set(toks)
        out: list[str] = []
        for w in [target_word.lower(), *[x.lower() for x in related_words]]:
            if w and w in tset:
                out.append(f"word:{w}")
        for stem in [target_stem.lower(), *[x.lower() for x in related_stems]]:
            if stem and len(stem) >= 4 and any(tok.startswith(stem) for tok in toks):
                out.append(f"stem:{stem}")
        return list(dict.fromkeys(out))

    if not quiet:
        print(f"\n[{game_id}] exp={experiment} target={target_word} guesser={guesser_model}")

    try:
        for turn in range(1, args.max_turns + 1):
            d_msgs = describer_messages(turn)
            d_text, ptok, ctok = call_chat(client, args.describer_model, d_msgs, args.describer_temperature, args.describer_max_output_tokens, args.retries, args.retry_sleep_seconds)
            ptok_total += ptok
            ctok_total += ctok

            clue = normalize_prefixed_line(d_text, "CLUE:")
            clue_body = clue.split(":", 1)[1].strip() if clue.upper().startswith("CLUE") and ":" in clue else clue.strip()
            v = violations(clue_body)
            empty = not bool(_WORD_RE.search(clue_body.lower()))
            if empty or (v and args.strict_taboo_rules):
                for _ in range(args.describer_repair_attempts):
                    labels = list(dict.fromkeys(([] if not empty else ["empty_clue"]) + v))
                    rd_msgs = [
                        *d_msgs,
                        {"role": "assistant", "content": clue},
                        {"role": "user", "content": (
                            "Invalid clue. It was empty and/or violated taboo rules.\n"
                            f"Violations: {', '.join(labels)}\n"
                            "Return a new clue with the same format: CLUE: <short clue>"
                        )},
                    ]
                    rd_text, ptok, ctok = call_chat(client, args.describer_model, rd_msgs, max(0.2, args.describer_temperature), args.describer_max_output_tokens, args.retries, args.retry_sleep_seconds)
                    ptok_total += ptok
                    ctok_total += ctok
                    clue = normalize_prefixed_line(rd_text, "CLUE:")
                    clue_body = clue.split(":", 1)[1].strip() if clue.upper().startswith("CLUE") and ":" in clue else clue.strip()
                    v = violations(clue_body)
                    empty = not bool(_WORD_RE.search(clue_body.lower()))
                    if not empty and (not v or not args.strict_taboo_rules):
                        break
            if empty:
                error = "describer_empty_clue"
                break
            if v and args.strict_taboo_rules:
                error = f"describer_rule_violation:{','.join(v)}"
                break

            prompt = game_prompt(clue_body)
            g_msgs = [{"role": "system", "content": GUESSER_SYSTEM}, {"role": "user", "content": prompt}]
            g_text, ptok, ctok = call_chat(client, guesser_model, g_msgs, args.guesser_temperature, args.guesser_max_output_tokens, args.retries, args.retry_sleep_seconds)
            ptok_total += ptok
            ctok_total += ctok

            guess = normalize_prefixed_line(g_text, "GUESS:")
            guess_word = extract_guess_word(guess)
            placeholder = bool(guess_word) and guess_word in placeholder_words and guess_word != target_word
            invalid = (not guess_word) or placeholder
            if invalid and args.empty_guess_retry_attempts > 0:
                for _ in range(args.empty_guess_retry_attempts):
                    issue = "it did not provide a guess word" if not guess_word else f"it used the placeholder guess '{guess_word}'"
                    rg_msgs = [
                        {"role": "system", "content": GUESSER_SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": guess},
                        {"role": "user", "content": (
                            f"Your last reply was invalid because {issue}.\n"
                            "Reply with exactly one non-empty guess in this form: GUESS: <word>"
                        )},
                    ]
                    rg_text, ptok, ctok = call_chat(client, guesser_model, rg_msgs, 0.0, args.guesser_max_output_tokens, args.retries, args.retry_sleep_seconds)
                    ptok_total += ptok
                    ctok_total += ctok
                    guess = normalize_prefixed_line(rg_text, "GUESS:")
                    guess_word = extract_guess_word(guess)
                    placeholder = bool(guess_word) and guess_word in placeholder_words and guess_word != target_word
                    invalid = (not guess_word) or placeholder
                    if not invalid:
                        break
            if invalid:
                error = "guesser_empty_guess" if not guess_word else f"guesser_placeholder_guess:{guess_word}"
                break

            turns.append({"turn": str(turn), "clue": clue_body, "clue_violations": v, "guess": guess, "guess_word": guess_word})
            final_prompt, final_completion, final_turn = prompt, guess, turn

            if args.sleep_between_calls_seconds > 0:
                time.sleep(args.sleep_between_calls_seconds)
            if guess_word == target_word:
                outcome = "success"
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outcome = "aborted"

    game = {
        "game_id": game_id, "game": "taboo", "game_role": "WordGuesser", "max_turns": args.max_turns,
        "describer_model": args.describer_model, "guesser_model": guesser_model,
        "experiment": experiment, "task_id": task_id, "task_label": f"synthetic_{target_word}",
        "source_task_id": source_task_id, "target_word": target_word, "related_words": related_words,
        "turns": turns, "final_turn": final_turn, "outcome": outcome, "prompt": final_prompt,
        "completion": final_completion, "usage": {"total_prompt_tokens": ptok_total, "total_completion_tokens": ctok_total},
        "error": error,
    }

    row = None
    final_guess_word = extract_guess_word(final_completion)
    if final_prompt and final_completion and final_guess_word:
        row = {
            "row_id": game_id, "prompt": final_prompt, "completion": final_completion, "source_split": "synthetic_openrouter",
            "game": "taboo", "game_role": "WordGuesser", "pair_key": "taboo::WordGuesser", "outcome": outcome,
            "experiment": experiment, "task_id": task_id, "task_label": f"synthetic_{target_word}",
            "source_task_id": source_task_id, "group_id": f"{experiment}::{task_id}",
            "describer_model": args.describer_model, "guesser_model": guesser_model, "target_word": target_word,
            "related_words": related_words, "turn_count": final_turn, "final_guess_word": final_guess_word,
            "generation_prompt_tokens": ptok_total, "generation_completion_tokens": ctok_total, "generation_cost_usd": 0.0,
            "messages": [{"role": "user", "content": final_prompt}, {"role": "assistant", "content": final_completion}],
        }

    if not quiet:
        if outcome == "aborted":
            print(f"  outcome={outcome} error={error}")
        else:
            print(f"  outcome={outcome} turns={final_turn} guess={extract_guess_word(final_completion)!r}")
    return game, row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic Taboo WordGuesser data via OpenRouter.")
    p.add_argument("--instances-json", type=str, default=DEFAULT_TABOO_INSTANCES_URL)
    p.add_argument("--env-file", type=str, default=".env")
    p.add_argument("--output-dir", type=str, default=f"runs/simple_wordguesser_v1/synthetic_openrouter_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    p.add_argument("--describer-model", type=str, default="qwen/qwen3-30b-a3b-instruct-2507")
    p.add_argument("--guesser-models", type=str, nargs="+", default=["allenai/olmo-3.1-32b-instruct", "mistralai/ministral-14b-2512", "google/gemma-3-27b-it"])
    p.add_argument("--games-per-guesser", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--describer-temperature", type=float, default=0.7)
    p.add_argument("--guesser-temperature", type=float, default=0.2)
    p.add_argument("--describer-max-output-tokens", type=int, default=40)
    p.add_argument("--guesser-max-output-tokens", type=int, default=16)
    p.add_argument("--describer-repair-attempts", type=int, default=2)
    p.add_argument("--timeout-seconds", type=float, default=90.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep-seconds", type=float, default=1.5)
    p.add_argument("--empty-guess-retry-attempts", type=int, default=1)
    p.add_argument("--placeholder-guess-words", type=str, nargs="+", default=list(DEFAULT_PLACEHOLDER_GUESS_WORDS))
    p.add_argument("--sleep-between-calls-seconds", type=float, default=0.0)
    p.add_argument("--strict-taboo-rules", dest="strict_taboo_rules", action="store_true")
    p.add_argument("--no-strict-taboo-rules", dest="strict_taboo_rules", action="store_false")
    p.set_defaults(strict_taboo_rules=True)
    p.add_argument("--save-hf-dataset", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    games_path = output_dir / "games.jsonl"
    rows_path = output_dir / "rows_wordguesser.jsonl"
    summary_path = output_dir / "summary.json"
    hf_path = output_dir / "rows_wordguesser_hf" if args.save_hf_dataset else None

    client = OpenAI(api_key=resolve_api_key(Path(args.env_file)), base_url="https://openrouter.ai/api/v1", timeout=args.timeout_seconds)
    instances = load_instances(args.instances_json, args.timeout_seconds)
    placeholder_words = {w.strip().lower() for w in args.placeholder_guess_words if w.strip()}

    order = [x for x in ("high_en", "medium_en", "low_en") if x in instances] or sorted(instances.keys())
    state = {}
    for k, v in instances.items():
        rows = list(v)
        rng.shuffle(rows)
        state[k] = {"rows": rows, "idx": 0}

    def sample_target(exp: str) -> dict[str, Any]:
        s = state[exp]
        rows = s["rows"]
        idx = int(s["idx"])
        if idx >= len(rows):
            rng.shuffle(rows)
            idx = 0
        s["idx"] = idx + 1
        return dict(rows[idx])

    schedule: list[tuple[int, int, str, str, dict[str, Any]]] = []
    counter = 0
    for gi, gm in enumerate(args.guesser_models):
        for li in range(args.games_per_guesser):
            exp = order[(gi + li) % len(order)]
            counter += 1
            schedule.append((counter, 1_000_000 + counter, gm, exp, sample_target(exp)))
    source_stats = {k: len(v) for k, v in instances.items()}

    if not args.quiet:
        print(f"output_dir: {output_dir}")
        print(f"games scheduled: {len(schedule)}")
        print(f"describer: {args.describer_model}")
        print(f"guessers: {args.guesser_models}")

    games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    fatal_error: str | None = None

    def checkpoint(partial: bool, fatal: str | None) -> dict[str, Any]:
        outcome_counts = Counter(g["outcome"] for g in games)
        finished = [g for g in games if g["outcome"] in {"success", "failure"}]
        success = sum(1 for g in finished if g["outcome"] == "success")
        mean_turns = float(sum(int(g.get("final_turn", 0)) for g in finished) / len(finished)) if finished else 0.0
        ptoks = int(sum(int(g["usage"]["total_prompt_tokens"]) for g in games))
        ctoks = int(sum(int(g["usage"]["total_completion_tokens"]) for g in games))
        payload = {
            "run_config": {
                "instances_json": args.instances_json, "describer_model": args.describer_model,
                "guesser_models": args.guesser_models, "placeholder_guess_words": args.placeholder_guess_words,
                "games_per_guesser": args.games_per_guesser, "max_turns": args.max_turns, "seed": args.seed,
                "checkpoint_every": args.checkpoint_every, "strict_taboo_rules": bool(args.strict_taboo_rules),
            },
            "files": {"games_jsonl": str(games_path), "rows_jsonl": str(rows_path), "summary_json": str(summary_path), "hf_rows_path": str(hf_path) if hf_path and hf_path.exists() else None},
            "results": {
                "games_requested": len(schedule), "games_written": len(games), "rows_written": len(rows),
                "outcome_counts": dict(outcome_counts), "finished_games": len(finished),
                "success_rate_finished_only": float(success / len(finished)) if finished else 0.0,
                "mean_turns_finished_only": mean_turns, "is_partial": partial, "fatal_error": fatal,
            },
            "usage": {"total_prompt_tokens": ptoks, "total_completion_tokens": ctoks},
            "word_source_stats": source_stats,
        }
        summary_path.write_text(json.dumps(payload, indent=2))
        return payload

    with games_path.open("w") as gf, rows_path.open("w") as rf:
        try:
            for game_id_num, task_id, gm, exp, target in schedule:
                game, row = play_game(args, client, placeholder_words, game_id_num, task_id, gm, exp, target, args.quiet)
                games.append(game)
                gf.write(json.dumps(game) + "\n")
                gf.flush()
                if row is not None:
                    rows.append(row)
                    rf.write(json.dumps(row) + "\n")
                    rf.flush()
                if args.checkpoint_every > 0 and (len(games) % args.checkpoint_every == 0):
                    checkpoint(True, None)
        except KeyboardInterrupt:
            fatal_error = "KeyboardInterrupt"
            if not args.quiet:
                print("\nInterrupted by user. Preserving partial outputs.")
        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            if not args.quiet:
                print(f"\nFatal error: {fatal_error}")
        finally:
            checkpoint(len(games) < len(schedule), fatal_error)

    if args.save_hf_dataset and rows:
        assert hf_path is not None
        if hf_path.exists():
            if hf_path.is_dir():
                shutil.rmtree(hf_path)
            else:
                hf_path.unlink()
        Dataset.from_list(rows).save_to_disk(str(hf_path))

    final = checkpoint(len(games) < len(schedule), fatal_error)
    finished = int(final["results"]["finished_games"])
    success = int(final["results"]["outcome_counts"].get("success", 0))

    print("\nDone.")
    print(f"games jsonl: {games_path}")
    print(f"rows  jsonl: {rows_path}")
    print(f"summary:     {summary_path}")
    if finished:
        print(f"success rate (finished only): {success}/{finished} = {success / finished:.4f}")


if __name__ == "__main__":
    main()
