"""Run the end-to-end attribution pipeline with simple sequential steps.

Default flow:
  1) build_sft_data.py
  2) finetune.py (base adapter)
  3) rebuild_attr_query.py
  4) score.py
  5) probe.py
  6) generate_continued_dataset.py
  7) finetune.py (all continuation arms)
  8) eval_harness.py (all continuation arms)

Usage:
    uv run run_experiments.py
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path


CONFIG = {
    "run_dir": "runs/smoltalk_v4",
    "python": sys.executable,
    "smoke_test": False,
    "query_category": None,  # None | "math" | "data-analysis" | "all"
    "score_output_dir": "runs/smoltalk_v4/scores_math_da",
    "run_eval": True,
    "eval_tasks": "arc_challenge,arc_easy,hellaswag,winogrande,ifeval,gsm8k",
    "eval_batch_size": "auto",
}


def run_command(cmd: list[str]) -> None:
    print(f"$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_config(cfg: dict) -> None:
    # rebuild_attr_query.py, probe.py, and generate_continued_dataset.py currently
    # use in-file CONFIG defaults, so this runner assumes the default run dir.
    if cfg["run_dir"] != "runs/smoltalk_v4":
        raise ValueError("run_dir must be 'runs/smoltalk_v4' with current script defaults.")


def step_build_sft_data(cfg: dict) -> None:
    cmd = [cfg["python"], "build_sft_data.py", "--output-dir", cfg["run_dir"]]
    if cfg["smoke_test"]:
        cmd.append("--smoke-test")
    run_command(cmd)


def step_finetune_base(cfg: dict, manifest: dict) -> None:
    run_dir = Path(cfg["run_dir"])
    run_command(
        [
            cfg["python"],
            "finetune.py",
            "--base-model",
            manifest["base_model"],
            "--train-data",
            manifest["splits"]["train"]["path"],
            "--val-data",
            manifest["splits"]["val"]["path"],
            "--output-dir",
            str(run_dir / "adapter"),
        ]
    )


def step_rebuild_attr_query(cfg: dict) -> None:
    cmd = [cfg["python"], "rebuild_attr_query.py"]
    if cfg["query_category"] is not None:
        cmd.extend(["--category", str(cfg["query_category"])])
    run_command(cmd)


def step_score(cfg: dict) -> None:
    run_dir = Path(cfg["run_dir"])
    run_command(
        [
            cfg["python"],
            "score.py",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--adapter-path",
            str(run_dir / "adapter"),
            "--output-dir",
            cfg["score_output_dir"],
        ]
    )


def step_probe(cfg: dict) -> None:
    run_command([cfg["python"], "probe.py"])


def step_generate_continued(cfg: dict) -> None:
    run_command([cfg["python"], "generate_continued_dataset.py"])


def step_finetune_arms(cfg: dict, manifest: dict) -> dict[str, str]:
    run_dir = Path(cfg["run_dir"])
    cont_manifest = load_json(run_dir / "continuation" / "continuation_manifest.json")
    adapter_by_arm: dict[str, str] = {}
    for arm_name, arm_info in cont_manifest["arms"].items():
        adapter_dir = run_dir / f"adapter_{arm_name}"
        run_command(
            [
                cfg["python"],
                "finetune.py",
                "--base-model",
                manifest["base_model"],
                "--train-data",
                arm_info["path"],
                "--val-data",
                manifest["splits"]["val"]["path"],
                "--output-dir",
                str(adapter_dir),
            ]
        )
        adapter_by_arm[arm_name] = str(adapter_dir)
    return adapter_by_arm


def step_eval_arms(cfg: dict, manifest: dict, adapter_by_arm: dict[str, str]) -> None:
    if not cfg["run_eval"]:
        return
    eval_dir = Path(cfg["run_dir"]) / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for arm_name, adapter_dir in adapter_by_arm.items():
        run_command(
            [
                cfg["python"],
                "eval_harness.py",
                "--base-model",
                manifest["base_model"],
                "--adapter-path",
                adapter_dir,
                "--tasks",
                cfg["eval_tasks"],
                "--apply-chat-template",
                "--batch-size",
                cfg["eval_batch_size"],
                "--output-json",
                str(eval_dir / f"{arm_name}.json"),
            ]
        )


def main() -> None:
    cfg = dict(CONFIG)
    validate_config(cfg)
    run_dir = Path(cfg["run_dir"])

    print("\n=== 1) build_sft_data ===")
    step_build_sft_data(cfg)
    manifest = load_json(run_dir / "manifest.json")

    print("\n=== 2) finetune base adapter ===")
    step_finetune_base(cfg, manifest)

    print("\n=== 3) rebuild_attr_query ===")
    step_rebuild_attr_query(cfg)

    print("\n=== 4) score ===")
    step_score(cfg)

    print("\n=== 5) probe ===")
    step_probe(cfg)

    print("\n=== 6) generate_continued_dataset ===")
    step_generate_continued(cfg)

    print("\n=== 7) finetune continuation arms ===")
    adapter_by_arm = step_finetune_arms(cfg, manifest)

    print("\n=== 8) eval continuation arms ===")
    step_eval_arms(cfg, manifest, adapter_by_arm)

    print("\nDone.")


if __name__ == "__main__":
    main()
