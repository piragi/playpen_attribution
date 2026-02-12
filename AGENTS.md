# Playpen Attribution Project Guide

## Goal
Build a reproducible pipeline for:
1. Training a LoRA SFT model on playpen game data.
2. Running gradient-based data attribution to identify high-impact training instances/words for learning behavior.
3. Using those insights to generate similarly hard words later (mechanistic-interpretability-driven iteration).

Current scope is step 1 + step 2.

## Tooling Rules
1. Always run Python and scripts with `uv` (for consistency and reproducibility).
2. Build attribution on top of the local Bergson library at `../bergson`.
3. Keep scripts simple and config-driven.

## Repository Workflow
1. Dataset prep:
   - Source: `colab-potsdam/playpen-data` (`interactions` config).
   - Filter to one game/role at a time.
   - Drop `aborted` outcomes.
2. SFT (LoRA):
   - Script: `data.py`.
   - Model: Gemma 3 1B Instruct with LoRA target modules.
   - Train/eval split comes from dataset `train` + `validation`.
3. Attribution (current target):
   - Use Bergson `reduce` + `score` flow for val-query vs train-index scoring.

## Current SFT Setup
Reference script:
- `data.py`

Behavior:
1. Filters train/eval with:
   - game match
   - role match
   - `outcome != "aborted"`
2. Trains LoRA adapters with `trl.SFTTrainer`.
3. Saves adapter to `./taboo_sft_lora`.
4. Uses prompt/completion conversion + `completion_only_loss=True` as a temporary workaround for Gemma chat-template masking limitations.

## Bergson Attribution Plan (Reduce + Score)

### Why this path
Bergson documents two main workflows:
1. Build full gradient index then query repeatedly.
2. Reduce query set + score a dataset against that query.

For this repo, we want one fixed query set (validation), so use reduce+score first.

References:
- `../bergson/README.md:52`
- `../bergson/docs/cli.rst:9`

### Math and semantics
Let:
- `g_i` = gradient vector for training item `i`.
- `q_j` = gradient vector for query item `j` (validation item).

`score=mean`:
1. Compute query aggregate:
   - `q_mean = (1 / M) * sum_j q_j` (with Bergson preprocessing/normalization options).
2. Score each train item:
   - Base scoring is inner product between train and query vectors.
   - If unit-normalized, this becomes cosine-like similarity.
3. Output one scalar score per train item.

Preconditioning (important):
1. Bergson can transform query gradients with a damped inverse preconditioner:
   - `q_tilde = q_mean @ (H + lambda * I)^(-1)`
2. Then scores are computed as:
   - `s_i = g_i dot q_tilde`
3. If no preconditioner path is provided, this defaults to the identity transform
   (i.e. plain inner product / cosine-like scoring without Hessian correction).
4. Bergson also supports mixing query and index preconditioners before inversion.

References:
- Query gradient loading and preprocessing:
  - `../bergson/bergson/score/score.py:166`
  - `../bergson/bergson/score/score.py:355`
  - `../bergson/bergson/score/score.py:360`
- Preconditioning and damped inverse:
  - `../bergson/bergson/score/score.py:118`
  - `../bergson/bergson/score/score.py:152`
  - `../bergson/bergson/score/score.py:159`
- Scorer inner product / nearest:
  - `../bergson/bergson/score/scorer.py:72`
  - `../bergson/bergson/score/scorer.py:76`
  - `../bergson/bergson/score/scorer.py:80`

### What `reduce` is doing
`reduce` aggregates per-item gradients into one gradient vector (sum/mean), optionally with unit normalization.

References:
- `../bergson/bergson/reduce.py:148`
- `../bergson/bergson/data.py:413`
- `../bergson/bergson/data.py:466`

### LoRA compatibility
Bergson can detect and target PEFT modules when model path points to a LoRA adapter.

References:
- `../bergson/bergson/utils/worker_utils.py:139`
- `../bergson/bergson/utils/worker_utils.py:167`
- `../bergson/bergson/utils/peft.py:5`

## Masking Policy Status
Current policy:
1. SFT currently uses `completion_only_loss=True` on converted prompt/completion records in `data.py`.
2. Bergson tokenization masks non-assistant tokens (`-100`) for conversation/prompt-completion.
3. This is close in spirit (assistant-targeted), but not perfectly identical to multi-turn assistant-only masking.

Reference:
- `data.py`
- `../bergson/bergson/data.py:514`
- `../bergson/bergson/data.py:547`
- `../bergson/bergson/data.py:577`

Future option:
1. If we later want all-token attribution, add an explicit all-token label mode in Bergson tokenization and wire it through reduce/score configs.

## Current Workarounds (Required)
### 1) Adapter config bootstrap workaround
Issue:
1. Bergson data setup calls `AutoConfig.from_pretrained(cfg.model)` to infer max length.
2. A LoRA adapter directory may not include a `config.json`, which breaks this call.

Workaround:
1. Ensure adapter directory contains base model config before reduce/score.
2. In code, this is:
   - `AutoConfig.from_pretrained(base_model).save_pretrained(adapter_path)`

Reference:
- `../bergson/bergson/utils/worker_utils.py:301`
- `score_samples.py:66`

### 2) Prompt/completion conversion workaround for template-span matching
Issue:
1. Bergson conversation tokenization finds assistant spans with string matching (`rfind`).
2. On some multi-turn playpen chats with repeated assistant strings, span matching can fail.

Workaround:
1. Convert each conversation into:
   - `prompt`: all turns before the final assistant turn
   - `completion`: content of the final assistant turn
2. Run Bergson with `prompt_column` + `completion_column` instead of `conversation_column`.

Reference:
- `../bergson/bergson/data.py:547`
- `../bergson/bergson/data.py:563`
- `score_samples.py:49`
- `score_samples.py:112`

### 3) SFT completion-only workaround for Gemma template masks
Issue:
1. `assistant_only_loss=True` in TRL requires chat-template assistant masks.
2. Gemma template in this setup does not expose masks via `{% generation %}` metadata.

Workaround:
1. Convert each example to plain `prompt` + `completion` text.
2. Drop raw `messages` before SFT preprocessing.
3. Train with `completion_only_loss=True`, `assistant_only_loss=False`.

Reference:
- `data.py:52`
- `data.py:73`
- `data.py:122`

## Next Investigation: Preconditioner Computation
Current status:
1. Attribution script runs with `skip_preconditioners=True` to avoid VRAM OOM in local setup.
2. This means current scores are unpreconditioned (identity transform).

Next step:
1. Profile and enable preconditioners safely (module subset, precision/batch strategy, or distributed preconditioner path), then compare ranking stability versus unpreconditioned scores.

Reference:
- `score_samples.py:111`
- `../bergson/bergson/collector/gradient_collectors.py:170`

## Practical Next Steps
1. Run SFT with current completion-only workaround (working baseline) and version artifacts.
2. Export/stabilize filtered train/val datasets with persistent row IDs.
3. Run Bergson reduce on val.
4. Run Bergson score on train with query path from reduce.
5. Load scores and report top-k / bottom-k influential instances with metadata (`meta`, messages, task IDs).
6. Investigate and enable preconditioners without OOM; compare preconditioned vs unpreconditioned rankings.
7. Optionally add a custom chat template with generation tags to restore strict `assistant_only_loss=True` path.

## Notes
1. Keep artifacts under run directories, not project root.
2. Prefer deterministic seeds for reproducibility.
3. Keep role-specific runs separate (`WordGuesser` vs `WordDescriber`) to avoid role mixing in attribution interpretation.
