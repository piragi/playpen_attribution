# Playpen Attribution - Active Notes

Last updated: 2026-02-18

## 1) Overall Goal
- Test whether Bergson attribution can select training examples that improve downstream task learning better than matched random selection under comparable training budget.
- Current downstream task: word-level success on `taboo::WordGuesser`.

## 2) Current Scope
- Keep pipeline minimal:
  - `dataset.py`
  - `finetune.py`
  - `score.py`
  - `eval_words.py`
  - `sae_analysis.py`
  - `train_bidir_classifier.py`
- Use `uv run` for all project commands.

## 3) Critical Bergson Rule (Must Keep)
- Bergson will apply a chat template when using `prompt_column` + `completion_column` tokenization.
- That tokenization does not match our finetune objective and can distort attribution quality.
- For aligned attribution, pass token IDs directly (pretokenized samples with `input_ids` and `labels`), so Bergson does not retokenize through chat template.
- In this repo, this is the `score.py --tokenization-mode finetune_raw` path.

## 4) Best Attribution Setup So Far
- Best-performing configuration in current runs:
  - `score_mode=mean`
  - `preconditioning_mode=query`
  - `projection_dim=32`
  - `tokenization_mode=finetune_raw`
- `nearest` was consistently weaker or less reliable than `mean` in this setup.

## 5) Data Scale Finding
- Main visible gain is from scale (`k=500` clearly better than `k=300`).
- But the added block (`ranks 301-500`) is not random filler:
  - `ranks 301-500` beat matched `random200` in the dedicated ablation.

## 6) Key Results (Single Seed, seed=42)
- Base model:
  - `22/152 = 0.1447`

- Mean scoring:
  - `k=300`, no overlap:
    - top: `52/152 = 0.3421`
    - random: `54/152 = 0.3553`
  - `k=500`, no overlap:
    - top: `62/152 = 0.4079`
    - random: `43/152 = 0.2829`
  - `k=500`, overlap allowed:
    - top: `67/152 = 0.4408`
    - random: `58/152 = 0.3816`

- Nearest scoring:
  - `k=500`, no overlap:
    - top: `54/152 = 0.3553`
    - random: `52/152 = 0.3421`
  - `k=500`, overlap allowed:
    - top: `52/152 = 0.3421`
    - random: `60/152 = 0.3947`

## 7) Scale vs Rank-Window Ablation
- Built from one fixed mean ranking (`k=500` run):
  - `top300_from_mean500`: `53/152 = 0.3487`
  - `top500_from_mean500`: `62/152 = 0.4079`
  - `tail200_rank301_500`: `55/152 = 0.3618`
  - `rand200_from_rank500plus`: `49/152 = 0.3224`
- Interpretation:
  - Scale helps strongly (`300 -> 500`).
  - Added ranks `301-500` also carry meaningful signal over random at same size.

## 8) Current Conclusion (Provisional)
- Use mean attribution with finetune-aligned token IDs as the active default.
- `k=500` is currently the minimum size where a strong top-vs-random difference is visible in this seed.
- Results are promising but still provisional until multi-seed replication.

## 8.1) Future Caveat
- Dataset size may still be too small for the strength of claim we want; weak or unstable top-vs-random gaps can come from limited sample count rather than true absence of attribution signal.

## 9) Canonical Artifacts
- Main manifest:
  - `runs/simple_wordguesser_v1/manifest.json`
- Best mean run (`k=500`):
  - `runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/`
- Scale/rank ablation manifest:
  - `runs/simple_wordguesser_v1/ablation_scale_from_mean500/manifest.json`
- Evaluation outputs:
  - `runs/simple_wordguesser_v1/word_eval_compare_finetune_aligned_k500.json`
  - `runs/simple_wordguesser_v1/word_eval_compare_mean_finetune_aligned_k500_overlap.json`
  - `runs/simple_wordguesser_v1/word_eval_compare_nearest_finetune_aligned_k500_nooverlap_v2.json`
  - `runs/simple_wordguesser_v1/word_eval_compare_nearest_finetune_aligned_k500_overlap.json`
  - `runs/simple_wordguesser_v1/word_eval_compare_top500_vs_top300_from_mean500.json`
  - `runs/simple_wordguesser_v1/word_eval_compare_tail200_vs_rand200_from_mean500.json`
- Restored component smoke outputs:
  - `runs/simple_wordguesser_v1/bidir_classifier_smoke_2026_02_17/summary.json`
  - `runs/simple_wordguesser_v1/bidir_classifier_smoke_2026_02_17/row_scores.jsonl`
  - `runs/simple_wordguesser_v1/sae_samples_layer17_train_base_smoke_2026_02_17/summary.json`
  - `runs/simple_wordguesser_v1/sae_samples_layer17_train_base_smoke_2026_02_17/layer_17_width_16k_l0_small.npz`

## 10) Restored Components (2026-02-17)
- Brought back and adapted to current format:
  - `train_bidir_classifier.py` (bidirectional BERT detector)
  - `sae_analysis.py` (per-sample SAE extraction)
- Compatibility changes applied:
  - Removed dependency on deleted `prompts.py`.
  - Uses current row schema (`prompt`, `completion`) with `messages` fallback.
  - Uses manifest split paths from `runs/simple_wordguesser_v1/manifest.json`.
  - Uses current attribution score artifacts from `score.py` runs.
- Smoke run status:
  - `train_bidir_classifier.py`: successful.
    - run output root: `runs/simple_wordguesser_v1/bidir_classifier_smoke_2026_02_17`
    - test AUC: `0.5648`
    - top-k overlap vs attribution labels: `17/64` (`0.2656`)
  - `sae_analysis.py`: successful.
    - run output root: `runs/simple_wordguesser_v1/sae_samples_layer17_train_base_smoke_2026_02_17`
    - `n_examples=1`, `total_tokens=128`, hook=`blocks.17.hook_resid_post`

## 10.1) Top-500 SAE + Bidir Run (2026-02-17)
- Requested setup run on `top_k` 500 subset:
  - SAE extraction over all 500 rows.
  - Bidirectional BERT training/eval on same 500 rows with 80/20 split.
- Supporting scoring run for aligned labels:
  - `runs/simple_wordguesser_v1/attribution_top500_for_bidir/summary.json`
  - finite score coverage: `500/500`
- SAE output:
  - `runs/simple_wordguesser_v1/sae_samples_layer17_top500_2026_02_17/summary.json`
  - `n_examples=500`
  - `total_tokens=73154`
  - `hook_name=blocks.17.hook_resid_post`
- Bidir classifier output:
  - `runs/simple_wordguesser_v1/bidir_classifier_top500_2026_02_17/summary.json`
  - split mode: `stratified`
  - train/test sizes: `400/100` (80/20)
  - labels: `n_positive_labels=100`
  - test AUC: `0.6050`
  - top-k overlap vs attribution labels: `48/100` (`0.4800`)

## 11) Active Commands
```bash
uv run dataset.py

uv run finetune.py train \
  --manifest runs/simple_wordguesser_v1/manifest.json \
  --train-split train_base \
  --eval-split eval \
  --output-dir runs/simple_wordguesser_v1/base_adapter

uv run score.py \
  --manifest runs/simple_wordguesser_v1/manifest.json \
  --adapter-path runs/simple_wordguesser_v1/base_adapter \
  --score-mode mean \
  --preconditioning-mode query \
  --tokenization-mode finetune_raw \
  --subset-k 500

uv run finetune.py train \
  --manifest runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/continuation_manifest.json \
  --train-split top_k \
  --eval-split eval \
  --output-dir runs/simple_wordguesser_v1/scratch_top_k_mean_finetune_aligned_k500

uv run finetune.py train \
  --manifest runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/continuation_manifest.json \
  --train-split matched_random_k \
  --eval-split eval \
  --output-dir runs/simple_wordguesser_v1/scratch_random_k_mean_finetune_aligned_k500

uv run eval_words.py \
  --manifest runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/continuation_manifest.json \
  --eval-split eval \
  --top-adapter runs/simple_wordguesser_v1/scratch_top_k_mean_finetune_aligned_k500 \
  --random-adapter runs/simple_wordguesser_v1/scratch_random_k_mean_finetune_aligned_k500 \
  --output-json runs/simple_wordguesser_v1/word_eval_compare_finetune_aligned_k500.json

uv run train_bidir_classifier.py \
  --manifest runs/simple_wordguesser_v1/manifest.json \
  --label-split score_pool \
  --score-split score_pool \
  --score-run-path runs/simple_wordguesser_v1/attribution_mean_finetune_aligned_k500/train_scores \
  --output-root runs/simple_wordguesser_v1/bidir_classifier_smoke_2026_02_17 \
  --max-rows 256 \
  --score-max-rows 256 \
  --top-k 64 \
  --split-mode grouped \
  --num-train-epochs 1

uv run sae_analysis.py \
  --manifest runs/simple_wordguesser_v1/manifest.json \
  --split train_base \
  --output-dir runs/simple_wordguesser_v1/sae_samples_layer17_train_base_smoke_2026_02_17 \
  --max-examples 1 \
  --max-tokens 128 \
  --sae-id layer_17_width_16k_l0_small \
  --sae-release gemma-scope-2-1b-it-res \
  --model-name google/gemma-3-1b-it
```

## 12) Synthetic Generation Status (2026-02-18)
- `generate_synthetic_openrouter.py` was simplified to a single-file lean generator while preserving:
  - strict taboo rule checks (target/related words + stems),
  - describer repair loop and guesser empty/placeholder repair loop,
  - checkpoint-safe writes to `games.jsonl`, `rows_wordguesser.jsonl`, `summary.json`,
  - optional `--save-hf-dataset`.
- Current generator size: `486` lines.
- Removed non-essential complexity:
  - OpenRouter pricing fetch/breakdown logic,
  - backward-compat hidden CLI args,
  - unused target-source mode (`train_completions`),
  - bulky per-call usage/cost breakdowns in summary.
- Format compatibility notes:
  - `rows_wordguesser.jsonl` keys match previous runs.
  - `games.jsonl` keeps the fields used in current workflow, including `game`, `game_role`, and `max_turns`.

## 12.1) Synthetic Quality Smoke vs Baseline
- Baseline run:
  - `runs/synthetic_data_row500_qwen_desc_gemma_2026_02_18_v1`
  - success: `303/501 = 0.6048`
  - mean turns: `2.0659`
- Refactored script quality smoke:
  - `runs/synthetic_quality_eval_30_refactor_2026_02_18`
  - setup matched baseline family:
    - describer: `qwen/qwen3-30b-a3b-instruct-2507`
    - guessers: `allenai/olmo-3.1-32b-instruct`, `mistralai/ministral-14b-2512`, `google/gemma-3-27b-it`
    - `games-per-guesser=10` (30 total)
  - success: `21/30 = 0.7000`
  - mean turns: `1.9000`
  - zero empty clue turns and zero placeholder guess rows in this smoke run.

## 12.2) Current Recommended Synthetic Command
```bash
uv run generate_synthetic_openrouter.py \
  --output-dir runs/synthetic_data_row500_qwen_desc_gemma_2026_02_18_v2 \
  --describer-model qwen/qwen3-30b-a3b-instruct-2507 \
  --guesser-models allenai/olmo-3.1-32b-instruct mistralai/ministral-14b-2512 google/gemma-3-27b-it \
  --games-per-guesser 167 \
  --max-turns 3 \
  --seed 42 \
  --checkpoint-every 1
```
