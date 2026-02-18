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
  - `train_bidir_classifier.py` (trimmed SAE-only score/rank predictor)
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
- Historical note: this section refers to the older bidirectional BERT classifier path.
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

## 10.2) SAE Predictor Refactor (2026-02-18)
- `train_bidir_classifier.py` was simplified to a compact SAE-only predictor.
- Removed:
  - text/BERT path,
  - hybrid text+SAE path,
  - top-vs-rest binary label objective.
- Active predictor objective:
  - regress attribution `score` or `rank` from SAE features.
- Active feature set:
  - `sample_stats` + per-feature activation strength (`feature_activation_mean`).
  - feature-presence-only path removed.
- `sae_analysis.py` now writes `feature_activation_mean` to NPZ artifacts.
- Compatibility caveat:
  - Older SAE NPZ files without `feature_activation_mean` are incompatible with default predictor settings and must be regenerated.
- Current default recommendation:
  - `target-type=rank`
  - hard-locked `stats+activation` features in predictor
  - grouped split for evaluation.

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
  --manifest runs/exp_halftrain_synthmix_2026_02_18/manifest_half_train.json \
  --split score_pool \
  --score-run-path runs/exp_halftrain_synthmix_2026_02_18/attribution_mean_half_train_k200_codex/train_scores \
  --sae-dir runs/exp_halftrain_synthmix_2026_02_18/sae_samples_layer17_score_pool_activation_codex \
  --target-type rank \
  --top-k 200 \
  --split-mode grouped \
  --output-root runs/exp_halftrain_synthmix_2026_02_18/regressor_sae_rank_full

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
  --games-per-guesser 333 \
  --max-turns 3 \
  --seed 42 \
  --checkpoint-every 1
```

## 13) SmolTalk Experiment — Next Phase

### 13.1) Motivation: Why Move Beyond Taboo
- The Taboo WordGuesser dataset is too small (~500 training rows) for robust attribution + SAE claims.
- Weak or unstable top-vs-random gaps (section 8.1) may come from limited sample count rather than absence of signal.
- Need a larger dataset with natural informativeness variance: some examples should be genuinely more instructive than others.

### 13.2) Why SmolTalk / smol-magpie-ultra
- **Dataset**: `HuggingFaceTB/smoltalk`, subset `smol-magpie-ultra` (431K samples).
- **Paper**: [SmolLM2: When Smol Goes Big](https://arxiv.org/abs/2502.02737) (Allal et al., 2025).
- **Designed for small models**: SmolLM2-Instruct (1.7B) was trained on this data. Gemma-3-1B-it is in the same capacity range. Not saturated in pretraining unlike GSM8K.
- **Built-in informativeness metadata** per example:
  - `category`: reasoning, math, coding, data-analysis, editing, creative-writing, role-playing, etc.
  - `difficulty`: very easy / easy / medium / hard
  - `quality`: poor / average / good / excellent
  - `reward_model_score`: continuous ArmoRM score
  - `conversation_tokens`: token count
- These metadata fields serve as **validation signals** — we can check whether our SAE fingerprint captures something the reward model score does not.
- **Multi-dimensional informativeness**: spans instruction following, reasoning, math, code, rewriting, summarization. Some examples teach capabilities that transfer broadly, others are narrow. The SAE should capture this diversity.
- **Synthetic data story**: SmolTalk is itself synthetic (generated via Magpie pipeline from Llama-3.1-405B-Instruct). Understanding which synthetic examples actually move the needle is exactly the question this project tries to answer.

### 13.3) Inspiration Paper Connection
- Rathi & Radford ([arxiv:2601.21571](https://arxiv.org/abs/2601.21571)) use SAEs to identify tokens tied to specific capabilities, then **filter them out** of pretraining data to prevent those capabilities from forming.
- This project **inverts that direction**: use SAEs to identify and **select in** the most beneficial training examples to amplify learning.
- SmolLM2 paper supports this: their quality-filtered FineMath4+ achieved 2x improvement on GSM8K over broader data, demonstrating that a small high-quality subset dramatically outperforms quantity.

### 13.4) Experimental Design
- **Training pool**: ~8K examples from `smol-magpie-ultra` train split, stratified by `category` to maintain diversity, keeping all quality/difficulty levels (the variance is the signal).
- **Query set**: ~500 examples from `smol-magpie-ultra` test split (for Bergson attribution gradient computation).
- **Eval set**: ~500 held-out examples from test split (for end-to-end performance measurement).
- **Token cap**: ~2K tokens per example to keep SAE extraction tractable.
- **Do not filter by quality or difficulty** — that variance is what we want attribution and SAE to discover.

### 13.5) Pipeline Steps
1. **Build dataset manifest**: Sample ~8K train + 500 query + 500 eval from smol-magpie-ultra. Adapt `dataset.py` or write new loader for SmolTalk format (`messages` column, multi-turn).
2. **Flatten multi-turn to prompt/completion**: smol-magpie-ultra has 3-turn conversations. Use last-assistant-turn extraction (same logic as Taboo's `to_prompt_completion`).
3. **Finetune**: LoRA SFT on Gemma-3-1B-it using the 8K training pool. Existing `finetune.py` should work with minimal changes.
4. **Attribution**: Run Bergson with `score_mode=mean`, `tokenization_mode=finetune_raw` against the 500 query examples. Existing `score.py` pipeline.
5. **SAE extraction**: Run `sae_analysis.py` on the 8K training pool. GemmaScope layer 17, 16k width. With 2K token cap, ~16M total tokens.
6. **SAE predictor**: Train ridge regression from SAE features to attribution rank. Existing `train_bidir_classifier.py`.
7. **Validation against metadata**: Correlate SAE-predicted rank with `reward_model_score`, `difficulty`, `quality`, `category`. Key question: does the SAE find something orthogonal to the reward model?
8. **Continuation finetuning**: Select top-k by SAE predictor vs matched random. Continue finetuning. Evaluate on held-out eval set.
9. **Eval metric**: eval loss on held-out 500, or adapt to a downstream benchmark (IFEval, MT-Bench) if feasible.

### 13.6) Format Adaptation Notes
- smol-magpie-ultra uses `messages` format: `[{role, content}, ...]` with 3 turns (user/assistant/user/assistant/user/assistant).
- Current pipeline expects `prompt` + `completion` columns.
- Flatten via last-assistant-turn extraction: everything before the final assistant message becomes `prompt`, the final assistant content becomes `completion`.
- Preserve metadata columns (`category`, `difficulty`, `quality`, `reward_model_score`) through the pipeline for validation.
- Group key for train/test splitting: `category` or a composite of category + difficulty.

### 13.7) Key Hypothesis
- Not all instruction-following data contributes equally to model learning.
- High-attribution examples have a detectable SAE fingerprint that differs from surface-level quality signals (reward model score, difficulty label).
- This fingerprint can transfer: once learned on real attribution labels, it can rank new/synthetic data without re-running expensive gradient-based attribution.
- Finetuning on SAE-selected subsets outperforms matched random subsets of the same size.
