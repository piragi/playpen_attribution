# Playpen Attribution Project Guide

## Goal
Build a reproducible pipeline for:
1. LoRA SFT on playpen game data.
2. Gradient-based data attribution to identify high-impact training examples.
3. SAE-based analysis of top-attributed data to characterize feature patterns that correlate with influence.

Current focus is a working end-to-end baseline for steps 1-3.

## Tooling Rules
1. Always run scripts with `uv`.
2. Use local Bergson source at `../bergson`.
3. Keep scripts simple and config-driven (single `CONFIG` dict per script).

## Repository Workflow
1. Prepare filtered train/validation data from `colab-potsdam/playpen-data` (`interactions`).
2. Train LoRA SFT adapter (`data.py`).
3. Run Bergson reduce+score attribution (`score_samples.py`).
4. Run SAE feature extraction on filtered training rows (`sae_analysis.py`).
5. Run contrast/regression analysis on cached SAE arrays (`sae_contrast.py`).

## Current Script Roles

### `data.py`
Purpose:
1. Filter by game/role and drop aborted outcomes.
2. Convert conversation to prompt/completion.
3. Train Gemma-3-1B-it LoRA via `trl.SFTTrainer`.

Current training policy:
1. `completion_only_loss=True`
2. `assistant_only_loss=False`
3. LoRA target modules: attention + MLP projections.

References:
1. `data.py:43`
2. `data.py:56`
3. `data.py:113`
4. `data.py:125`

### `score_samples.py`
Purpose:
1. Build filtered train/val splits with stable `row_id`.
2. Run Bergson `reduce` on validation.
3. Run Bergson `score` on train and dump top/bottom examples.

Current attribution setup:
1. `score_mode="mean"`
2. `unit_normalize=True`
3. `preconditioning.mode="mixed"`
4. `preconditioning.projection_dim=32`
5. `mixing_coefficient=0.99`

References:
1. `score_samples.py:15`
2. `score_samples.py:138`
3. `score_samples.py:176`
4. `score_samples.py:189`
5. `score_samples.py:232`

### `sae_analysis.py`
Purpose:
1. Run SAE encode on prompts.
2. Aggregate layer-level top features.
3. Save per-sample arrays for fast downstream analysis.

Current default mode:
1. `input_mode="train_dataset"`
2. Loads full filtered train split from `runs/taboo_attr/data/train`
3. Defaults to `layer_17_width_16k_l0_small`
4. Saves per-sample outputs to `runs/taboo_attr/sae_samples_layer17_all_train`

Per-sample arrays schema:
1. `feature_presence`: `[n_samples, d_sae]` bool.
2. `sample_stats`: `[n_samples, 10]` float32.
3. `seq_lens`: `[n_samples]` int32.
4. `stat_feature_names`: names of the 10 sample stats.

References:
1. `sae_analysis.py:12`
2. `sae_analysis.py:51`
3. `sae_analysis.py:270`
4. `sae_analysis.py:343`
5. `sae_analysis.py:505`

### `sae_contrast.py`
Purpose:
1. Load Bergson train scores.
2. Load precomputed SAE arrays (`.npz`) + metadata (`examples.jsonl`).
3. Label top-k attributed rows vs rest.
4. Fit weighted logistic regression on `sample_stats`.
5. Report AUC + per-feature enrichment/depletion from `feature_presence`.

Important change:
1. No model/SAE forward pass here anymore.
2. It is now a fast analysis script on cached arrays.

References:
1. `sae_contrast.py:10`
2. `sae_contrast.py:29`
3. `sae_contrast.py:45`
4. `sae_contrast.py:149`
5. `sae_contrast.py:244`

## Bergson Scoring Semantics
Let:
1. `g_i`: gradient for train item `i`.
2. `q_j`: gradient for query item `j` (validation).

With `score="mean"`:
1. Query aggregate: `q_mean = (1/M) * sum_j q_j`.
2. Score each train sample by inner product with processed query vector.

Preconditioning:
1. Query-side transform can use damped inverse `(H + lambda I)^(-1)`.
2. Mixed mode can combine query/index preconditioners before inversion.
3. If omitted, it degenerates to identity (inner-product / cosine-like when normalized).

Primary Bergson references:
1. `../bergson/bergson/score/score.py`
2. `../bergson/bergson/score/scorer.py`
3. `../bergson/bergson/reduce.py`

## Current Artifacts (Taboo WordGuesser)
Main run root:
1. `runs/taboo_attr/`

Key outputs:
1. `runs/taboo_attr/data/train` and `runs/taboo_attr/data/val`
2. `runs/taboo_attr/train_scores/`
3. `runs/taboo_attr/ranked_examples.json`
4. `runs/taboo_attr/sae_feature_activity_layer17_all_train.json`
5. `runs/taboo_attr/sae_samples_layer17_all_train/`
6. `runs/taboo_attr/sae_feature_activity_layers7_13_17_22_all_train.json`
7. `runs/taboo_attr/sae_samples_layers7_13_17_22_all_train/`
8. `runs/taboo_attr/sae_contrast_layer7.json`
9. `runs/taboo_attr/sae_contrast_layer13.json`
10. `runs/taboo_attr/sae_contrast_layer17.json`
11. `runs/taboo_attr/sae_contrast_layer22.json`
12. `runs/taboo_attr/sae_contrast_layers7_13_17_22_summary.json`

## Current Layer-Wise Separation Baseline
Top-100 attributed vs rest using cached SAE stats:
1. Layer 7: test AUC `0.7979`
2. Layer 13: test AUC `0.8832`
3. Layer 17: test AUC `0.8558`
4. Layer 22: test AUC `0.8427`

Interpretation:
1. All tested layers separate above random.
2. Layers 13 and 17 are strongest in current setup.

## Known Workarounds / Constraints
1. LoRA adapter config bootstrap for Bergson:
   `AutoConfig.from_pretrained(base_model).save_pretrained(adapter_path)`
2. Prompt/completion conversion is used to avoid fragile assistant-span matching on repeated multi-turn text.
3. SFT uses completion-only loss due Gemma chat-template mask limitations in this setup.

## Practical Commands
1. SFT: `uv run data.py`
2. Attribution: `uv run score_samples.py`
3. SAE extraction (default layer 17 full train): `uv run sae_analysis.py`
4. Contrast analysis from cached arrays: `uv run sae_contrast.py`

## Next Priorities
1. Compare preconditioning modes (`none`, `query`, `mixed`) and ranking stability.
2. Add grouped splits by `task_id` for contrast analysis to reduce leakage risk.
3. Expand SAE contrast from sample-level stats to token-level or turn-level localized analyses.
4. Map enriched SAE feature IDs back to concrete token spans for mechanistic inspection.
5. Planned synthetic-data experiment: generate additional multi-turn Taboo episodes with Claude Haiku, GPT-4o, and Gemini Flash, then compare logistic-aligned selection vs matched-random selection under the same token budget.
6. A first batch of a few thousand episodes is a reasonable target, since the current filtered baseline is only 1,320 rows total (1,056 train + 264 validation).
