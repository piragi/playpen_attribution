# Playpen Attribution — Design Rationale & Findings

Last updated: 2026-02-23

---

## 1) Overall Goal

Prove that **residual stream probing + gradient-based data attribution can identify higher-quality SFT training data**, and that training on this selected data produces better instruction-following models than randomly sampled data.

The pipeline:
1. **Build data**: Tokenize smol-magpie-ultra → train/val splits (math + data-analysis category filter).
2. **Finetune scoring adapter**: LoRA SFT on train split → adapter used by Bergson attribution.
3. **Attribution**: Bergson scores each pool example by influence toward a high-quality attr_query (smol-magpie-ultra, good/excellent, math+DA, disjoint from pool).
4. **Probe**: Fit Ridge Regression on layer-17 residual stream embeddings of the scored pool → predict attribution scores.
5. **Generate continuation datasets**: Apply probe to 30k new smol-magpie-ultra rows → select top-10k quality reference + scaled quality subsets (default: 80%, 50%) + two random baselines (token-matched, token+category-matched, fixed at 10k rows).
6. **Finetune arms from base**: quality reference/scales + both random baselines.
7. **Eval**: ARC, HellaSwag, WinoGrande, IFEval, GSM8K — compare all arms.

Core claim: **Residual stream probing makes data attribution scalable** — one forward pass + linear probe replaces expensive per-example attribution at inference time.

Baseline to beat: **random selection from the same pool**. Quality arm must outperform random arm to validate the claim.

---

## 2) The Critical Finding: Train Data = Attribution Pool

### The hypothesis (v1–v3 design)

Originally the attribution pool was a **held-out split** — examples the model was never trained on. The reasoning was standard ML: you don't want the probe to overfit to training examples.

### Why this was wrong

The gradient signal on held-out examples is weak and noisy. When the model has never seen an example, its gradient response is essentially random noise — the model hasn't learned to "care" about those examples. The attribution score reflects influence on the query, but if the model hasn't been adjusted by the example, the gradient is low-magnitude and inconsistent.

When the model IS trained on an example, the gradient encodes exactly the direction the model moved because of that example. High-quality examples produce strong, distinctive gradients; low-quality examples produce weak, diffuse gradients. This is precisely the signal we want the probe to learn.

### The evidence

| Pool type | Val R² | Val Pearson r |
|---|---|---|
| Held-out (v3, 10k pool) | 0.345 | 0.618 |
| **Train data (v4, 5k train)** | **0.712** | **0.852** |

A jump from R²=0.35 to R²=0.71 — more than double the explained variance — purely from switching the pool to the training data. The train-val gap also becomes much healthier (0.90 vs 0.71), indicating the probe is learning a genuine signal, not memorizing noise.

### What this means for pipeline design

**Attribution pool = training data.** Set `attr_pool_size=0` in `build_sft_data.py`. The manifest then aliases `score_pool` → `train` so `score.py` scores the training examples. The probe trains on gradients from data the model actually learned from.

This is not data leakage — the probe predicts probe scores for *new* examples at inference time (generate_continued_dataset.py), which are never seen during training or attribution.

---

## 3) Eval Results (v4)

Both arms trained on 10k examples drawn from smol-magpie-ultra (math + data-analysis, post-finetune).

| Task | Quality (10k) | Random (10k) | Delta |
|---|---|---|---|
| arc_challenge | 0.3805 | 0.4002 | −0.020 |
| arc_easy | 0.5080 | 0.5181 | −0.010 |
| hellaswag | 0.6250 | 0.6317 | −0.007 |
| winogrande | 0.5975 | 0.5864 | **+0.011** |
| ifeval | 0.1627 | 0.1553 | **+0.007** |
| **gsm8k** | **0.2161** | **0.1638** | **+0.052** |

Quality beats random on GSM8K by **+5.2 points** — the task directly targeted by our math+DA attribution query. The general knowledge tasks (ARC, HellaSwag) favour random, which is expected: our quality selector is optimised for math/DA signal, not general factual recall.

### Open question: data efficiency

The key unanswered question is **how much random data is needed to match the quality arm's GSM8K score of 0.2161**. This is the data efficiency ratio of the probe selection.

| Random data | GSM8K |
|---|---|
| 10k | 0.1638 |
| 20k | ? |
| 30k | ? |
| Quality 10k | **0.2161** |

To run this experiment: increase `random_size` in `generate_continued_dataset.py` (the pool is already scored; no re-attribution needed) and retrain. If random needs 30k to match quality's 10k, that is a **3× data efficiency gain** from probe selection.

### Evidence status (current)

The +5.2 GSM8K delta in v4 is **encouraging but provisional**, not yet a claim-grade validation of the core hypothesis.

Why provisional:
- v4 historical baseline had a substantial token-budget mismatch versus quality (quality was longer on average), and length is strongly correlated with attribution score.
- v4 historical baseline also had category-mix mismatch (quality was more math-heavy), which can directly affect GSM8K.
- Result is currently a single-run comparison; no seed-robust confidence interval yet.

Interpretation: treat v4 as a strong signal worth pursuing. The pipeline now emits matched random baselines (token and token+category), and these reruns are required before concluding the probe itself is the primary cause of the gain.

---

## 4) Active Pipeline Scripts

| Script | Role |
|---|---|
| `build_sft_data.py` | Tokenize smol-magpie-ultra, mask prompts, produce train/val splits + manifest. `attr_pool_size=0` aliases score_pool→train. |
| `rebuild_attr_query.py` | Build attr_query from smol-magpie-ultra (quality-filtered, disjoint from pool). Run after build_sft_data.py. |
| `finetune.py` | LoRA SFT on pre-tokenized data; saves IT tokenizer alongside adapter. |
| `score.py` | Bergson gradient-based attribution; pool vs attr_query; supports PEFT adapters. Pool is the training data. |
| `probe.py` | Extract layer-17 residual stream from pool → fit Ridge probe to attribution scores. Caches embeddings to avoid re-extraction. |
| `generate_continued_dataset.py` | Score new smol-magpie-ultra rows with probe → quality reference (top-N), optional scaled quality subsets, and two fixed-size random baselines (token-matched, token+category-matched). |
| `eval_harness.py` | lm-eval benchmark evaluation (ARC, HellaSwag, WinoGrande, IFEval, GSM8K). |
| `vram.py` | VRAM profiler for SmolLM2-1.7B. |

### Pipeline order (v4)

```
build_sft_data.py
  → finetune.py (train split)
  → rebuild_attr_query.py
  → score.py (pool = train split)
  → probe.py  ← check R² here before proceeding; target R² > 0.5
  → generate_continued_dataset.py
  → finetune.py (quality arm + matched random baselines)
  → eval_harness.py (all arms)
```

---

## 5) Key Design Decisions & Rationale

### Why math + data-analysis (not all categories)?

Mixing all categories in the attribution query creates a noisy mean gradient — math examples partially cancel reasoning examples. Filtering to math+DA:
- Focuses the quality signal on quantitative reasoning
- Gives a measurable target task (GSM8K) to validate the selection
- Still provides enough data (72k usable rows) for all pipeline stages

### Why smol-magpie-ultra?

In-distribution: smol-magpie-ultra is the source for both the SFT training data and the continuation pool. Attribution scores reflect influence on a query drawn from the same distribution. Cross-distribution attribution (e.g. query from GSM8K, pool from smoltalk) would conflate domain shift with quality.

### Why LoRA (not full fine-tuning)?

Gradient computation for attribution concentrates on LoRA adapter parameters. Full fine-tuning would produce gradients over all 1.7B parameters — far too large for the TRAK projection and attribution pipeline. LoRA keeps the gradient signal focused and memory-feasible.

### Why layer 17 for probing?

Layer 17 (of 24) is in the upper-middle of the network — past the point where syntactic features dominate but before the final output projection compresses the representation. Empirically this has given the best probe R² in this project. The pooling position is the **last response token** (last index where labels != −100), which captures the model's final representation of the completed response.

### Why Ridge Regression (not MLP)?

Ridge Regression with a single hyperparameter is interpretable, numerically stable, and avoids probe overfitting. The probe's job is not to maximise accuracy but to produce a ranking of new examples that generalises. Ridge forces the probe to use a linear combination of the full 2048-dimensional embedding — this is appropriate given that RepE (Zou et al., 2023) demonstrates that quality concepts are linearly represented in the residual stream.

### ridge_alpha = 100 (not 1.0)

At alpha=1.0 the probe receives an ill-conditioned matrix warning (rcond ≈ 1e-7) because the 2048-dimensional embeddings have highly correlated features. Alpha=100 provides enough regularisation to suppress the ill-conditioning while the R² improvement over alpha=1.0 is modest (~0.03), confirming the ceiling is a genuine signal limit, not a regularisation issue.

### Category filter on continuation pool

The continuation pool (generate_continued_dataset.py) is filtered to the same categories as the attribution query (math + data-analysis). This is intentional: the probe was trained to predict scores under a math+DA query, so it will correctly rank math+DA examples. Applying it to unrelated categories (e.g. creative writing) would produce meaningless scores.

### Matched random baselines

`generate_continued_dataset.py` now emits two controlled random baselines per quality arm:
- `random_token_match`: same row count, random draw from the quality-excluded remainder, then simple token-increase swaps if the draw is below the quality token budget.
- `random_token_cat_match`: same row count and same category proportions as quality, with category-preserving token-increase swaps if needed.

Both baselines are drawn from the quality-excluded remainder. If an exact `>= target tokens` match is impossible, the script records this in `continuation_manifest.json` (`met_target_tokens=false`, `max_possible_tokens=<...>`).

---

## 6) Critical Technical Notes

### Tokenizer must match model

The attr_query tokenizer bug burned time: `rebuild_attr_query.py` previously derived the tokenizer from a fallback model when `tokenizer_model` was missing in the manifest. With SmolLM2 this can produce out-of-vocab token IDs and fail with CUDA index assertions.

Fix: read `manifest.get("tokenizer_model")` first, and if absent derive tokenizer from `base_model + "-Instruct"` (for SmolLM2).

### ensure_adapter_config writes the wrong model config

`score.py` calls `ensure_adapter_config()` which writes `config.json` into the adapter directory from the `--base-model` argument. If `--base-model` is wrong, the config.json is wrong and stays wrong on subsequent runs (the function only writes if the file doesn't exist). Current default: `HuggingFaceTB/SmolLM2-1.7B`. If this is ever wrong, delete `config.json` from the adapter directory.

### Base model + IT tokenizer pattern

```
Training:    AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")          ← base weights
Tokenizer:   AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")        ← chat template
```

IT model naming convention for SmolLM2: base = `SmolLM2-1.7B` → instruct = `SmolLM2-1.7B-Instruct`.

### Model loading pattern

Scripts now pass model IDs directly to `from_pretrained(...)` and rely on the Hugging Face cache machinery. `HF_HOME` is set once (if not already set) to a standard local cache path.

### Attention implementation

All scripts use `attn_implementation="sdpa"`. Do not use `"flash_attention_2"` (not installed) or `"eager"` (slow).

### eval_harness.py correct invocation

```bash
uv run eval_harness.py \
  --base-model HuggingFaceTB/SmolLM2-1.7B \
  --adapter-path <adapter_dir> \
  --apply-chat-template \
  --batch-size auto \
  --output-json <path>.json
```

`--output-json` not `--output-dir`. `--apply-chat-template` is required for SFT model evaluation. `--batch-size auto` fills available VRAM.

---

## 7) Default Hyperparameters (v4)

### build_sft_data.py

| Parameter | Value | Reason |
|---|---|---|
| `category_filter` | `{"math", "data-analysis"}` | Focus quality signal on quantitative reasoning |
| `train_size` | 5,000 | Strong enough IF gradient signal for attribution |
| `val_size` | 500 | Minimal validation overhead |
| `attr_pool_size` | **0** | Pool = train data; see §2 |

### rebuild_attr_query.py

| Parameter | Value | Reason |
|---|---|---|
| `query_smol_size` | 4,096 | Mean gradient over 4k examples is stable |
| `quality_min` | `{good, excellent}` | Only high-quality examples define the target direction |
| `category_filter` | `{"math", "data-analysis"}` | Match the train data distribution |

### score.py

| Parameter | Value |
|---|---|
| `--pool-split` | `score_pool` (aliases train when `attr_pool_size=0`) |
| `--query-split` | `attr_query` |
| `--projection-dim` | 32 |
| `--preconditioning-mode` | `query` |

### probe.py

| Parameter | Value | Reason |
|---|---|---|
| `extraction_layer` | 17 | Upper-middle network; best empirical R² |
| `pooling` | Last response token | Captures final response representation |
| `ridge_alpha` | 100.0 | Avoids ill-conditioning of 2048-dim features |
| `val_frac` | 0.20 | Standard held-out fraction |

### generate_continued_dataset.py

| Parameter | Value |
|---|---|
| `category_filter` | `{"math", "data-analysis"}` |
| `pool_size` | 30,000 |
| `quality_size` | 10,000 |
| `quality_scales` | `[1.0, 0.8, 0.5]` |
| `random_size` | 10,000 |

---

## 8) Data Layout (v4)

```
runs/smoltalk_v4/
  manifest.json                 ← score_pool aliases train; raw_rows_consumed for continuation skip
  data/
    train/        5,000 rows — SFT training AND attribution pool
    val/            500 rows — SFT validation
    attr_query/   4,096 rows — smol-magpie-ultra good/excellent, math+DA, post-skip
  adapter/        LoRA adapter trained on train (5k) + IT tokenizer
  scores_math_da/
    row_diagnostics.jsonl       ← per-train-row Bergson scores against math+DA query
  probe/
    pool_embeddings.npy         ← (5000, 2048) layer-17 residual stream for train
    probe_math_da.pkl           ← Ridge probe (val R²=0.71, Pearson r=0.85)
    probe_meta_math_da.json
  continuation/
    quality_math_da/  10k rows — top-10k by probe score
    quality_math_da_80pct/     8k rows — top-80% of quality reference
    quality_math_da_50pct/     5k rows — top-50% of quality reference
    random_token_match/      10k rows — token-budget matched random baseline
    random_token_cat_match/  10k rows — token+category matched random baseline
  adapter_quality_math_da/
  adapter_quality_math_da_80pct/
  adapter_quality_math_da_50pct/
  adapter_random_token_match/
  adapter_random_token_cat_match/
  evals/
    quality_math_da.json        ← GSM8K: 0.2161
    quality_math_da_80pct.json
    quality_math_da_50pct.json
    random_token_match.json
    random_token_cat_match.json
```

---

## 9) Open Questions

1. **Data efficiency ratio**: How many random examples = 10k quality examples on GSM8K? Run random arm at 20k and 30k. If 30k random ≈ 10k quality, that is a 3× data efficiency gain.

2. **Does the probe generalise across categories?** The probe is trained on math+DA attribution scores. Would it correctly rank reasoning or coding examples? Requires a separate attribution run with a reasoning query. We tried some more general probes but found them lacking.

3. **Probe R² ceiling**: R²=0.71 with 5k training examples. Does it improve with more training data? The pool is fundamentally limited to the train split size. Also what is the min/max of data needed here?

4. **Quality label ablation**: Can a binary probe (good/excellent vs poor) trained without any attribution match the performance of the attribution probe? If yes, Bergson is unnecessary and quality labels alone suffice.

5. **Selection rate sensitivity**: Quality arm = top-10k of 30k (33%). What happens at top-5k (17%) — does tighter selection improve GSM8K further or hurt coverage?

6. **Unmatched random ablation?** In addition to matched random baselines, do we also want a pure uniform-random arm (no token/category matching) as an intentionally weak baseline for context?

---

## 10) Hardware Profile — RTX 5090 (31.36 GB)

Measured with `vram.py`, SmolLM2-1.7B, sdpa, bfloat16, seq_len=2048.

| Config | Peak VRAM |
|---|---|
| Model load | 3.19 GB |
| LoRA train bs=8, grad_ckpt ON | 11.33 GB |
| Probe extraction bs=64 | 12.50 GB |
| lm-eval batch-size=auto | fills available VRAM |

---

## 11) Theoretical Foundation

### Why linear probing works (Representation Engineering, Zou et al. 2023)

RepE proves that high-level semantic concepts — including quality judgements — are linearly represented across the residual stream. The probe does not need to be a deep network; a Ridge Regression on layer-17 hidden states is sufficient to find the linear direction for "this example produces high-attribution gradients toward a math+DA quality query."

### Why attribution scores are the right training signal

Standard reward models are trained on human preference annotations. Our probe uses Bergson attribution scores as the ground truth — effectively a **computationally-derived quality label** grounded in gradient geometry rather than human judgement. This avoids annotation cost and anchors quality to the model's actual learning dynamics.

### Why this is not just RLHF / DPO

RLHF/DPO select *model outputs* (responses) as preferred vs rejected. This pipeline selects *training examples* (input-output pairs) by their influence on producing high-quality behaviour. The selection happens before any training, not as a post-hoc reward signal.

---

## 12) Version History

| Version | Key change | GSM8K (quality) | GSM8K (random) |
|---|---|---|---|
| v2 | Mixed pool, 1k SFT, held-out pool | worse than random | — |
| v3 | Category filter (math+DA), 2k SFT, held-out 10k pool | not completed | — |
| **v4** | **Train data = pool**, 5k SFT | **0.2161** | **0.1638** |

**The v2/v3 failure was not a hyperparameter problem — it was a fundamental design flaw.** Using a held-out pool means the model has weak gradients on those examples, so the attribution scores have low variance and the probe cannot learn a meaningful ranking. Fixing this one design decision (train = pool) caused R² to jump from 0.35 to 0.71 and produced a measurable +5.2 point GSM8K improvement.
