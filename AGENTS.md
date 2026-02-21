# Playpen Attribution — Active Notes

Last updated: 2026-02-20

---

## 1) Overall Goal

Prove that **mechanistic interpretability (SAE feature analysis) combined with data attribution can identify higher-quality pretraining data**, and that training on this selected data produces better models than randomly sampled or heuristically filtered data.

Concretely:
1. **Phase 1**: Continued pretraining of Gemma-3-270m-pt on FineWeb → establish a training run whose samples can be attributed.
2. **Phase 2**: Compute gradient-based data attribution (Bergson) → label each training chunk by influence on benchmarks.
3. **Phase 3**: Extract SAE features (GemmaScope 2) from each chunk using the **base model** → train a lightweight classifier that predicts attribution from SAE features.
4. **Phase 4**: Apply the classifier to a larger FineWeb pool → select top-scoring data → continue pretraining on it → compare against random baseline.

Core claim: **SAE fingerprinting makes data attribution scalable** — one forward pass through model + SAE + small classifier replaces expensive gradient computation.

Benchmarks: ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande, PIQA (lm-eval harness, log-prob scoring, base models only — no SFT needed).

---

## 2) Key Design Decisions & Rationale

### Why Gemma-3-270m-pt?

- **GemmaScope 2** provides production-quality JumpReLU SAEs for all Gemma 3 sizes (270M–27B), covering residual stream, attention, MLP at every layer. Training SAEs from scratch for another model would cost weeks.
- SAE widths of 16k, 64k, 256k, 1M available with L0 ≈ 10, 50, 150.
- SAELens loads GemmaScope 2 directly.
- 270M fits in 10GB VRAM (RTX 3080) for full fine-tuning with gradient checkpointing.
- Trade-off: We can't reproduce Gemma's original training exactly (distillation from larger teacher, 2T tokens). But we only need a consistent experimental setup where data quality is the only variable.

### Why raw FineWeb (not FineWeb-Edu)?

Our project needs a data pool with **high quality variance**. Pre-filtered datasets (FineWeb-Edu, FineMath-4+) have too little variance — most samples are decent, leaving little room for the SAE detector to demonstrate value. Raw FineWeb contains everything from garbage to excellent content.

FineWeb-Edu's quality classifier (Llama-3 annotations, score ≥ 3) is a natural **baseline to beat**.

We use `CC-MAIN-2024-10` (single CommonCrawl dump) as recommended for smaller runs.

### Why base model for SAE analysis (not v1)?

The GemmaScope 2 SAEs were trained on the **base Gemma-3-270m-pt** activations. Running SAE extraction on v1 (continued-pretrained) creates a distribution mismatch — the SAE's decoder was optimized for base model residual streams, and v1's activations have drifted. This mismatch means SAE features may fire incorrectly or miss important patterns.

Using the base model for SAE extraction ensures the features are interpretable and well-calibrated.

### Why preserve FineWeb streaming order in filtered dataset?

Language models benefit from seeing documents in a natural order. Web crawl data has implicit structure — documents from the same site or topic cluster together. Shuffling the filtered dataset would destroy this locality, potentially hurting learning. We sort the selected chunks by their original stream position.

---

## 3) Active Pipeline Scripts

| Script | Role |
|---|---|
| `pretrain.py` | Full-model continued pretraining on FineWeb (or saved dataset) |
| `build_pretrain_query.py` | Builds score_pool (FineWeb chunks) + attr_query (benchmark examples) for attribution |
| `score.py` | Bergson gradient-based attribution pipeline |
| `eval_harness.py` | lm-eval benchmark evaluation (ARC, HellaSwag, WinoGrande, PIQA) |
| `sae_analysis.py` | SAE feature extraction from base model |
| `train_bidir_classifier.py` | SAE fingerprint classifier (predicts attribution from SAE features) |
| `build_continuation_data.py` | Builds filtered (SAE-selected) + random continuation datasets |

---

## 4) Critical Technical Notes

### Bergson rule

Pass **pre-tokenized data** (`input_ids` + `labels` columns) directly to score.py using
`--tokenization-mode bergson_chat`. Bergson skips its internal tokenization when `input_ids`
is already present, avoiding chat-template or tokenization mismatch.

- Pool (FineWeb chunks): `labels = input_ids` — full causal supervision, no masking.
- Query (benchmark examples): prompt tokens masked with `-100`, only answer tokens supervised.

### Best attribution config

```
--score-mode mean
--preconditioning-mode query
--projection-dim 32
--tokenization-mode bergson_chat
```

No `--adapter-path` needed for full pretrained models — score.py falls back to `--base-model` automatically.

### lm-eval + Gemma3 fix

lm-eval 0.4.11 passes `dtype=` to `Gemma3ForCausalLM.__init__` instead of `torch_dtype=`,
causing a TypeError. Fix applied directly to the installed package:

```
.venv/lib/python3.13/site-packages/lm_eval/models/huggingface.py
lines 635, 718: dtype= → torch_dtype=
```

### FineWeb stream is deterministic

`HuggingFaceFW/fineweb` with `CC-MAIN-2024-10` streams in fixed order (no shuffle).
Same `--max-tokens` → identical chunks. The `skip_chunks` mechanism in `build_continuation_data.py` relies on this.

### Benchmark query format

- **WinoGrande**: prompt = sentence with blank + both options listed; completion = correct fill-in word
- **ARC-Challenge**: prompt = question + lettered choices; completion = correct letter (e.g. ` A`)
- Both use `max_query_length=256`; ~2 supervised tokens per example (answer + EOS)

---

## 5) Continued Pretraining Hyperparameters

### Why the v1 run regressed (and the fix)

**v1 used lr=1e-4** with weight_decay=0.1. This caused benchmark regression across all tasks (worst: ARC-Easy -0.073). The root cause:

1. **LR too high**: 1e-4 is in the range of the original pretraining peak LR for a 270M model (typically 5e-4 to 1e-3). For continued pretraining, the literature recommends starting at **eta_min of the original schedule** — roughly 10x lower than the original peak. See "Reuse, Don't Retrain" (2024).
2. **Weight decay too aggressive**: 0.1 is standard for pretraining from scratch but too high for CPT. The Gemma2 continual pretraining paper used 0.01.
3. **No data replay**: Even 1-5% of diverse original pretraining data helps prevent forgetting.

Key references:
- "Reuse, Don't Retrain" (Databloom, 2024): Start CPT at eta_min of original model, decay further. Their baseline: eta_max=4.5e-4 → CPT started at 4.5e-5.
- "Simple and Scalable Strategies to Continually Pre-train LLMs" (Ibrahim et al., 2024, TMLR): LR re-warming + re-decaying + replay matches full retraining.
- "Full-Parameter Continual Pretraining of Gemma2" (2025): lr=2e-4 for 2B with weight_decay=0.01, still needed EWC regularization.

### v2 parameters (conservative)

| Parameter | v1 (regressed) | v2 (new) | Rationale |
|---|---|---|---|
| `learning_rate` | 1e-4 | **3e-5** | ~3-10x below original peak; in the safe CPT zone per literature |
| `min_learning_rate` | 1e-5 | **3e-6** | Maintains 10:1 peak-to-floor ratio (cosine decay) |
| `warmup_steps` | 500 | **200** | ~5% of total steps; shorter warmup standard for CPT |
| `weight_decay` | 0.1 | **0.01** | Per Gemma2 CPT paper; less aggressive regularization |
| `gradient_accumulation` | 16 | 16 | Unchanged — effective batch = 16 × 1024 = 16K tokens/step |
| `max_length` | 1024 | 1024 | Unchanged |
| `gradient_checkpointing` | on | on | Required for 10GB VRAM |
| `max_grad_norm` | 1.0 | 1.0 | Unchanged |
| `seed` | 42 | 42 | Unchanged |

### Token budget: 200M for v2

v1 used 100M tokens (0.37 tokens/param for 270M model). With the lower LR, the effective "learning dose" (LR × steps) is ~3x smaller per token. To compensate:

- **200M tokens** (~0.74 tokens/param) — still well below the danger zone for forgetting
- Gives the model more exposure to see the effect of data quality differences
- Runtime: ~6h (can run overnight)
- At lr=3e-5, 200M tokens gives roughly the same total parameter update magnitude as 65M tokens at the old lr=1e-4

The filtered vs random arms in Phase 4 should also use 200M tokens (or match whatever Phase 1 uses) for consistency.

---

## 6) v1 Results (Pilot Run — Regressed)

### Phase 1: Continued pretraining

Config: lr=1e-4, wd=0.1, warmup=500, 100M tokens on FineWeb CC-MAIN-2024-10.
Result: `runs/pretrain_270m_v1/`, train_loss=3.076, 6103 steps, ~3.2h.

| Task | Base | v1 Pretrained | Delta |
|---|---|---|---|
| arc_challenge | 0.2833 | 0.2705 | -0.013 |
| arc_easy | 0.5694 | 0.4966 | **-0.073** |
| hellaswag | 0.4136 | 0.3937 | -0.020 |
| winogrande | 0.5359 | 0.5288 | -0.007 |
| piqa | 0.6823 | 0.6681 | -0.014 |

ARC-Easy regressed most because FineWeb lacks curated science content. All benchmarks regressed — clear sign of too-aggressive hyperparameters.

### Phase 2: Attribution

Scored all 97,658 FineWeb pool chunks by gradient influence on benchmark query loss.
Config: `--score-mode mean --preconditioning-mode query --projection-dim 32`.
Top/bottom 10% (9,766 each) labeled for SAE classifier training.

### Phase 3: SAE feature extraction + classifier

SAE features extracted from **v1 model** (mistake — should have used base model; SAE was trained on base).
19,532 samples (9,766 positive + 9,766 negative), top-256 features per sample.

Classifier ablation results (all using SAEFingerprint architecture, mean pooling, K=256):

| ID | Config | Val AUROC | Val Acc |
|---|---|---|---|
| A | Logistic regression on 10 global stats | 0.623 | 0.593 |
| B | L1 logistic regression on sparse 16k vector | 0.639 | 0.607 |
| C | Mean pooling, no global stats | **0.748** | **0.682** |
| D | Mean pooling + global stats | 0.745 | 0.680 |
| E | Attention pooling, no global stats | 0.742 | 0.678 |
| F | Attention pooling + global stats | 0.740 | 0.681 |

K ablation (all mean pooling, no global stats):

| K | Val AUROC | Val Acc |
|---|---|---|
| 64 | 0.715 | 0.653 |
| 128 | 0.724 | 0.657 |
| **256** | **0.745** | **0.678** |
| 512 | 0.743 | 0.665 |

**Winner: Model C with K=256** — mean pooling, no global stats, top-256 features. Simple and effective. Global stats add no value; attention pooling adds complexity without benefit.

### Phase 4: Continuation (v1 — also regressed)

Continued pretraining from v1 on random arm (100M more tokens, same aggressive lr=1e-4):

| Task | Base | v1 | v1+random | Delta (v1→random) |
|---|---|---|---|---|
| arc_challenge | 0.2833 | 0.2705 | 0.2679 | -0.003 |
| arc_easy | 0.5694 | 0.4966 | 0.4941 | -0.003 |
| hellaswag | 0.4136 | 0.3937 | 0.3814 | -0.012 |
| winogrande | 0.5359 | 0.5288 | 0.5351 | +0.006 |
| piqa | 0.6823 | 0.6681 | 0.6665 | -0.002 |

Further regression on most benchmarks. Confirms the LR was too aggressive throughout.

---

## 7) v2 Plan (Full Redo)

The v1 pipeline validated that the approach works end-to-end (0.748 AUROC classifier despite noisy labels from regressed model). Now redo with correct hyperparameters.

### Phase 1: Continued pretraining (v2)

```bash
uv run pretrain.py \
  --base-model google/gemma-3-270m \
  --output-dir runs/pretrain_270m_v2 \
  --max-tokens 200000000 \
  --learning-rate 3e-5 \
  --min-learning-rate 3e-6 \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --save-steps 1000 \
  --save-total-limit 5
```

**Gate check**: Eval with `eval_harness.py` immediately after. If any benchmark drops more than 0.01 from base, stop and investigate before proceeding. Expected: flat or slight improvement.

Runtime: ~6h for 200M tokens. Run overnight.

### Phase 2: Build attribution datasets (v2)

```bash
uv run build_pretrain_query.py \
  --base-model runs/pretrain_270m_v2 \
  --output-dir runs/pretrain_attribution_v2 \
  --max-tokens 200000000
```

This re-tokenizes the same FineWeb stream to produce the score_pool (now ~195K chunks for 200M tokens) and benchmark query sets.

### Phase 3: Attribution (v2)

```bash
uv run score.py \
  --manifest runs/pretrain_attribution_v2/manifest.json \
  --base-model runs/pretrain_270m_v2 \
  --pool-split score_pool \
  --query-split attr_query \
  --tokenization-mode bergson_chat \
  --score-mode mean \
  --preconditioning-mode query \
  --projection-dim 32 \
  --output-dir runs/pretrain_attribution_v2/attribution_mean
```

### Phase 4: SAE extraction (v2)

Use **base model** (`google/gemma-3-270m`) for SAE extraction — already configured in `sae_analysis.py`.

Update CONFIG paths to point to v2 diagnostics:
```python
CONFIG = {
    "base_model": "google/gemma-3-270m",  # base, not v2!
    "diagnostics_path": "runs/pretrain_attribution_v2/attribution_mean/row_diagnostics.jsonl",
    "pool_path": "runs/pretrain_attribution_v2/data/score_pool",
    "output_dir": "runs/pretrain_attribution_v2/sae_features/layer12_width16k",
    ...
}
```

### Phase 5: Train SAE fingerprint classifier (v2)

Same architecture as v1 winner: Model C (mean pooling, K=256, no global stats).
Expected AUROC should be **higher** than 0.748 since:
- Attribution labels are cleaner (non-regressing v2 model)
- SAE features are from base model (correct distribution match)

### Phase 6: Build continuation datasets (v2)

Update `build_continuation_data.py` CONFIG:
```python
CONFIG = {
    "base_model": "google/gemma-3-270m",  # base model for SAE
    "classifier_path": "runs/pretrain_attribution_v2/sae_classifier/K256/best_model.pt",
    "skip_chunks": 195316,  # 200M tokens / 1024 = ~195K chunks
    "n_target_chunks": 195316,  # match Phase 1 token budget
    "output_dir": "runs/pretrain_continuation_v2",
    ...
}
```

This generates:
- `random/` — next 200M tokens in stream order
- `filtered/` — top 200M tokens by SAE score, stream order preserved
- `filtered_positions.json` — for overlap analysis
- `manifest.json` — includes overlap statistics

### Phase 7: Continuation pretraining (v2)

Both arms start from **v2** (the Phase 1 checkpoint), not from base:

```bash
# Filtered arm
uv run pretrain.py \
  --base-model runs/pretrain_270m_v2 \
  --dataset-path runs/pretrain_continuation_v2/filtered \
  --output-dir runs/pretrain_continuation_v2/filtered_model \
  --max-tokens 200000000

# Random arm
uv run pretrain.py \
  --base-model runs/pretrain_270m_v2 \
  --dataset-path runs/pretrain_continuation_v2/random \
  --output-dir runs/pretrain_continuation_v2/random_model \
  --max-tokens 200000000
```

Both use the same conservative LR (3e-5 default). Same compute budget, same starting checkpoint — only the data differs.

### Phase 8: Evaluation

```bash
# Eval all models
for model in google/gemma-3-270m runs/pretrain_270m_v2 runs/pretrain_continuation_v2/filtered_model runs/pretrain_continuation_v2/random_model; do
  uv run eval_harness.py --model-path $model --output-path ${model}/eval_v2.json
done
```

**Success criteria**:
- v2 does not regress from base (validates LR fix)
- filtered_model > random_model on most benchmarks (SAE selection works)
- Bonus: filtered_model > v2 (selected data improves over random FineWeb)

---

## 8) Canonical Artifacts

### v1 (pilot — regressed, kept for reference)

| Path | Contents |
|---|---|
| `runs/pretrain_270m_v1/` | Phase 1 pretrained model (lr=1e-4, regressed) |
| `runs/pretrain_270m_v1/pretrain_metrics.json` | Training metrics |
| `runs/pretrain_270m_v1/eval_base_full.json` | Base model benchmarks |
| `runs/pretrain_270m_v1/eval_pretrained_full.json` | v1 model benchmarks |
| `runs/pretrain_attribution_v1/` | Attribution datasets + scores + SAE features + classifier |
| `runs/pretrain_attribution_v1/sae_classifier/ablation_results.json` | Full classifier ablation |
| `runs/pretrain_continuation_v1/` | Random + filtered datasets + random continuation model |

### v2 (to be generated)

| Path | Contents |
|---|---|
| `runs/pretrain_270m_v2/` | Phase 1 pretrained model (lr=3e-5, 200M tokens) |
| `runs/pretrain_attribution_v2/` | Attribution datasets + scores |
| `runs/pretrain_attribution_v2/sae_features/` | SAE features (base model) |
| `runs/pretrain_attribution_v2/sae_classifier/` | Trained classifier |
| `runs/pretrain_continuation_v2/` | Random + filtered datasets + both continuation models |

---

## 9) Hardware Profile (RTX 3080, 10GB)

| Config | VRAM | Notes |
|---|---|---|
| eager, seq=1024, bs=1, grad ckpt, full training | 7.6 GB | Phase 1 & 4 pretraining |
| eager, seq=1024, bs=1, inference + SAE | 4.1 GB | SAE extraction (Phase 3) |
| SAE classifier training | <1 GB | CPU-feasible |
| lm-eval benchmarks | ~4 GB | Evaluation |

---

## 10) Open Questions

1. **Data replay**: Should we mix in ~5% diverse data during Phase 4 continuation to reduce forgetting risk? The literature strongly recommends it for distribution shifts, but FineWeb→FineWeb is same-domain.

2. **Pool factor**: Currently scoring 3x the target chunks and selecting the top 1x. Should this be larger (5x, 10x) to give the classifier more room to separate signal from noise?

3. **Layer ablation**: v1 only tested layer 12. The spec calls for ablating layers 9, 12, 15, and concatenated. Worth doing if classifier AUROC is low, but if 0.75+ repeats, layer 12 is fine.

4. **FineWeb-Edu baseline**: The spec calls for a third arm using FineWeb-Edu's quality scores as data selection. This is a stretch goal — the filtered vs random comparison is the core result.
