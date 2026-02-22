# Playpen Attribution — Active Notes

Last updated: 2026-02-22

---

## 1) Overall Goal

Prove that **SAE feature analysis combined with data attribution can identify higher-quality SFT training data**, and that training on this selected data produces better instruction-following models than randomly sampled or heuristically filtered data.

Concretely:
1. **Phase 1**: LoRA SFT of Gemma-3-1b (base) on SmolTalk → establish a training run whose samples can be attributed. ✅
2. **Phase 2**: Compute gradient-based data attribution (Bergson) → score each pool example by influence toward a high-quality query (smol-magpie-ultra). ← next
3. **Phase 3**: Extract residual stream embeddings from the base model at layer 17 for the pool → fit a Ridge Regression probe to predict attribution scores.
4. **Phase 4**: Apply the probe to a new 100k SmolTalk block → select top 10% → compare quality arm vs random arm on benchmarks.

Core claim: **Residual stream probing makes data attribution scalable** — one forward pass through the base model + a linear probe replaces both the SAE pipeline and the custom neural classifier.

Baseline to beat: **random selection from the same pool**. If the SAE-selected quality arm outperforms the random arm, we have a result.

Benchmarks: ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande, MMLU (SmolTalk paper set; lm-eval harness, log-prob scoring, `--apply-chat-template` required for SFT models).

---

## 2) Key Design Decisions & Rationale

### Why Gemma-3-1b base (not IT)?

- **GemmaScope 2** SAEs are trained on base model activations (layer 17, width 16k). Using the IT model for SAE extraction creates a distribution mismatch.
- Base model has **higher gradient variance** → cleaner attribution labels.
- We borrow the IT tokenizer for its chat template (same vocabulary, same token IDs). This is the standard approach for SFT from a base model.

### Why LoRA (not full fine-tuning)?

- Gradient computation for attribution concentrates on LoRA adapter parameters, keeping the signal focused.
- Fits in much less VRAM; faster iteration.

### Why SmolTalk `all` config for train/pool?

High quality variance (mix of Magpie-Ultra, Smol-Rewrite, Smol-Constraints, etc.) makes attribution signal meaningful. `smol-magpie-ultra` is a config of `HuggingFaceTB/smoltalk` that has quality labels preserved; the `all` config strips per-source metadata.

### Attribution query: smol-magpie-ultra (good/excellent)

The attr_query defines "quality" in gradient space — attribution scores measure how much a pool example helped produce behaviour like the query.

- **Old query** (ARC-Challenge + WinoGrande): biased toward very short multiple-choice Q&A (~71 tokens/example). Created a -0.33 length correlation in attribution scores — shorter pool examples scored higher simply because they resembled the short query, not because they were better training data.
- **New query**: 1,024 smol-magpie-ultra examples filtered to quality ∈ {good, excellent} (~820 tokens/example). Richer, more representative of what the model actually trained on.

Load directly from the `smol-magpie-ultra` config (which preserves quality labels), not from `all` (which strips them).

### No residualization

Bergson uses `unit_normalize=True` + query-side TRAK preconditioning. The remaining length correlation in scores is content-based (longer examples genuinely more aligned with the smol-magpie-ultra query direction), not a magnitude artifact. Do **not** regress out length.

---

## 3) Active Pipeline Scripts

| Script | Role |
|---|---|
| `build_sft_data.py` | Tokenize SmolTalk with IT chat template, mask prompts, produce train/val/attr_pool splits |
| `rebuild_attr_query.py` | Rebuild attr_query from smol-magpie-ultra (quality-filtered) without touching other splits |
| `finetune.py` | LoRA SFT on pre-tokenized data; saves IT tokenizer alongside adapter |
| `eval_harness.py` | lm-eval benchmark evaluation (ARC, HellaSwag, WinoGrande, MMLU) |
| `score.py` | Bergson gradient-based attribution; pool vs attr_query; supports PEFT adapters |
| `probe.py` | Extract layer-17 residual stream from pool → fit Ridge probe to attribution scores |
| `generate_continued_dataset.py` | Score new 100k SmolTalk block with probe → quality arm (top 10%) + random baseline |

**Note:** `sae_analysis.py`, `train_bidir_classifier.py`, `build_continuation_data.py` were removed on this branch. They remain on `sft_pivot` if needed for reference.

---

## 4) Critical Technical Notes

### Base model + IT tokenizer pattern

```
Training:    AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt")   ← base weights
Tokenizer:   AutoTokenizer.from_pretrained("google/gemma-3-1b-it")           ← chat template
```

After training, the IT tokenizer is saved alongside the LoRA adapter. Downstream scripts
(`eval_harness.py`, `score.py`) load tokenizer from the adapter directory.

### Bergson rule

Pass **pre-tokenized data** (`input_ids` + `labels` columns) directly. Bergson skips internal
tokenization when `input_ids` is already present.

- Pool (SmolTalk attr_pool): prompt tokens masked with `-100`, only assistant tokens supervised.
- Query (attr_query): same masking — smol-magpie-ultra examples formatted with IT chat template.

### Prompt masking

`_mask_prompt()` in `build_sft_data.py` uses cumulative prefix tokenization to find exact token
boundaries per assistant turn. Multi-turn conversations are handled correctly.

### score.py + PEFT

Pass `--adapter-path <adapter_dir>` to `score.py`. It calls `ensure_adapter_config()` which
injects a `config.json` (base model architecture) into the adapter directory so Bergson can
load the full model via PEFT.

score.py defaults: `unit_normalize=True`, `query_preconditioner` (TRAK KFAC eigendecomposition
on query side), `projection_dim=32`, `mixing_coefficient=0.99`. These are correct — do not change.

### eval_harness.py + IT tokenizer

When `--adapter-path` is set, `build_model_args()` automatically appends
`tokenizer=<adapter_path>` so lm-eval picks up the IT tokenizer (with chat template) from the
adapter directory. Always pass `--apply-chat-template` for SFT model evaluation.

### lm-eval + Gemma3 fix

lm-eval 0.4.11 passes `dtype=` to `Gemma3ForCausalLM.__init__` instead of `torch_dtype=`,
causing a TypeError. Fix applied as a runtime monkey-patch at the top of `eval_harness.py`
(wraps `AutoModelForCausalLM.from_pretrained` to translate `dtype=` → `torch_dtype=`).
This survives `uv sync` / reinstalls — do NOT patch the installed package directly.

### SmolTalk streaming

Always use `streaming=True` in `load_dataset` for SmolTalk. Without it, the full ~1M-row
dataset downloads before processing begins.

### In-training eval disabled

Gemma 3's 262k vocabulary causes OOM when computing eval loss (logits cast to float32 =
~1 GB per example at seq_len=1024). Use `eval_harness.py` post-training instead.

---

## 5) Data Layout

```
runs/smoltalk_v1/
  manifest.json
  data/
    train/        100,000 rows — SFT training
    val/            5,000 rows — SFT validation
    attr_pool/     50,000 rows — attribution pool (scored by Bergson)
    attr_query/     1,024 rows — smol-magpie-ultra quality∈{good,excellent}
  adapter/        LoRA adapter + IT tokenizer (Gemma-3-1b)
  scores/
    row_diagnostics.jsonl   ← per-pool-row Bergson scores (probe.py reads this)
    summary.json
    subsets/
  probe/
    pool_embeddings.npy     ← (50000, 2048) layer-17 residual stream for pool
    probe.pkl               ← fitted Ridge regression
    probe_meta.json         ← R², Pearson r, layer, alpha
  continuation/
    continuation_manifest.json
    quality/    — top-10k by probe score
    random/     — 10k from non-quality remainder
  adapter_quality/   LoRA adapter continued on quality arm
  adapter_random/    LoRA adapter continued on random arm
```

---

## 6) Default Hyperparameters

### build_sft_data.py

| Parameter | Value |
|---|---|
| `dataset_name` | `HuggingFaceTB/smoltalk` |
| `dataset_config` | `all` |
| `base_model` | `google/gemma-3-1b-pt` |
| `tokenizer_model` | `google/gemma-3-1b-it` |
| `max_length` | 1024 |
| `train_size` | 100,000 |
| `val_size` | 5,000 |
| `attr_pool_size` | 50,000 |

### rebuild_attr_query.py

| Parameter | Value |
|---|---|
| `query_smol_size` | 1,024 |
| `quality_min` | `{good, excellent}` |
| Source | `smol-magpie-ultra` config (first N passing filter, stream order) |

### finetune.py

| Parameter | Value |
|---|---|
| `base_model` | `google/gemma-3-1b-pt` |
| `num_train_epochs` | 2 |
| `learning_rate` | 3e-4 |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |
| `lr_scheduler_type` | cosine |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` |

### probe.py

| Parameter | Value |
|---|---|
| `extraction_layer` | 17 (try 14 if R² is low) |
| `pooling` | Last response token position (last index where labels != -100) |
| `ridge_alpha` | 1.0 |
| `val_frac` | 0.20 |

### generate_continued_dataset.py

| Parameter | Value |
|---|---|
| `extraction_layer` | 17 (must match probe.py) |
| `pool_size` | 100,000 new SmolTalk rows to score |
| `quality_size` | 10,000 (top 10% by probe score) |
| `random_size` | 10,000 (drawn from non-quality remainder) |

---

## 7) Hardware Profile

### RTX 3080 (10 GB) — previous, reference only

| Config | VRAM |
|---|---|
| 270M LoRA @ seq=1024, bs=1, grad ckpt | 5.24 GB |
| 1B LoRA @ seq=1024, bs=1, grad ckpt | 8.42 GB |
| 270M or 1B LoRA @ seq=2048+ | OOM |

### RTX 5090 (32 GB) — current

Profiling TBD (`vram.py`). Expected to unlock seq_len ≥ 2048 and larger batch sizes.
Update this section after profiling run.

---

## 8) CPT/FineWeb History (v1/v2 — abandoned)

The original plan was continued pretraining (CPT) on FineWeb. Abandoned because:

- **v1** (lr=1e-4, wd=0.1, 100M tokens): Regressed on all benchmarks (worst: ARC-Easy −0.073).
- **v2** (lr=3e-5, wd=0.01, 200M tokens): Plan was written but never run — pivot to SFT instead.

SAE classifier results from v1 pilot (despite regressed labels):

| Model | Val AUROC | Val Acc |
|---|---|---|
| Logistic regression (10 global stats) | 0.623 | 0.593 |
| L1 logistic on sparse 16k vector | 0.639 | 0.607 |
| **Mean pooling, K=256** | **0.748** | **0.682** |
| Attention pooling, K=256 | 0.742 | 0.678 |

Winner: mean pooling, K=256, no global stats. Architecture carries forward.

---

## 9) Open Questions

1. **Probe R²**: If R² < 0.05 on held-out pool examples, the probe is not learning the attribution direction. Try layer 14 (mid-network) vs layer 17 (later). Possible cause: score.py was run with the old ARC/WinoGrande query — re-run with smol-magpie-ultra attr_query first.

2. **Benchmark sensitivity**: lm-eval on a 1B model over short finetunes has real variance. Compare deltas vs the v1 baseline rather than absolute numbers. If quality and random arms look identical, check probe R² and selection threshold.

3. **Selection rate**: quality arm = top-10% of 100k pool. If signal is weak, try tightening to top-5% (5k examples).

4. **Layer ablation**: Start with layer 17. If R² is low, try 14. Further ablation (9, 12, 20) is possible but probably not worth it before seeing R² results.

5. **seq_len**: After 5090 profiling, consider increasing max_length beyond 1024.


## 11) Theoretical Foundation & Related Work

While linear probes are traditionally used in Mechanistic Interpretability as purely diagnostic tools (to prove a feature exists in superposition), using them as active, production-grade filters draws on two highly mainstream paradigms: **Representation Engineering (RepE)** and **Reward Modeling**. 

This pipeline essentially trains a Data Quality Reward Model using gradient attribution (Bergson) as the ground-truth signal instead of human annotators.

### 1. Representation Engineering: A Top-Down Approach to AI Transparency
**Source:** Zou, A., Phan, L., et al. (Center for AI Safety, 2023). *arXiv:2310.01405*
* **The Concept:** Instead of looking at individual neurons (bottom-up), RepE proves that high-level semantic concepts (like honesty, harmlessness, and utility) are linearly represented across the macro-activation space of an LLM's residual stream. 
* **How it connects to our pipeline:** Zou et al. successfully used linear probes (and PCA) on the middle-to-late hidden states of models like LLaMA-2 to detect and steer complex behaviors. By training a Ridge Regression probe on Layer 14 to predict Bergson scores, we are directly applying RepE to find the linear vector for "Training Data Quality."

### 2. Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
**Source:** Ouyang, L., Wu, J., et al. (OpenAI, 2022). *NeurIPS 2022*
* **The Concept:** The foundational paper for RLHF. To scale human preferences, OpenAI trained a "Reward Model" to predict a scalar quality score for any given text generation. 
* **How it connects to our pipeline:** Standard Reward Models (like those in InstructGPT and Llama 2) are structurally identical to our proposed `probe.py`. They consist of a base Transformer where the final vocabulary unembedding matrix is replaced by a single linear projection head. We are using the exact same architecture, but our "human preference" is replaced by the rigorous, mathematical gradient attribution score from Phase 2.

### 3. The Internal State of an LLM Knows When It's Lying
**Source:** Azaria, A., & Mitchell, T. (2023). *Findings of the Association for Computational Linguistics: EMNLP 2023*
* **The Concept:** The authors trained a simple classifier (SAPLMA) on the hidden layer activations of an LLM to predict whether the statement the LLM was processing was true or false. 
* **How it connects to our pipeline:** This paper proves that LLMs internally compute rigorous semantic evaluations (like factual accuracy) in their hidden states, and that extracting these states via a probe is vastly more reliable than relying on the model's output probabilities (which are skewed by token length and frequency). This validates our choice to probe the residual stream directly rather than using the model as an "LLM-as-a-judge" via prompting.