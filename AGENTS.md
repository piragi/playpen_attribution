# Playpen Attribution — Active Notes

Last updated: 2026-02-22

---

## 1) Overall Goal

Prove that **residual stream probing + gradient-based data attribution can identify higher-quality SFT training data**, and that training on this selected data produces better instruction-following models than randomly sampled data.

Concretely:
1. **Build data**: Tokenize smol-magpie-ultra → train/val/attr_pool splits (in-distribution).
2. **Finetune scoring adapter**: LoRA SFT on train split → adapter used by Bergson attribution.
3. **Attribution**: Bergson scores each attr_pool example by influence toward a high-quality attr_query (also from smol-magpie-ultra, good/excellent only, disjoint from pool).
4. **Probe**: Fit Ridge Regression on layer-17 residual stream embeddings of attr_pool → predict attribution scores. Also run a quality-label ablation (binary good/excellent probe, no attribution needed).
5. **Generate continuation datasets**: Apply probe to 200k new smol-magpie-ultra rows → select top-50k (quality arm) + random-50k (random arm).
6. **Finetune 3 arms from base**: SFT (in-distribution train), quality arm, random arm.
7. **Eval**: ARC, HellaSwag, WinoGrande, IFEval, GSM8K — compare arms.

Core claim: **Residual stream probing makes data attribution scalable** — one forward pass + linear probe replaces expensive per-example attribution at inference time.

Baseline to beat: **random selection from the same pool**. Quality arm must outperform random arm to validate the claim.

---

## 2) Key Design Decisions & Rationale

### Why SmolLM2-1.7B (not Gemma)?

- SAEs are no longer part of the pipeline — Gemma was chosen for GemmaScope SAE compatibility, which is now irrelevant.
- SmolLM2-1.7B is HuggingFaceTB's own model, well-matched to SmolTalk (their dataset), making the SFT signal cleaner.
- Native SDPA support (no flash_attn package needed), fits comfortably in 31 GB VRAM at seq_len=2048.

### Why smol-magpie-ultra for everything (v2 design)?

v1 used SmolTalk `all` for pool/train but smol-magpie-ultra for the query. This created a distribution mismatch — the attribution query was in a different domain than the training data being scored. v2 uses smol-magpie-ultra for query, attr_pool, AND train splits (disjoint rows), eliminating the mismatch.

### Why LoRA (not full fine-tuning)?

- Gradient computation for attribution concentrates on LoRA adapter parameters, keeping the signal focused.
- Fits in much less VRAM; faster iteration.

### Attribution query: smol-magpie-ultra (good/excellent, disjoint from pool)

The attr_query defines "quality" in gradient space — attribution scores measure how much a pool example helped produce behaviour like the query.

- 4,096 rows, quality ∈ {good, excellent}, category filter = None (all categories).
- Built by `rebuild_attr_query.py`, which skips the rows already consumed by `build_sft_data.py` to ensure no overlap with attr_pool.

### No residualization

Bergson uses `unit_normalize=True` + query-side TRAK preconditioning. Length correlation in scores is content-based, not magnitude artifact. Do **not** regress out length.

---

## 3) Active Pipeline Scripts

| Script | Role |
|---|---|
| `build_sft_data.py` | Tokenize smol-magpie-ultra, mask prompts, produce train/val/attr_pool splits + manifest |
| `rebuild_attr_query.py` | Build attr_query from smol-magpie-ultra (quality-filtered, disjoint from pool) |
| `finetune.py` | LoRA SFT on pre-tokenized data; saves IT tokenizer alongside adapter |
| `eval_harness.py` | lm-eval benchmark evaluation (ARC, HellaSwag, WinoGrande, IFEval, GSM8K) |
| `score.py` | Bergson gradient-based attribution; pool vs attr_query; supports PEFT adapters |
| `probe.py` | Extract layer-17 residual stream from pool → fit Ridge probe to attribution scores (or quality labels) |
| `generate_continued_dataset.py` | Score 200k new smol-magpie-ultra rows with probe → quality arm (top 50k) + random baseline (50k) |
| `vram.py` | VRAM profiler for SmolLM2-1.7B under training and inference configs |

---

## 4) Critical Technical Notes

### Base model + IT tokenizer pattern

```
Training:    AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")          ← base weights
Tokenizer:   AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")        ← chat template
```

IT model naming convention for SmolLM2: base ends in `SmolLM2-1.7B` → instruct is `SmolLM2-1.7B-Instruct`
(Gemma pattern was `-pt` → `-it`; SmolLM2 has no `-pt` suffix so append `-Instruct` instead.)

After training, the IT tokenizer is saved alongside the LoRA adapter. Downstream scripts load tokenizer from the adapter directory.

### resolve_model_path() pattern

`from_pretrained` with a HuggingFace model ID can fail when `HF_HOME` is non-standard. All scripts use `resolve_model_path()` which checks the local cache at `/workspace/.hf_home/hub/models--{org}--{name}/snapshots/` and returns the local snapshot path directly:

```python
def resolve_model_path(model_id: str) -> str:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = "models--" + model_id.replace("/", "--")
    snapshots_dir = hf_home / "hub" / slug / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[-1])
    return model_id
```

### Attention implementation

All scripts use `attn_implementation="sdpa"` (PyTorch built-in scaled dot product attention). Do **not** use `"eager"` (slow) or `"flash_attention_2"` (package not installed).

### Bergson rule

Pass **pre-tokenized data** (`input_ids` + `labels` columns) directly. Bergson skips internal tokenization when `input_ids` is already present.

- Pool (attr_pool): prompt tokens masked with `-100`, only assistant tokens supervised.
- Query (attr_query): same masking — smol-magpie-ultra examples formatted with IT chat template.

### score.py + PEFT

Pass `--adapter-path <adapter_dir>` to `score.py`. It calls `ensure_adapter_config()` which injects a `config.json` into the adapter directory so Bergson can load the full model via PEFT.

score.py defaults: `unit_normalize=True`, `query_preconditioner` (TRAK KFAC eigendecomposition on query side), `projection_dim=32`, `mixing_coefficient=0.99`. These are correct — do not change.

The manifest's `score_pool` key aliases `attr_pool` — score.py uses `--pool-split score_pool` by default.

### probe.py — two modes

- `use_quality_labels=False` (default): trains probe to predict Bergson attribution scores (regression). Saves `probe.pkl`.
- `use_quality_labels=True`: trains probe on binary labels (good/excellent=1, else=0) using the `quality` field in attr_pool. Saves `probe_quality_labels.pkl`. Reports AUC + accuracy instead of R².

Pool embeddings are cached to `pool_embeddings.npy` — if this file exists, probe.py skips model loading and reuses it. Delete the file to force re-extraction.

### build_sft_data.py — attr_query handling

When `query_smol_size=0` (v2 default), `build_sft_data.py` skips saving the attr_query split and omits it from the manifest. `rebuild_attr_query.py` populates it separately. This avoids a KeyError when the attr_query dataset is empty.

### rebuild_attr_query.py — output path

Derives the output path from `manifest["splits"]["attr_query"]["path"]` if it exists, otherwise defaults to `<manifest_dir>/data/attr_query`. This allows it to work even when `build_sft_data.py` did not create the attr_query entry.

### eval_harness.py

When `--adapter-path` is set, `build_model_args()` automatically appends `tokenizer=<adapter_path>` so lm-eval picks up the IT tokenizer from the adapter directory. Always pass `--apply-chat-template` for SFT model evaluation.

### SmolTalk streaming

Always use `streaming=True` in `load_dataset` for SmolTalk. Without it, the full dataset downloads before processing begins.

### In-training eval disabled

SmolLM2's 49k vocabulary causes OOM when computing eval loss at long seq_len. Use `eval_harness.py` post-training instead.

---

## 5) Data Layout (v2)

```
runs/smoltalk_v2/
  manifest.json                 ← paths, row counts, raw_rows_consumed
  data/
    train/        45,000 rows — SFT training (smol-magpie-ultra)
    val/           2,000 rows — SFT validation
    attr_pool/    15,000 rows — attribution pool (scored by Bergson)
    attr_query/    4,096 rows — smol-magpie-ultra quality∈{good,excellent}, disjoint from pool
  adapter/        LoRA scoring adapter + IT tokenizer (SmolLM2-1.7B)
  scores/
    row_diagnostics.jsonl       ← per-pool-row Bergson scores
    summary.json
    subsets/
  probe/
    pool_embeddings.npy         ← (15000, 2048) layer-17 residual stream for attr_pool
    probe.pkl                   ← fitted Ridge regression (attribution scores)
    probe_quality_labels.pkl    ← fitted Ridge regression (binary quality labels)
    probe_meta.json
  continuation/
    quality/    — top-50k by probe score (from 200k new smol-magpie-ultra rows)
    random/     — 50k from non-quality remainder
  sft_adapter/     LoRA adapter trained on train split (in-distribution baseline)
  quality_adapter/ LoRA adapter trained on quality arm
  random_adapter/  LoRA adapter trained on random arm
  eval_sft.json
  eval_quality.json
  eval_random.json
```

---

## 6) Default Hyperparameters

### build_sft_data.py

| Parameter | Value |
|---|---|
| `dataset_name` | `HuggingFaceTB/smoltalk` |
| `dataset_config` | `smol-magpie-ultra` |
| `base_model` | `HuggingFaceTB/SmolLM2-1.7B` |
| `tokenizer_model` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `max_length` | 2048 |
| `train_size` | 45,000 |
| `val_size` | 2,000 |
| `attr_pool_size` | 15,000 |
| `query_smol_size` | 0 (disabled; rebuild_attr_query.py handles it) |

### rebuild_attr_query.py

| Parameter | Value |
|---|---|
| `query_smol_size` | 4,096 |
| `quality_min` | `{good, excellent}` |
| `category_filter` | None (all categories) |
| Source | `smol-magpie-ultra`, skips rows consumed by build_sft_data.py |

### finetune.py

| Parameter | Value |
|---|---|
| `base_model` | `HuggingFaceTB/SmolLM2-1.7B` |
| `num_train_epochs` | 2 |
| `learning_rate` | 3e-4 |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |
| `lr_scheduler_type` | cosine |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 4 (effective batch = 32) |
| `gradient_checkpointing` | True |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` |

### probe.py

| Parameter | Value |
|---|---|
| `extraction_layer` | 17 |
| `pooling` | Last response token (last index where labels != -100) |
| `ridge_alpha` | 1.0 |
| `val_frac` | 0.20 |
| `batch_size` | 64 |

### generate_continued_dataset.py

| Parameter | Value |
|---|---|
| `extraction_layer` | 17 (must match probe.py) |
| `pool_size` | 200,000 new smol-magpie-ultra rows to score |
| `quality_size` | 50,000 (top 25% by probe score) |
| `random_size` | 50,000 (drawn from non-quality remainder) |
| `batch_size` | 64 |

---

## 7) Hardware Profile — RTX 5090 (31.36 GB)

Measured with `vram.py`, SmolLM2-1.7B, sdpa, bfloat16, seq_len=2048.

| Config | Peak VRAM |
|---|---|
| Model load | 3.19 GB |
| Inference bs=1 | 0.20 GB |
| LoRA train bs=1, no grad_ckpt | 7.17 GB |
| LoRA train bs=2, no grad_ckpt | 14.13 GB |
| LoRA train bs=4, no grad_ckpt | OOM |
| LoRA train bs=1, grad_ckpt ON | 1.15 GB |
| LoRA train bs=2, grad_ckpt ON | 2.83 GB |
| LoRA train bs=4, grad_ckpt ON | 5.66 GB |
| LoRA train bs=8, grad_ckpt ON | 11.33 GB |
| Inference bs=16 | 3.12 GB |
| Inference bs=32 | 6.25 GB |
| Inference bs=64 | 12.50 GB |
| Inference bs=128 | OOM |

**Recommended settings**: training with grad_ckpt + bs=8; probe/generate with bs=64.

---

## 8) Open Questions

1. **Probe R²**: If R² < 0.05 on held-out pool examples, the probe is not learning the attribution direction. Try layer 14 (mid-network) vs layer 17.

2. **Quality labels ablation**: Does the binary quality label probe (no attribution needed) achieve comparable selection quality to the attribution probe? If yes, attribution is redundant and quality labels alone suffice.

3. **Benchmark sensitivity**: lm-eval on a 1.7B model has real variance. Compare deltas vs SFT arm rather than absolute numbers. If quality and random arms look identical, check probe R² and selection threshold.

4. **Selection rate**: quality arm = top-50k of 200k pool (~25%). If signal is weak, try tightening to top-10% (20k examples).

---

## 9) CPT/FineWeb History (v1/v2 — abandoned)

The original plan was continued pretraining (CPT) on FineWeb. Abandoned because:

- **v1** (lr=1e-4, wd=0.1, 100M tokens): Regressed on all benchmarks (worst: ARC-Easy −0.073).
- **v2** (lr=3e-5, wd=0.01, 200M tokens): Plan was written but never run — pivot to SFT instead.

---

## 10) Theoretical Foundation & Related Work

While linear probes are traditionally used in Mechanistic Interpretability as purely diagnostic tools, using them as active filters draws on two mainstream paradigms: **Representation Engineering (RepE)** and **Reward Modeling**.

This pipeline essentially trains a Data Quality Reward Model using gradient attribution (Bergson) as the ground-truth signal instead of human annotators.

### Representation Engineering (Zou et al., 2023 — arXiv:2310.01405)

Proves that high-level semantic concepts are linearly represented across the residual stream. By training a Ridge Regression probe on layer-17 hidden states to predict Bergson scores, we apply RepE to find the linear vector for "Training Data Quality."

### InstructGPT / RLHF (Ouyang et al., NeurIPS 2022)

Standard Reward Models are structurally identical to our probe — a base Transformer with the unembedding matrix replaced by a linear projection head. We use the same architecture, but Bergson attribution replaces human preference labels.

### SAPLMA (Azaria & Mitchell, EMNLP 2023)

Proves that LLMs internally compute semantic evaluations in hidden states, and probing residual stream is more reliable than output probabilities. Validates probing over LLM-as-a-judge approaches.
