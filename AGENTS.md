# Playpen Attribution — Active Notes

Last updated: 2026-02-21

---

## 1) Overall Goal

Prove that **SAE feature analysis combined with data attribution can identify higher-quality SFT training data**, and that training on this selected data produces better instruction-following models than randomly sampled or heuristically filtered data.

Concretely:
1. **Phase 1**: LoRA SFT of Gemma-3-270m (base) on SmolTalk → establish a training run whose samples can be attributed.
2. **Phase 2**: Compute gradient-based data attribution (Bergson) → label each training example by influence on validation loss.
3. **Phase 3**: Extract SAE features (GemmaScope 2) from each example using the **base model** → train a lightweight classifier that predicts attribution from SAE features.
4. **Phase 4**: Apply the classifier to a larger SmolTalk pool → select top-scoring data → continue fine-tuning on it → compare against random and Magpie-score baselines.

Core claim: **SAE fingerprinting makes data attribution scalable** — one forward pass through model + SAE + small classifier replaces expensive gradient computation.

Baseline to beat: **Magpie quality score** (0–5, built into SmolTalk rows). If SAE-selected data outperforms Magpie-filtered data, we have a result.

Benchmarks: ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande, MMLU (SmolTalk paper set; lm-eval harness, log-prob scoring, `--apply-chat-template` required for SFT models).

---

## 2) Key Design Decisions & Rationale

### Why Gemma-3-270m base (not IT)?

- **GemmaScope 2** SAEs are trained on base model activations. Using the IT model for SAE extraction creates a distribution mismatch — SAE features are miscalibrated on IT activations.
- Base model has **higher gradient variance** → cleaner attribution labels. IT has already learned instruction structure, so gradients are smaller and noisier for attribution.
- We borrow the IT tokenizer for its chat template (same vocabulary, same token IDs — the only difference is `chat_template` in `tokenizer_config.json`). This is the standard approach for SFT from a base model.

### Why LoRA (not full fine-tuning)?

- Gradient computation for attribution concentrates on LoRA adapter parameters, keeping the signal focused.
- Fits in much less VRAM; faster iteration.
- LoRA adapters are the de facto standard for SFT at this scale.

### Why SmolTalk?

- High quality variance: mix of Magpie-Ultra, Smol-Rewrite, Smol-Constraints, etc. Magpie quality scores (0–5) give a natural baseline to beat.
- Realistic instruction-following data — direct application of the technique.
- Previously, CPT on FineWeb regressed on all benchmarks regardless of hyperparameters (see §6). SFT on instruction data is a cleaner experimental setup.

### Why the `all` config, not `smol-magpie-ultra`?

`smol-magpie-ultra` is a separate dataset, not a config of `HuggingFaceTB/smoltalk`. Available configs: `all`, `smol-magpie-ultra` (as separate HF dataset), `smol-constraints`, `smol-rewrite`, etc. Use `all` for maximum quality variance.

---

## 3) Active Pipeline Scripts

| Script | Role |
|---|---|
| `build_sft_data.py` | Tokenize SmolTalk with IT chat template, mask prompts, produce train/val/attr_pool/attr_query splits |
| `finetune.py` | LoRA SFT on pre-tokenized data; saves IT tokenizer alongside adapter |
| `eval_harness.py` | lm-eval benchmark evaluation (ARC, HellaSwag, WinoGrande, MMLU) |
| `score.py` | Bergson gradient-based attribution pipeline; supports PEFT adapters |
| `sae_analysis.py` | SAE feature extraction from base model (GemmaScope 2) |
| `train_bidir_classifier.py` | SAE fingerprint classifier (predicts attribution from SAE features) |

---

## 4) Critical Technical Notes

### Base model + IT tokenizer pattern

```
Training:    AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")  ← base weights
Tokenizer:   AutoTokenizer.from_pretrained("google/gemma-3-270m-it")      ← chat template
```

After training, the IT tokenizer is saved alongside the LoRA adapter. Downstream scripts
(`eval_harness.py`, `score.py`) load tokenizer from the adapter directory.

### Bergson rule

Pass **pre-tokenized data** (`input_ids` + `labels` columns) directly. Bergson skips internal
tokenization when `input_ids` is already present.

- Pool (SmolTalk attr_pool): prompt tokens masked with `-100`, only assistant tokens supervised.
- Query (attr_query): same masking — ARC-Challenge + WinoGrande formatted as instruct examples.

### Prompt masking

`_mask_prompt()` in `build_sft_data.py` uses cumulative prefix tokenization to find exact token
boundaries per assistant turn. Multi-turn conversations are handled correctly.

### score.py + PEFT

Pass `--adapter-path <adapter_dir>` to `score.py`. It calls `ensure_adapter_config()` which
injects a `config.json` (base model architecture) into the adapter directory so Bergson can
load the full model via PEFT.

### eval_harness.py + IT tokenizer

When `--adapter-path` is set, `build_model_args()` automatically appends
`tokenizer=<adapter_path>` so lm-eval picks up the IT tokenizer (with chat template) from the
adapter directory. Always pass `--apply-chat-template` for SFT model evaluation.

### lm-eval + Gemma3 fix

lm-eval 0.4.11 passes `dtype=` to `Gemma3ForCausalLM.__init__` instead of `torch_dtype=`,
causing a TypeError. Fix applied directly to the installed package:

```
.venv/lib/python3.13/site-packages/lm_eval/models/huggingface.py
lines 635, 718: dtype= → torch_dtype=
```

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
    train/        50,000 rows — SFT training
    val/          2,000 rows  — SFT validation (Bergson query)
    attr_pool/    20,000 rows — attribution pool (scored by Bergson)
    attr_query/   1,024 rows  — ARC-Challenge + WinoGrande as instruct examples
  adapter/        LoRA adapter + IT tokenizer
  scores/
    row_diagnostics.jsonl
    summary.json
    subsets/
```

---

## 6) Default Hyperparameters

### build_sft_data.py

| Parameter | Value |
|---|---|
| `dataset_name` | `HuggingFaceTB/smoltalk` |
| `dataset_config` | `all` |
| `base_model` | `google/gemma-3-270m` |
| `tokenizer_model` | `google/gemma-3-270m-it` |
| `max_length` | 1024 (to be updated after 5090 profiling) |
| `train_size` | 50,000 |
| `val_size` | 2,000 |
| `attr_pool_size` | 20,000 |
| `query_per_task` | 512 |

### finetune.py

| Parameter | Value |
|---|---|
| `base_model` | `google/gemma-3-270m` |
| `num_train_epochs` | 2 |
| `learning_rate` | 3e-4 |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |
| `lr_scheduler_type` | cosine |
| `per_device_train_batch_size` | 1 (to be updated after 5090 profiling) |
| `gradient_accumulation_steps` | 16 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` |

Effective batch size = per_device_train_batch_size × gradient_accumulation_steps.
After 5090 profiling, increase batch size and reduce accumulation steps accordingly.

---

## 7) Hardware Profile

### RTX 3080 (10 GB) — previous, reference only

| Config | VRAM |
|---|---|
| 270M LoRA @ seq=1024, bs=1, grad ckpt | 5.24 GB |
| 1B LoRA @ seq=1024, bs=1, grad ckpt | 8.42 GB |
| 270M or 1B LoRA @ seq=2048+ | OOM |

### RTX 5090 (32 GB) — current

Profiling TBD (`vram_profile.py`). Expected to unlock:
- seq_len ≥ 2048 for both models
- per_device_train_batch_size ≥ 4 for 270M

Update this section after profiling run.

---

## 8) CPT/FineWeb History (v1/v2 — abandoned)

The original plan was continued pretraining (CPT) on FineWeb. This was abandoned because:

- **v1** (lr=1e-4, wd=0.1, 100M tokens): Regressed on all benchmarks (worst: ARC-Easy −0.073).
- **v2** (lr=3e-5, wd=0.01, 200M tokens): Plan was written but never run — pivot to SFT instead.

Root causes of CPT regression:
1. LR too high (should be eta_min of original schedule, ~10x below original peak)
2. Weight decay too aggressive for CPT
3. No data replay

SAE classifier results from v1 pilot (despite regressed labels):

| Model | Val AUROC | Val Acc |
|---|---|---|
| Logistic regression (10 global stats) | 0.623 | 0.593 |
| L1 logistic on sparse 16k vector | 0.639 | 0.607 |
| **Mean pooling, K=256** | **0.748** | **0.682** |
| Attention pooling, K=256 | 0.742 | 0.678 |

Winner: mean pooling, K=256, no global stats. This architecture carries forward to v3.

---

## 9) Open Questions

1. **seq_len**: After 5090 profiling, decide whether to increase max_length beyond 1024. Higher seq_len passes more SmolTalk examples through the filter and includes more complex multi-turn conversations.

2. **Pool size**: Currently scoring 20k examples from attr_pool against 2k val. Is 20k enough to train a good classifier? May need to increase attr_pool_size.

3. **SAE layer choice**: v1 used layer 12 only. The classifier may benefit from ablating layers 9, 12, 15, or concatenated. Worth doing if AUROC < 0.75.

4. **Continuation experiment design**: After classifier training, should the continuation fine-tune start from the original LoRA adapter (Phase 1) or from base? Starting from the adapter is more natural (simulates real data curation).
