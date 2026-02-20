# Playpen Attribution — Active Notes

Last updated: 2026-02-20

---

## 1) Overall Goal

Test whether Bergson attribution can identify FineWeb pretraining chunks that improve
downstream benchmark performance better than matched random selection under a comparable
compute budget.

Benchmarks: WinoGrande, ARC-Challenge, ARC-Easy, HellaSwag, PIQA (lm-eval harness, log-prob scoring).

---

## 2) Active Pipeline Scripts

| Script | Role |
|---|---|
| `pretrain.py` | Full-model continued pretraining on FineWeb |
| `build_pretrain_query.py` | Builds score_pool (97K FineWeb chunks) + attr_query (benchmark train examples) |
| `score.py` | Bergson gradient-based attribution pipeline |
| `eval_harness.py` | lm-eval benchmark evaluation (ARC, HellaSwag, WinoGrande, PIQA) |
| `sae_analysis.py` | SAE feature extraction (Phase 4, not yet started) |
| `train_bidir_classifier.py` | SAE-based attribution rank predictor (Phase 4, not yet started) |

---

## 3) Critical Bergson Rule

Pass **pre-tokenized data** (`input_ids` + `labels` columns) directly to score.py using
`--tokenization-mode bergson_chat`. Bergson skips its internal tokenization when `input_ids`
is already present in the dataset, avoiding any chat-template or tokenization mismatch.

- Pool (FineWeb chunks): `labels = input_ids` — full causal supervision, no masking.
- Query (benchmark examples): prompt tokens masked with `-100`, only answer tokens supervised.

---

## 4) Best Attribution Config (current)

```
--score-mode mean
--preconditioning-mode query
--projection-dim 32
--tokenization-mode bergson_chat
```

No `--adapter-path` needed for full pretrained models — score.py falls back to `--base-model`
automatically when `--adapter-path` is omitted.

---

## 5) Phase Structure

### Phase 1 — Continued Pretraining (Done)

Continue pretrain Gemma-3-270m on 100M tokens of FineWeb (CC-MAIN-2024-10).

```bash
uv run pretrain.py \
  --base-model google/gemma-3-270m \
  --output-dir runs/pretrain_270m_v1
```

Result: `runs/pretrain_270m_v1/`
- train_loss: 3.076, 6103 steps, ~3.2 hours
- Slight benchmark regression vs base (expected — catastrophic forgetting on small LR shift from Google's optimum)

### Phase 2 — Build Attribution Datasets (Done)

Rebuild the exact same FineWeb chunks used in Phase 1 (same stream, same config → deterministic)
and format WinoGrande + ARC-Challenge train splits as prompt/completion pairs for attribution.

```bash
uv run build_pretrain_query.py
```

Result: `runs/pretrain_attribution_v1/`
- `data/score_pool`: 97,658 FineWeb chunks × 1024 tokens (pre-tokenized, full causal supervision)
- `data/attr_query`: 250 WinoGrande + 250 ARC-Challenge train examples (answer tokens only supervised)
- `manifest.json`

### Phase 3 — Attribution (Next)

Score every FineWeb pool chunk by its gradient influence on the benchmark query loss.
Select top-5000 and matched random-5000 for continuation.

```bash
uv run score.py \
    --manifest runs/pretrain_attribution_v1/manifest.json \
    --base-model runs/pretrain_270m_v1 \
    --pool-split score_pool \
    --query-split attr_query \
    --tokenization-mode bergson_chat \
    --score-mode mean \
    --preconditioning-mode query \
    --projection-dim 32 \
    --subset-k 5000 \
    --output-dir runs/pretrain_attribution_v1/attribution_mean_k5000
```

### Phase 4 — Continuation Pretraining (Next after Phase 3)

Continue pretraining from `runs/pretrain_270m_v1` separately on top-k and random-k subsets.
Both arms start from the same checkpoint with the same compute budget.

Adapt `pretrain.py` to load from a saved HF dataset (the subset) instead of streaming FineWeb.

```bash
# Top-k arm
uv run pretrain.py \
  --base-model runs/pretrain_270m_v1 \
  --output-dir runs/pretrain_attribution_v1/continue_top_k5000

# Random-k arm
uv run pretrain.py \
  --base-model runs/pretrain_270m_v1 \
  --output-dir runs/pretrain_attribution_v1/continue_random_k5000
```

Then evaluate both with eval_harness.py on WinoGrande + ARC test sets.
Gate: top-k must beat random-k before proceeding to Phase 5.

### Phase 5 — SAE Fingerprint (Future)

Once Phase 4 validates attribution, train a SAE-based fast detector to predict attribution rank
from GemmaScope SAE activations — enabling cheap data selection without re-running Bergson.

---

## 6) Benchmark Results

### Base vs Pretrained (full eval, 5 benchmarks)

| Task | Metric | Base (gemma-3-270m) | Pretrained (270m_v1) | Delta |
|---|---|---|---|---|
| arc_challenge | acc_norm | 0.2833 | 0.2705 | -0.013 |
| arc_easy | acc_norm | 0.5694 | 0.4966 | **-0.073** |
| hellaswag | acc_norm | 0.4136 | 0.3937 | -0.020 |
| winogrande | acc | 0.5359 | 0.5288 | -0.007 |
| piqa | acc_norm | 0.6823 | 0.6681 | -0.014 |

Regression is expected: 100M FineWeb tokens nudges the model off Google's carefully tuned
optimum. WinoGrande/PIQA (commonsense reasoning) are robust; ARC-Easy (factual recall)
regresses most because FineWeb lacks curated science content.

Eval artifacts: `runs/pretrain_270m_v1/eval_base_full.json`, `runs/pretrain_270m_v1/eval_pretrained_full.json`

---

## 7) Key Technical Notes

### lm-eval + Gemma3 fix

lm-eval 0.4.11 passes `dtype=` to `Gemma3ForCausalLM.__init__` instead of `torch_dtype=`,
causing a TypeError. Fix applied directly to the installed package:

```
.venv/lib/python3.13/site-packages/lm_eval/models/huggingface.py
lines 635, 718: dtype= → torch_dtype=
```

Also remove any `torch_dtype=` from eval_harness.py model_args string (would cause duplicate kwarg).

### FineWeb pool is deterministic

`HuggingFaceFW/fineweb` with `CC-MAIN-2024-10` streams in a fixed order (no shuffle).
Running `build_pretrain_query.py` again with the same `--max-tokens` reproduces the identical
97,658 chunks used in Phase 1 training.

### Benchmark query format

- **WinoGrande**: prompt = sentence with blank + both options listed; completion = correct fill-in word
- **ARC-Challenge**: prompt = question + lettered choices; completion = correct letter (e.g. ` A`)
- Both use `max_query_length=256`; ~2 supervised tokens per example (answer + EOS)

---

## 8) Canonical Artifacts

| Path | Contents |
|---|---|
| `runs/pretrain_270m_v1/` | Phase 1 pretrained model + tokenizer |
| `runs/pretrain_270m_v1/pretrain_metrics.json` | Training metrics |
| `runs/pretrain_270m_v1/eval_base_full.json` | Base model benchmark scores |
| `runs/pretrain_270m_v1/eval_pretrained_full.json` | Pretrained model benchmark scores |
| `runs/pretrain_attribution_v1/manifest.json` | Attribution manifest (pool + query paths) |
| `runs/pretrain_attribution_v1/data/score_pool` | 97,658 pre-tokenized FineWeb chunks |
| `runs/pretrain_attribution_v1/data/attr_query` | 500 benchmark train examples (pre-tokenized) |
