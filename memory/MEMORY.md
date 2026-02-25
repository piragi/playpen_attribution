# Attribution Pipeline — Project Memory

## Project overview
Interpretability research codebase: LoRA SFT + Bergson attribution scoring.
Pipeline: build_sft_data → finetune → rebuild_attr_query → score → probe → generate_continued_dataset → finetune (ablation arms)

## Key design decisions (post-refactor)
- CONFIG dicts in each file (not YAML), intentional for researcher readability
- Token-budget matching for random arm was dropped (too complex, not empirically critical)
- Quality-labels ablation in probe.py was removed (no longer used)
- Ablation arms: quality_{probe}, quality_{probe}_50pct, random (uniform), label_good_excellent

## Shared helpers live in pipeline_common.py
- `load_tokenizer(base_model)` — IT model + pad_token setup
- `load_model_with_hook(base_model, adapter_path, layer, dtype, device)` — returns (model, captured)
- `mask_prompt(messages, tokenizer, max_length)` — chat template tokenization with label masking
- `get_magpie_score(row)` — extract quality score from SmolTalk row
- `ensure_hf_home_env()` — called in each script's module level

## Important: mask_prompt / get_magpie_score
Previously in build_sft_data.py as _mask_prompt/_get_magpie_score (with underscore).
Now in pipeline_common.py without underscore. rebuild_attr_query.py and generate_continued_dataset.py import from there directly (not from build_sft_data).

## finetune.py CLI
No subcommands for train (default), `merge` is a positional subcommand:
- `uv run finetune.py [train args...]`
- `uv run finetune.py merge --adapter-path ...`

## Line counts (post-refactor)
pipeline_common: 181, build_sft_data: 175, rebuild_attr_query: 161,
probe: 202, generate_continued_dataset: 259, finetune: 250
Total: ~1,228 (was ~1,700)
