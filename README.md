# playpen-attribution (minimal reset)

Current active scope:
1. build a single-game dataset (`taboo::WordGuesser`)
2. train a base LoRA adapter
3. score training points with Bergson
4. compare continuation on `top_k` vs `matched_random_k`

## Active scripts
1. `dataset.py` - builds manifest + dataset splits (`train_base`, `attr_query`, `eval`)
2. `finetune.py` - base training, continuation training, and adapter merge
3. `score.py` - Bergson reduce/score + diagnostics + subset construction

## Core commands
```bash
uv run dataset.py
uv run finetune.py train --manifest runs/simple_wordguesser_v1/manifest.json
uv run score.py --manifest runs/simple_wordguesser_v1/manifest.json --adapter-path runs/simple_wordguesser_v1/base_adapter
uv run finetune.py train --manifest runs/simple_wordguesser_v1/attribution/continuation_manifest.json --train-split top_k --resume-adapter runs/simple_wordguesser_v1/base_adapter --output-dir runs/simple_wordguesser_v1/continue_top_k
uv run finetune.py train --manifest runs/simple_wordguesser_v1/attribution/continuation_manifest.json --train-split matched_random_k --resume-adapter runs/simple_wordguesser_v1/base_adapter --output-dir runs/simple_wordguesser_v1/continue_random_k
```

## Legacy
Previous scripts are archived in `legacy/previous_pipeline_2026_02_17/`.
