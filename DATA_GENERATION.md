# Data Generation (Taboo `WordGuesser`)

This note documents where Guesser data comes from, where files are written, and how API wiring works when collecting new interactions with two models (`Describer` + `Guesser`).

## 1) What we need for this project

We want dataset rows for:
- `meta.game == "taboo"`
- `meta.game_role == "WordGuesser"`
- `meta.outcome != "aborted"`

This matches the active pipeline in `dataset.py` (default `--game taboo --role WordGuesser`).

## 2) Current in-repo data path (no API needed)

Using:

```bash
uv run dataset.py
```

`dataset.py` reads:
- Hugging Face dataset: `colab-potsdam/playpen-data`
- config: `interactions`

and writes:
- `runs/simple_wordguesser_v1/manifest.json`
- `runs/simple_wordguesser_v1/data/train_base`
- `runs/simple_wordguesser_v1/data/attr_query`
- `runs/simple_wordguesser_v1/data/eval`

No provider API key is needed for this path (only HF dataset download access).

## 3) Where raw generated interactions live in Playpen/Clembench

When generating fresh game runs via Playpen, the raw logs are written under:
- `playpen-records/.../interactions.json`
- sibling request traces like `*.requests.json`

The clemcore results layout is instance-based under run/game/experiment/instance directories, so the robust way to consume is:
- glob: `playpen-records/**/interactions.json`

Important role mapping for Taboo:
- game roles are `["Describer", "Guesser"]` in Clembench Taboo
- Playpen starter trainers assign `[teacher, learner]`
- therefore: teacher = Describer, learner = Guesser

So if we train/collect from Guesser perspective, the learner model is the Guesser side.

## 4) Conversion script and output location

Playpen includes a converter:
- `playpen/examples/trl/data_utils.py`

It scans `**/interactions.json` and writes:
- `playpen/examples/trl/results.jsonl`

Each JSONL row includes:
- `messages`
- `meta.game`
- `meta.game_role`
- `meta.outcome`
- `meta.model`
- `meta.task_id`, etc.

To keep only the target data for this project, filter that JSONL to:
- `meta.game == "taboo"`
- `meta.game_role == "WordGuesser"`
- `meta.outcome != "aborted"`

## 5) API/key wiring for two-model data generation

### 5.1 One provider (simplest)

If both models are on one provider/backend (for example both OpenAI), one key block is enough in `key.json`.

Example:

```json
{
  "openai": {
    "organisation": "org_xxx",
    "api_key": "sk-..."
  }
}
```

### 5.2 Mixed providers

If learner and teacher use different backends/providers, add both entries in the same `key.json` (for example `openai` + `anthropic`, or `openai` + `openrouter`).

### 5.3 OpenRouter / OpenAI-compatible

Current clemcore backends support:
- `openrouter` (dedicated backend)
- `openai_compatible` (requires `base_url` + `api_key`)

Note: some older templates show `generic_openai_compatible`; backend code in current clemcore expects `openai_compatible`.

## 6) Minimal operational flow to generate Guesser data

1. Configure `key.json` for the backend(s) of learner/teacher.
2. Run Playpen with both models (`-l learner`, `-t teacher`) on Taboo.
3. Collect logs from `playpen-records/**/interactions.json`.
4. Convert with `playpen/examples/trl/data_utils.py` to JSONL.
5. Filter to `taboo + WordGuesser + non-aborted`.
6. Feed into this repo pipeline (or keep using `uv run dataset.py` with HF `interactions` if no new generation is needed).

## 6.1) In-repo OpenRouter smoke generator

For quick local synthetic generation without wiring full Playpen runs, this repo now includes:

- `generate_synthetic_openrouter.py`

It:
- reads `OPENROUTER_API_KEY` or `OPENROUTER_API` from `.env`
- runs Taboo-style `Describer` + `Guesser` rollouts
- uses Clembench Taboo instances by default (`target_source=instances`):
  - source: `taboo/in/instances.json`
  - enforces forbidden target + related-word constraints for clues
  - retries invalid clues before continuing
- records per-call token usage and cost from OpenRouter model pricing
- writes:
  - `games.jsonl` (full rollout logs + usage)
  - `rows_wordguesser.jsonl` (training-ready `prompt`/`completion` rows)
  - `summary.json` (aggregated success/cost stats)

Example smoke run:

```bash
uv run generate_synthetic_openrouter.py \
  --output-dir runs/simple_wordguesser_v1/synthetic_openrouter_smoke_2026_02_18 \
  --describer-model qwen/qwen3-30b-a3b-instruct-2507 \
  --guesser-models allenai/olmo-3.1-32b-instruct mistralai/ministral-14b-2512 google/gemma-3-27b-it \
  --games-per-guesser 1 \
  --max-turns 3 \
  --save-hf-dataset
```

## 7) Target words and `high_en` / `medium_en` / `low_en`

`experiment` labels like `high_en`, `medium_en`, `low_en` are frequency buckets:
- `high_en`: high-frequency English target words
- `medium_en`: medium-frequency English target words
- `low_en`: low-frequency English target words

In Taboo, these buckets come from splitting a frequency-ranked English word list into three bins after filtering very rare words.

### 7.1 Where the target word actually lives

For `colab-potsdam/playpen-data` `interactions`:
- `WordGuesser` rows do **not** directly store `target_word`.
- The target can be recovered from:
  1. the matching `WordDescriber` prompt (it includes the literal target word), or
  2. Taboo instances keyed by `(experiment, task_id)`.

For this dataset, `(experiment, task_id)` matches Taboo `in/instances.json` (not `instances_v1.6.json`).

### 7.2 Practical mapping rule

Use:
- key = `(meta.experiment, meta.task_id)`
- lookup -> `target_word` in the corresponding Taboo instances file

This is the cleanest way to get "what word should be guessed" for each Guesser sample.

## 8) Direct file locations (URLs + local paths)

### 8.1 Playpen repo (GitHub URLs)

- CLI with learner/teacher flags and model loading:
  - https://github.com/phisad/playpen/blob/main/playpen/cli.py
- Starter trainers with Taboo role assignment (`[teacher, learner]`):
  - https://github.com/phisad/playpen/blob/main/playpen/starters/sequential_trainer.py
  - https://github.com/phisad/playpen/blob/main/playpen/starters/batch_trainer.py
  - https://github.com/phisad/playpen/blob/main/playpen/starters/branching_trainer.py
- Playpen key template:
  - https://github.com/phisad/playpen/blob/main/key.json.template
- Playpen model registry:
  - https://github.com/phisad/playpen/blob/main/model_registry.json
- Interaction-to-JSONL converter:
  - https://github.com/phisad/playpen/blob/main/examples/trl/data_utils.py

### 8.2 Clemcore repo (GitHub URLs)

- OpenRouter backend:
  - https://github.com/clp-research/clemcore/blob/main/clemcore/backends/openrouter_api.py
- OpenAI-compatible backend:
  - https://github.com/clp-research/clemcore/blob/main/clemcore/backends/openai_compatible_api.py
- OpenAI backend:
  - https://github.com/clp-research/clemcore/blob/main/clemcore/backends/openai_api.py
- Backend discovery/loading:
  - https://github.com/clp-research/clemcore/blob/main/clemcore/backends/backend_registry.py
- Key loading/lookup:
  - https://github.com/clp-research/clemcore/blob/main/clemcore/backends/key_registry.py
- Clemcore key template:
  - https://github.com/clp-research/clemcore/blob/main/key.json.template

### 8.3 This repo (local paths)

- Main manifest:
  - `runs/simple_wordguesser_v1/manifest.json`
- Base dataset splits:
  - `runs/simple_wordguesser_v1/data/train_base`
  - `runs/simple_wordguesser_v1/data/attr_query`
  - `runs/simple_wordguesser_v1/data/eval`
- OpenRouter synthetic generator:
  - `generate_synthetic_openrouter.py`
- Smoke output folder (current):
  - `runs/simple_wordguesser_v1/synthetic_openrouter_smoke_2026_02_18/summary.json`
  - `runs/simple_wordguesser_v1/synthetic_openrouter_smoke_2026_02_18/games.jsonl`
  - `runs/simple_wordguesser_v1/synthetic_openrouter_smoke_2026_02_18/rows_wordguesser.jsonl`
  - `runs/simple_wordguesser_v1/synthetic_openrouter_smoke_2026_02_18/rows_wordguesser_hf`

### 8.4 Optional: Clembench Taboo references (GitHub URLs)

- Taboo roles/game metadata:
  - https://github.com/clp-research/clembench/blob/main/taboo/clemgame.json
- Taboo game master (`Describer`/`Guesser` player mapping):
  - https://github.com/clp-research/clembench/blob/main/taboo/master.py
- Taboo instance generator (frequency buckets):
  - https://github.com/clp-research/clembench/blob/main/taboo/instancegenerator.py
- Taboo instance file used for `(experiment, task_id) -> target_word`:
  - https://github.com/clp-research/clembench/blob/main/taboo/in/instances.json
