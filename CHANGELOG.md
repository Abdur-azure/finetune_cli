# Changelog

All notable changes to this project are documented here.

---

## [2.4.0] — Sprint 6: "Documented" — 2025-02-27

### Added
- `README.md` fully rewritten for v2 — all 6 commands, 4 training methods, Python API example
- `CONTRIBUTING.md` — how to add trainers, CLI commands, sprint conventions, lessons summary
- `docs/api.md` — added `FullFineTuner`, `InstructionTrainer`, `merge`, `recommend`, `upload` sections
- `CLAUDE.md` sprint history updated through Sprint 6
- `cli/CONTEXT.md` updated to include `merge` command

---

## [2.3.0] — Sprint 5: "Merge & Release" — 2025-02-27

### Added
- `finetune-cli merge` subcommand — merges LoRA adapter into base model, saves
  standalone model runnable without PEFT installed
- `--dtype` flag (float32 | float16 | bfloat16) for merge output precision
- `tests/test_merge.py` — 8 unit tests covering happy path, error cases, dtype validation
- Upload lesson from lessons.md now fully implemented as a standalone CLI command

---

## [2.1.0] — Sprint 2: "Expand" — 2025-02-26

### Added
- `FullFineTuner` trainer — trains all parameters, issues VRAM warning for models >1B params
- `InstructionTrainer` trainer — alpaca-style `{instruction, input, response}` dataset formatting + LoRA
- `TrainerFactory` wired for `full_finetuning` and `instruction_tuning` methods
- `finetune-cli recommend` command — inspects model param count + VRAM, outputs ready-to-use YAML config
- Unit tests: `test_full_trainer.py`, `test_instruction_trainer.py`, `test_recommend.py`

### Fixed
- `cli/__init__.py` — removed stale `setup()` call that crashed pytest collection on Windows

---

## [2.2.0] — Sprint 3: "First Run" — 2025-02-26

### Added
- `examples/generate_sample_data.py` — generates `data/sample.jsonl` + `data/instructions.jsonl`, stdlib only
- `examples/configs/instruction_tuning.yaml` — new runnable config for InstructionTrainer
- `examples/configs/full_finetuning.yaml` — new runnable config for FullFineTuner
- `examples/configs/lora_gpt2.yaml` — updated to point to generated sample data
- Integration tests: `test_instruction_tuning_saves_adapter`, `test_recommend_produces_runnable_config`
- 5-minute quickstart section in `docs/usage.md`
- `CLAUDE.md` at repo root — session context for AI-assisted development
- `CONTEXT.md` in every key subpackage folder

### Fixed
- `InstructionTrainer._maybe_format()` — skip reformatting when dataset already has `input_ids` column
- `test_full_trainer.py` VRAM warning test — replaced real tensor allocation with `MagicMock.numel()`

---

## [2.0.0] — 2025-02-26

### Summary
Complete rewrite from a monolithic interactive script (v1) to a production-grade
modular CLI framework (v2). Breaking change — v1 `finetune_cli.py` is deprecated.

---

### Migration Guide — v1 → v2

#### Running the tool

```bash
# v1 (interactive prompts)
python finetune_cli.py

# v2 (CLI subcommands)
python -m finetune_cli.cli train --model gpt2 --dataset ./data.jsonl
python -m finetune_cli.cli train --config examples/configs/lora_gpt2.yaml

# Or install for the shell command
pip install -e .
finetune-cli train --model gpt2 --dataset ./data.jsonl
```

#### Training

```bash
# v1
python finetune_cli.py
# → interactive: enter model name, dataset path, LoRA params one by one

# v2 — flags
finetune-cli train \
  --model gpt2 \
  --dataset ./data.jsonl \
  --lora-r 8 \
  --epochs 3 \
  --output ./output

# v2 — config file (recommended for reproducibility)
finetune-cli train --config examples/configs/lora_gpt2.yaml
```

#### Evaluation / Benchmarking

```bash
# v1 — benchmark was embedded in the training wizard

# v2 — dedicated subcommands
finetune-cli evaluate --model-path ./output --dataset ./test.jsonl
finetune-cli benchmark gpt2 ./output --dataset ./test.jsonl --metrics rougeL,bleu
```

#### Uploading to HuggingFace

```bash
# v1
python finetune_cli.py
# → interactive: step 8 "Upload to HuggingFace?"

# v2
finetune-cli upload ./output username/my-model --token $HF_TOKEN

# v2 — merge LoRA adapter before uploading
finetune-cli upload ./output username/my-model \
  --merge-adapter \
  --base-model gpt2
```

---

### What's New in v2

#### Architecture
- Fully modular: `core/`, `models/`, `data/`, `trainers/`, `evaluation/`, `cli/`
- All config objects are Pydantic-validated and YAML/JSON serialisable
- Frozen dataclasses for all result types — no accidental mutation
- Custom exception hierarchy (`FineTuneError` → specific subclasses)
- Structured logging via `utils/logging.py`

#### Data Pipeline (`data/`)
- `quick_load()` — one-liner for experiments
- `prepare_dataset()` — full pipeline with validation splits
- `DataPipeline` class — reusable, stateful pipeline with statistics
- Auto-detection of text columns across JSON, JSONL, CSV, Parquet, TXT
- HuggingFace Hub loading with streaming support

#### Trainers (`trainers/`)
- `LoRATrainer` — PEFT LoRA with auto-detected target modules
- `QLoRATrainer` — 4-bit quantized LoRA via BitsAndBytes
- `TrainerFactory` — single entry point: `TrainerFactory.train(...)`
- `BaseTrainer` — composable base; subclasses only override PEFT setup

#### Evaluation (`evaluation/`)
- `RougeMetric`, `BleuMetric`, `PerplexityMetric`
- `BenchmarkRunner.evaluate()` — single-model scoring
- `BenchmarkRunner.run_comparison()` — before/after delta report
- `BenchmarkReport.summary()` — formatted table with ▲/▼ indicators

#### CLI (`cli/`)
- `train` — supports `--config` file or individual flags
- `evaluate` — score a saved checkpoint
- `benchmark` — compare base vs fine-tuned
- `upload` — push to HuggingFace Hub with optional adapter merge

#### Testing
- 35+ unit tests — all pass with mocked HF dependencies (no GPU needed)
- Integration test — real GPT-2, 1 step, asserts `config.json` saved
- `conftest.py` at repo root — pytest path fix for Windows/macOS/Linux
- CI workflow — pytest matrix across Python 3.10/3.11/3.12 + ruff + mypy

---

### Removed in v2
- Interactive multi-step wizard (`finetune_cli.py` main loop)
- Hardcoded training parameters
- Raw `print()` logging
- `setup.py` (replaced by `pyproject.toml`)

---

## [1.0.0] — 2025-01-27

### Initial release
- Monolithic `finetune_cli.py` with interactive 8-step wizard
- LoRA, QLoRA, AdaLoRA, Prefix Tuning, P-Tuning
- ROUGE/BLEU benchmarking
- HuggingFace Hub upload
- MkDocs documentation site