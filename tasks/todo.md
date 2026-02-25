# Implementation Plan: Phases 3–5 + Tests

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Phase 3: Trainer System

- [x] `trainers/base.py` — Abstract `BaseTrainer`, `TrainingResult` dataclass
- [x] `trainers/lora_trainer.py` — `LoRATrainer` (PEFT + HF Trainer)
- [x] `trainers/qlora_trainer.py` — `QLoRATrainer` (4-bit quantized LoRA)
- [x] `trainers/factory.py` — `TrainerFactory` registry pattern
- [x] `trainers/__init__.py` — Public exports

## Phase 4: Evaluation System

- [x] `evaluation/metrics.py` — ROUGE, BLEU, Perplexity calculators
- [x] `evaluation/benchmarker.py` — `BenchmarkRunner`, before/after comparison
- [x] `evaluation/__init__.py` — Public exports

## Phase 5: Typer CLI

- [x] `cli/main.py` — `train`, `evaluate`, `benchmark` subcommands
- [x] `cli/__init__.py` — Exports

## Tests

- [x] `tests/test_config.py` — ConfigBuilder, Pydantic validation
- [x] `tests/test_trainers.py` — LoRATrainer unit tests (mocked HF)
- [x] `tests/test_evaluation.py` — Metrics unit tests
- [x] `tests/conftest.py` — Shared fixtures

## Task Tracking

- [x] `tasks/todo.md` (this file)
- [x] `tasks/lessons.md`

---

## Review

### What was built
- Full trainer system (LoRA, QLoRA, factory)
- Evaluation system (metrics + benchmarker)
- Typer-based CLI wiring all phases together
- Test suite with mocked dependencies

### Architecture decisions
- `BaseTrainer` uses HF `Trainer` internally; subclasses only override PEFT setup
- `QLoRATrainer` extends `LoRATrainer` — adds BitsAndBytes 4-bit config
- `BenchmarkRunner` is model-agnostic; accepts any callable generator
- CLI `train` command is thin — delegates to `ConfigBuilder` + `TrainerFactory`