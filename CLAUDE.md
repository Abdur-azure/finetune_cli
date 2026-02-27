# CLAUDE.md

Context file for AI-assisted development. Read this at the start of every session.

---

## Project

**finetune-cli** — production-grade LLM fine-tuning framework with a modular CLI.
Version: 2.0.0 | License: MIT | Python: 3.10+

---

## Repo layout

```
finetune_cli/          # Main package
  core/                # Types, config, exceptions (no deps on other subpackages)
  data/                # Dataset loading, tokenization, pipeline
  models/              # Model + tokenizer loading, target module detection
  trainers/            # LoRA, QLoRA, Full, Instruction trainers + factory
  evaluation/          # ROUGE, BLEU, Perplexity metrics + BenchmarkRunner
  cli/                 # Typer CLI — train, evaluate, benchmark, upload, recommend
  utils/               # Logging only
  tests/               # Unit + integration tests

examples/
  configs/             # Four runnable YAML configs (lora, qlora, instruction, full)
  generate_sample_data.py  # Creates data/sample.jsonl + data/instructions.jsonl

docs/                  # MkDocs source (Material theme)
tasks/                 # todo.md + lessons.md — read these every session
```

---

## Architecture rules — enforce these in every change

1. **Dependency direction is one-way**: `cli → trainers/evaluation/data → models → core`. Never import upward or sideways between subpackages.
2. **`core/` has zero internal deps** — it imports only stdlib and pydantic.
3. **All config objects are frozen dataclasses** (`@dataclass(frozen=True)`). Never mutate them.
4. **All errors extend `FineTuneError`** from `core/exceptions.py`. Never raise raw builtins from module code.
5. **Always use `get_logger(__name__)`** from `utils/logging.py`. Never `logging.getLogger` directly.
6. **TrainingResult must carry `output_dir: Path`** — downstream CLI and evaluation depend on it.
7. **`trainers/__init__.py` must export every new trainer** — CLI and tests import from there.
8. **`data/__init__.py` must mirror CLI imports exactly** — trace the full import chain when adding exports.

---

## Key types (read before touching core/)

| Type | Location | Purpose |
|------|----------|---------|
| `TrainingMethod` | `core/types.py` | Enum of all training methods |
| `PipelineConfig` | `core/config.py` | Top-level Pydantic config |
| `ConfigBuilder` | `core/config.py` | Fluent builder for PipelineConfig |
| `TrainingConfig` | `core/types.py` | Frozen dataclass for training hyper-params |
| `LoRAConfig` | `core/types.py` | Frozen dataclass for LoRA params |
| `TrainingResult` | `trainers/base.py` | Frozen dataclass returned by all trainers |
| `EvaluationResult` | `evaluation/benchmarker.py` | Frozen dataclass, per-model scores |
| `BenchmarkReport` | `evaluation/benchmarker.py` | Before/after comparison |

---

## Adding a new trainer — checklist

1. Create `trainers/<name>_trainer.py`, extend `BaseTrainer`
2. Implement `_setup_peft(model) -> model`
3. Add the new `TrainingMethod` enum value to `core/types.py` if missing
4. Wire the new method in `trainers/factory.py` `TrainerFactory.create()`
5. Export from `trainers/__init__.py`
6. Write `tests/test_<name>_trainer.py` — mock HF Trainer, no GPU needed
7. Update `docs/api.md` and `docs/configuration.md`

---

## Test commands

```bash
# Unit tests (fast, no GPU)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (requires torch + transformers, CPU ok)
pytest tests/test_integration.py -v -s

# Full suite
pytest tests/ -v
```

---

## Sprint history

| Sprint | Name | Status |
|--------|------|--------|
| 1 | Stable Foundation | ✅ Complete |
| 2 | Expand | ✅ Complete |
| 3 | First Run | ✅ Complete |
| 4 | Hardened | ✅ Complete |
| 5 | Merge & Release | ✅ Complete |
| 6 | Documented | ✅ Complete |

Current task state: `tasks/todo.md`
Accumulated lessons: `tasks/lessons.md`

---

## Workflow expectations

- Read `tasks/todo.md` and `tasks/lessons.md` before starting any session
- Read `CONTEXT.md` for whichever subpackage you're touching
- Run AST parse verification before handing off any new file
- Never mark a task complete without proving it works
- Update `tasks/lessons.md` after every correction