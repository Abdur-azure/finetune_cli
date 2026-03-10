# Contributing to xlmtec

Thank you for your interest in xlmtec — a production-grade CLI for fine-tuning
Large Language Models with LoRA, QLoRA, DPO, distillation, pruning, and more.

All contributions are welcome: bug fixes, new trainers, new CLI commands,
documentation improvements, and test coverage.

---

## Quick start

```bash
git clone https://github.com/Abdur-azure/xlmtec.git
cd xlmtec
pip install -e ".[ml,dev]"
pytest tests/ -v
```

---

## What you can contribute

| Area | Examples |
|------|---------|
| **New training method** | ORPO, SimPO, spectrum fine-tuning |
| **New export backend** | TensorRT, Core ML, ExecuTorch |
| **New notifier** | Discord, Teams, PagerDuty |
| **New inference feature** | streaming output, token probabilities |
| **Bug fix** | anything in the issue tracker |
| **Docs** | guides, examples, API reference |
| **Tests** | increasing coverage on undertested modules |

---

## Development workflow

### 1. Fork and branch

```bash
git checkout -b feat/my-feature      # new feature
git checkout -b fix/issue-123        # bug fix
git checkout -b docs/improve-sweep   # docs only
```

### 2. Follow the module pattern

Each subsystem lives in its own package (`xlmtec/<subsystem>/`):

```
xlmtec/<subsystem>/
  __init__.py      # docstring only — never re-export here (circular import risk)
  <module>.py      # implementation
  CONTEXT.md       # what this package does, rules, extension pattern
```

When adding a new trainer:
1. Create `xlmtec/trainers/<name>_trainer.py` extending `BaseTrainer`
2. Register it in `xlmtec/trainers/factory.py` (the registry dict)
3. Add the `TrainingMethod` enum value in `xlmtec/core/types.py`
4. Write tests in `tests/test_<name>_trainer.py` (mock model + tokenizer)
5. Update `xlmtec/trainers/CONTEXT.md`

### 3. Code style

```bash
ruff check xlmtec/ tests/    # lint
ruff format xlmtec/ tests/   # format
mypy xlmtec/                 # type-check
```

- Line length: 100
- No top-level `torch` / `transformers` imports in any module (keep install fast)
- Heavy optional deps must be imported **inside the function** that needs them,
  always **after** any `dry_run` early return

### 4. Tests

```bash
pytest tests/ -v                          # all tests
pytest tests/test_sweep.py -v             # one module
pytest tests/ --cov=xlmtec --cov-report=term-missing
```

- All new code needs tests
- Use `unittest.mock` — never download real models in unit tests
- Optional deps (optuna, trl, etc.) must be mocked via `patch.dict("sys.modules", ...)`

### 5. Open a pull request

- Target: `main`
- Title format: `feat: add ORPO trainer` / `fix: sweep dry-run returns wrong code`
- Fill in the PR template — description, test evidence, checklist

---

## Commit message convention

```
feat: add feature distillation trainer
fix: correct warmup_ratio field name in TrainingConfig
docs: add hyperparameter sweep guide
test: add 8 tests for SweepRunner
refactor: extract ConfigBuilder from config.py
chore: bump version to 3.28.0
```

---

## Reporting a bug

Use the [bug report template](https://github.com/Abdur-azure/xlmtec/issues/new?template=bug_report.yml).

Please include:
- xlmtec version (`xlmtec --version`)
- Python version
- OS
- Full traceback
- Minimal reproduction (config file + dataset sample if possible)

---

## Requesting a feature

Use the [feature request template](https://github.com/Abdur-azure/xlmtec/issues/new?template=feature_request.yml).

---

## Code of conduct

Be respectful. Constructive feedback only. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## License

By contributing, you agree your changes will be licensed under the
[MIT License](../LICENSE) that covers this project.