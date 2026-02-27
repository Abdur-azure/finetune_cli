# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 8: "DPO"

- [x] trainers/dpo_trainer.py — DPOTrainer + validate_dpo_dataset
- [x] trainers/factory.py — DPO wired
- [x] trainers/__init__.py — DPOTrainer exported
- [x] cli/main.py — DPO added to _LORA_METHODS
- [x] examples/configs/dpo.yaml — runnable example config
- [x] tests/test_dpo_trainer.py — 10 unit tests
- [x] tests/test_cli_train.py — test_dpo_via_flags added
- [x] CHANGELOG.md + audit_repo.py updated
- [ ] Run: pytest tests/test_dpo_trainer.py -v

---

## Previously Completed

### Sprint 7: "CI Tight"
- [x] ci.yml paths, pytest-timeout, ruff config, absolute imports in tests

### Sprint 6: "Documented"
- [x] README, CONTRIBUTING, docs/api.md, CLAUDE.md, cli/CONTEXT.md

### Sprint 5: "Merge & Release"
- [x] finetune-cli merge command + test_merge.py (8 tests)

### Sprint 4: "Hardened"
- [x] CLI lora guard, test_cli_train.py, CHANGELOG, audit_repo

### Sprint 3: "First Run"
- [x] Examples, sample data, integration tests, CLAUDE.md, CONTEXT.md files

### Sprint 2: "Expand"
- [x] FullFineTuner, InstructionTrainer, recommend command, 60 tests

### Sprint 1: "Stable Foundation"
- [x] All tests green, CI, pyproject.toml, docs

---

## Acceptance Gate
pytest tests/test_dpo_trainer.py -v — all 10 tests green.