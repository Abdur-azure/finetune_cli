# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 11: "Version Sync"

- [x] pyproject.toml — version 2.0.0 → 2.8.0
- [x] tasks/CONTEXT.md — Sprint 9 + 10 rows added
- [x] CLAUDE.md — sprint history through Sprint 10
- [x] docs/index.md — version 2.8.0
- [x] CONTRIBUTING.md — Sprint-end checklist section added
- [x] CHANGELOG.md — Sprint 11 entry

---

## Previously Completed

### Sprint 10: "DPO Runnable"
- [x] dpo_sample.jsonl generator, local config, trl optional dep, docs/configuration.md

### Sprint 9: "Housekeeping"
- [x] CONTEXT.md, CLAUDE.md, docs/index.md, trainers/CONTEXT.md, docs/api.md synced

### Sprint 8: "DPO"
- [x] DPOTrainer, validate_dpo_dataset, factory, tests, example config

### Sprint 7: "CI Tight"
- [x] ci.yml paths, pytest-timeout, ruff lint, absolute imports

### Sprint 6: "Documented"
- [x] README, CONTRIBUTING, docs/api.md, CLAUDE.md, cli/CONTEXT.md

### Sprint 5: "Merge & Release"
- [x] finetune-cli merge command + test_merge.py

### Sprint 4: "Hardened"
- [x] CLI lora guard, test_cli_train.py, CHANGELOG, audit_repo

### Sprint 3: "First Run"
- [x] Examples, sample data, integration tests, CLAUDE.md, CONTEXT.md files

### Sprint 2: "Expand"
- [x] FullFineTuner, InstructionTrainer, recommend, 60 tests

### Sprint 1: "Stable Foundation"
- [x] All tests green, CI, pyproject.toml, docs

---

## Acceptance Gate
No code changed. Review: python audit_repo.py and pip show finetune-cli → 2.8.0.