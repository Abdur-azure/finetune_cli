# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 7: "CI Tight"

- [x] .github/workflows/ci.yml — install pytest-timeout, add --timeout=60 to unit step
- [x] tasks/CONTEXT.md — sprint table updated through Sprint 6
- [x] docs/index.md — version 2.4.0, 70+ tests, merge + recommend in component table
- [x] CHANGELOG.md — Sprint 7 entry
- [ ] Push to main — verify CI goes green

---

## Previously Completed

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
Push to main — GitHub Actions CI goes green on all 3 Python versions.