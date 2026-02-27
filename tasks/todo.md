# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 5: "Merge & Release"

- [x] cli/main.py — merge subcommand (local adapter → standalone model)
- [x] tests/test_merge.py — 8 unit tests
- [x] CHANGELOG.md — Sprint 5 entry
- [x] audit_repo.py — test_merge.py added
- [x] lessons.md — f-string Panel pattern + unactioned lessons pattern
- [ ] Run: pytest tests/test_merge.py -v

---

## Previously Completed

### Sprint 4: "Hardened"
- [x] cli/main.py lora guard, test_cli_train.py, CHANGELOG, audit_repo

### Sprint 3: "First Run"
- [x] Examples, sample data, integration tests, CLAUDE.md, CONTEXT.md files

### Sprint 2: "Expand"
- [x] FullFineTuner, InstructionTrainer, recommend command, 60 tests

### Sprint 1: "Stable Foundation"
- [x] All tests green, CI, pyproject.toml, docs

---

## Acceptance Gate
pytest tests/test_merge.py -v — all 8 tests green.